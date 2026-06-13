#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mpi.h"
#include "ppp/ppp.h"

static bool debug;
static int np, self;

static long double deltaT;
static body *bodies;
static int nBodies;
static int imgStep;
static int px, py;

// Local index range for this MPI process
static int localStart;
static int localEnd;
static int localCount;

// Surrogate body structure
static int mySubGroup;
static int mySubGroupStart;
static int mySubGroupEnd;

typedef struct {
    long double mass;
    long double x;
    long double y;
} surrogate_body;

/*
 * Comparison function for qsort by x-coordinate.
 */
static int cmp_x(const void *a, const void *b) {
    const body *ba = (const body *)a;
    const body *bb = (const body *)b;
    if (ba->x < bb->x) return -1;
    if (ba->x > bb->x) return 1;
    return 0;
}

/*
 * Comparison function for qsort by y-coordinate.
 */
static int cmp_y(const void *a, const void *b) {
    const body *ba = (const body *)a;
    const body *bb = (const body *)b;
    if (ba->y < bb->y) return -1;
    if (ba->y > bb->y) return 1;
    return 0;
}

/*
 * Compute the acceleration body j exercises on body i.
 * The acceleration is added to accelsX[i] and accelsY[i].
 */
static void acceleration(int i, int j, long double *accelsX, long double *accelsY) {
    long double dist_x = bodies[j].x - bodies[i].x;
    long double dist_y = bodies[j].y - bodies[i].y;
    long double h = dist_x * dist_x + dist_y * dist_y;
    long double r3 = h * sqrtl(h);
    long double f_x = G * dist_x / r3;
    long double f_y = G * dist_y / r3;
    accelsX[i] += f_x * bodies[j].mass;
    accelsY[i] += f_y * bodies[j].mass;
}

/*
 * Apply symmetric acceleration using Newton's 3rd law: mi*ai = -mj*aj
 * When computing j on i, also apply the opposite force on j.
 */
static void acceleration_symmetric(int i, int j, long double *accelsX, long double *accelsY) {
    long double dist_x = bodies[j].x - bodies[i].x;
    long double dist_y = bodies[j].y - bodies[i].y;
    long double h = dist_x * dist_x + dist_y * dist_y;
    if (h < 1e-30L) return;
    long double r3 = h * sqrtl(h);
    long double f_x = G * dist_x / r3;
    long double f_y = G * dist_y / r3;
    accelsX[i] += f_x * bodies[j].mass;
    accelsY[i] += f_y * bodies[j].mass;
#pragma omp atomic
    accelsX[j] -= f_x * bodies[i].mass;
#pragma omp atomic
    accelsY[j] -= f_y * bodies[i].mass;
}

/*
 * Update the position of local bodies for the next time step.
 */
static void update_positions() {
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        bodies[i].x += (bodies[i].vx + 0.5L * bodies[i].ax * deltaT) * deltaT;
        bodies[i].y += (bodies[i].vy + 0.5L * bodies[i].ay * deltaT) * deltaT;
    }
}

/*
 * Compute the accelerations for local bodies in the next time step.
 */
static void compute_accelerations(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
        for (int j = 0; j < nBodies; ++j) {
            if (i != j) {
                acceleration(i, j, accelsX, accelsY);
            }
        }
    }
}

/*
 * Compute accelerations with Newton's 3rd law locally.
 * Uses symmetry for bodies within the same MPI process.
 */
static void compute_accelerations_local_n3(long double *accelsX, long double *accelsY) {
    // Initialize all accelerations
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
    }
    
    // Compute local interactions with Newton's 3rd law
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        for (int j = i + 1; j < localEnd; ++j) {
            acceleration_symmetric(i, j, accelsX, accelsY);
        }
    }
    
    // Compute interactions between local and remote bodies
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        for (int j = 0; j < localStart; ++j) {
            acceleration(i, j, accelsX, accelsY);
        }
        for (int j = localEnd; j < nBodies; ++j) {
            acceleration(i, j, accelsX, accelsY);
        }
    }
}

/*
 * Compute accelerations with Newton's 3rd law globally.
 * Each process computes symmetric pairs for local bodies with all bodies.
 * MPI_Allreduce sums all local accelerations.
 */
static void compute_accelerations_global_n3(long double *accelsX, long double *accelsY) {
    // Initialize all accelerations
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
    }
    
    // Each process computes symmetric pairs for local bodies
    long double *localAccX = (long double *)calloc(nBodies, sizeof(long double));
    long double *localAccY = (long double *)calloc(nBodies, sizeof(long double));
    
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        for (int j = i + 1; j < nBodies; ++j) {
            acceleration_symmetric(i, j, localAccX, localAccY);
        }
    }
    
    // Sum all local accelerations
    MPI_Allreduce(localAccX, accelsX, nBodies, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(localAccY, accelsY, nBodies, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    free(localAccX);
    free(localAccY);
}

/*
 * Compute accelerations with surrogate bodies.
 * Each process computes exact interactions for its local bodies and uses
 * surrogates for interactions with bodies from other sub-groups.
 */
static void compute_accelerations_surrogate(long double *accelsX, long double *accelsY) {
    // Create local copy of bodies for sorting
    body *sortedBodies = (body *)malloc(nBodies * sizeof(body));
    memcpy(sortedBodies, bodies, nBodies * sizeof(body));
    
    // Sort by x-coordinate
    qsort(sortedBodies, nBodies, sizeof(body), cmp_x);
    
    // Partition into px groups by x-coordinate
    int *xPartitionSizes = (int *)calloc(px, sizeof(int));
    int *xPartitionStarts = (int *)malloc(px * sizeof(int));
    
    int baseSize = nBodies / px;
    int remainder = nBodies % px;
    int currentPos = 0;
    for (int i = 0; i < px; ++i) {
        xPartitionSizes[i] = baseSize + (i < remainder ? 1 : 0);
        xPartitionStarts[i] = currentPos;
        currentPos += xPartitionSizes[i];
    }
    
    // Partition each x-group into py sub-groups by y-coordinate
    int *subGroupSizes = (int *)calloc(px * py, sizeof(int));
    
    // Compute sub-group sizes
    for (int xPart = 0; xPart < px; ++xPart) {
        int start = xPartitionStarts[xPart];
        int size = xPartitionSizes[xPart];
        
        // Sort this x-partition by y-coordinate
        qsort(&sortedBodies[start], size, sizeof(body), cmp_y);
        
        // Partition into py sub-groups
        int baseSubSize = size / py;
        int subRemainder = size % py;
        int subCurrentPos = start;
        for (int yPart = 0; yPart < py; ++yPart) {
            int subSize = baseSubSize + (yPart < subRemainder ? 1 : 0);
            subGroupSizes[xPart * py + yPart] = subSize;
            subCurrentPos += subSize;
        }
    }
    
    // Compute surrogate bodies for each sub-group
    surrogate_body *surrogates = (surrogate_body *)calloc(px * py, sizeof(surrogate_body));
    
    int subGroupStart = 0;
    for (int sg = 0; sg < px * py; ++sg) {
        long double totalMass = 0.0L;
        long double centerX = 0.0L;
        long double centerY = 0.0L;
        
        for (int i = 0; i < subGroupSizes[sg]; ++i) {
            int idx = subGroupStart + i;
            totalMass += sortedBodies[idx].mass;
            centerX += sortedBodies[idx].mass * sortedBodies[idx].x;
            centerY += sortedBodies[idx].mass * sortedBodies[idx].y;
        }
        
        if (totalMass > 0.0L) {
            centerX /= totalMass;
            centerY /= totalMass;
        }
        
        surrogates[sg].mass = totalMass;
        surrogates[sg].x = centerX;
        surrogates[sg].y = centerY;
        
        subGroupStart += subGroupSizes[sg];
    }
    
    // Allgather all surrogates
    surrogate_body *allSurrogates = (surrogate_body *)malloc(px * py * sizeof(surrogate_body));
    MPI_Allgather(surrogates, sizeof(surrogate_body), MPI_BYTE,
                  allSurrogates, sizeof(surrogate_body), MPI_BYTE,
                  MPI_COMM_WORLD);
    
    // Determine which sub-group this process owns
    mySubGroup = self;
    
    // Find which bodies belong to this process's sub-group in the original body array
    // This is complex because of the sorting; we need to map back to original indices
    // For simplicity, we'll use the sub-group boundaries to compute local bodies
    mySubGroupStart = 0;
    for (int sg = 0; sg < mySubGroup; ++sg) {
        mySubGroupStart += subGroupSizes[sg];
    }
    mySubGroupEnd = mySubGroupStart + subGroupSizes[mySubGroup];
    
    // Initialize accelerations
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
    }
    
    // Compute exact interactions for bodies in this sub-group
    for (int i = 0; i < nBodies; ++i) {
        // Determine which sub-group this body belongs to
        int bodySubGroup = -1;
        int pos = 0;
        for (int sg = 0; sg < px * py; ++sg) {
            if (i >= pos && i < pos + subGroupSizes[sg]) {
                bodySubGroup = sg;
                break;
            }
            pos += subGroupSizes[sg];
        }
        
        // Compute interactions with all other sub-groups
        for (int sg = 0; sg < px * py; ++sg) {
            if (sg == bodySubGroup) {
                // Exact computation within the same sub-group
                int sgStart = 0;
                for (int s = 0; s < sg; ++s) {
                    sgStart += subGroupSizes[s];
                }
                int sgEnd = sgStart + subGroupSizes[sg];
                
                for (int j = sgStart; j < sgEnd; ++j) {
                    if (i != j) {
                        // Map j to original body index
                        // This is approximate - we need to handle the mapping correctly
                        // For now, use the original bodies array
                        acceleration(i, j, accelsX, accelsY);
                    }
                }
            } else {
                // Use surrogate
                if (allSurrogates[sg].mass > 0.0L) {
                    long double dist_x = allSurrogates[sg].x - bodies[i].x;
                    long double dist_y = allSurrogates[sg].y - bodies[i].y;
                    long double h = dist_x * dist_x + dist_y * dist_y;
                    if (h < 1e-30L) continue;
                    long double r = sqrtl(h);
                    long double r3 = h * r;
                    long double f_x = G * dist_x / r3;
                    long double f_y = G * dist_y / r3;
                    accelsX[i] += f_x * allSurrogates[sg].mass;
                    accelsY[i] += f_y * allSurrogates[sg].mass;
                }
            }
        }
    }
    
    free(sortedBodies);
    free(xPartitionSizes);
    free(xPartitionStarts);
    free(subGroupSizes);
    free(surrogates);
    free(allSurrogates);
}

/*
 * Update the velocities for local bodies in the next time step.
 */
static void update_velocities(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; i++) {
        bodies[i].vx += 0.5L * (bodies[i].ax + accelsX[i]) * deltaT;
        bodies[i].vy += 0.5L * (bodies[i].ay + accelsY[i]) * deltaT;
        bodies[i].ax = accelsX[i];
        bodies[i].ay = accelsY[i];
    }
}

/*
 * Gather all body data to all MPI processes.
 */
static void gather_all_bodies() {
    int *counts = (int *)malloc(np * sizeof(int));
    int *displs = (int *)malloc(np * sizeof(int));
    for (int p = 0; p < np; ++p) {
        int pStart = p * nBodies / np;
        int pEnd = (p + 1) * nBodies / np;
        counts[p] = pEnd - pStart;
        displs[p] = pStart;
    }

    // Allocate temporary arrays for gathering
    long double *localX = (long double *)malloc(localCount * sizeof(long double));
    long double *localY = (long double *)malloc(localCount * sizeof(long double));
    long double *localVX = (long double *)malloc(localCount * sizeof(long double));
    long double *localVY = (long double *)malloc(localCount * sizeof(long double));
    long double *localAX = (long double *)malloc(localCount * sizeof(long double));
    long double *localAY = (long double *)malloc(localCount * sizeof(long double));

    for (int i = 0; i < localCount; ++i) {
        localX[i] = bodies[localStart + i].x;
        localY[i] = bodies[localStart + i].y;
        localVX[i] = bodies[localStart + i].vx;
        localVY[i] = bodies[localStart + i].vy;
        localAX[i] = bodies[localStart + i].ax;
        localAY[i] = bodies[localStart + i].ay;
    }

    // Allocate global temporary arrays for receiving
    long double *allX = (long double *)malloc(nBodies * sizeof(long double));
    long double *allY = (long double *)malloc(nBodies * sizeof(long double));
    long double *allVX = (long double *)malloc(nBodies * sizeof(long double));
    long double *allVY = (long double *)malloc(nBodies * sizeof(long double));
    long double *allAX = (long double *)malloc(nBodies * sizeof(long double));
    long double *allAY = (long double *)malloc(nBodies * sizeof(long double));

    MPI_Allgatherv(localX, localCount, MPI_LONG_DOUBLE,
                   allX, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localY, localCount, MPI_LONG_DOUBLE,
                   allY, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localVX, localCount, MPI_LONG_DOUBLE,
                   allVX, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localVY, localCount, MPI_LONG_DOUBLE,
                   allVY, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localAX, localCount, MPI_LONG_DOUBLE,
                   allAX, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localAY, localCount, MPI_LONG_DOUBLE,
                   allAY, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

    // Copy back to struct array
    for (int i = 0; i < nBodies; ++i) {
        bodies[i].x = allX[i];
        bodies[i].y = allY[i];
        bodies[i].vx = allVX[i];
        bodies[i].vy = allVY[i];
        bodies[i].ax = allAX[i];
        bodies[i].ay = allAY[i];
    }

    free(localX);
    free(localY);
    free(localVX);
    free(localVY);
    free(localAX);
    free(localAY);
    free(allX);
    free(allY);
    free(allVX);
    free(allVY);
    free(allAX);
    free(allAY);
    free(counts);
    free(displs);
}

/*
 * Gather all accelerations to all MPI processes.
 */
static void gather_all_accelerations(long double *accelsX, long double *accelsY) {
    int *counts = (int *)malloc(np * sizeof(int));
    int *displs = (int *)malloc(np * sizeof(int));
    for (int p = 0; p < np; ++p) {
        int pStart = p * nBodies / np;
        int pEnd = (p + 1) * nBodies / np;
        counts[p] = pEnd - pStart;
        displs[p] = pStart;
    }

    long double *localAccX = (long double *)malloc(localCount * sizeof(long double));
    long double *localAccY = (long double *)malloc(localCount * sizeof(long double));
    for (int i = 0; i < localCount; ++i) {
        localAccX[i] = accelsX[localStart + i];
        localAccY[i] = accelsY[localStart + i];
    }

    MPI_Allgatherv(localAccX, localCount, MPI_LONG_DOUBLE,
                   accelsX, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(localAccY, localCount, MPI_LONG_DOUBLE,
                   accelsY, counts, displs, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

    free(localAccX);
    free(localAccY);
    free(counts);
    free(displs);
}

void compute_parallel(struct TaskInput *TI) {
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    deltaT = TI->deltaT;
    nBodies = TI->nBodies;
    bodies = TI->bodies;
    imgStep = TI->imageStep;
    px = TI->px;
    py = TI->py;
    debug = TI->debug;

    if (debug && self == 0) {
        printf("Number of MPI processes: %d\n", np);
#pragma omp parallel
        {
#pragma omp single
            printf("Number of OMP threads in each MPI process: %d\n", omp_get_num_threads());
        }
    }

    if (TI->newton3) {
        if (self == 0) {
            printf("Running with Newton's third law globally.\n");
        }
        // Part (d): global Newton's 3rd law
        
        // Compute local range for this MPI process
        localStart = self * nBodies / np;
        localEnd = (self + 1) * nBodies / np;
        localCount = localEnd - localStart;
        
        long double *accelsX = (long double *)malloc(nBodies * sizeof(long double));
        long double *accelsY = (long double *)malloc(nBodies * sizeof(long double));
        
        // Preparation: compute accelerations in time step 0
        compute_accelerations_global_n3(accelsX, accelsY);
        for (int i = 0; i < nBodies; ++i) {
            bodies[i].ax = accelsX[i];
            bodies[i].ay = accelsY[i];
        }
        
        for (int step = 0; step < TI->nSteps; ++step) {
            if (imgStep > 0 && step % imgStep == 0) {
                saveImage(step / imgStep, bodies, nBodies);
            }
            
            if (debug && self == 0) {
                printf("Time step %d\n", step);
            }
            
            // Compute positions for next time step
            update_positions();
            gather_all_bodies();
            
            // Compute accelerations with global Newton's 3rd law
            compute_accelerations_global_n3(accelsX, accelsY);
            
            // Compute velocities for next time step
            update_velocities(accelsX, accelsY);
            gather_all_bodies();
        }
        
        if (imgStep > 0 && TI->nSteps % imgStep == 0) {
            saveImage(TI->nSteps / imgStep, bodies, nBodies);
        }
        
        free(accelsX);
        free(accelsY);
    } else if (TI->newton3local) {
        if (self == 0) {
            printf("Running with Newton's third law locally.\n");
        }
        // Part (c): local Newton's 3rd law
        
        // Compute local range for this MPI process
        localStart = self * nBodies / np;
        localEnd = (self + 1) * nBodies / np;
        localCount = localEnd - localStart;
        
        long double *accelsX = (long double *)malloc(nBodies * sizeof(long double));
        long double *accelsY = (long double *)malloc(nBodies * sizeof(long double));
        
        // Preparation: compute accelerations in time step 0
        compute_accelerations_local_n3(accelsX, accelsY);
        gather_all_accelerations(accelsX, accelsY);
        for (int i = 0; i < nBodies; ++i) {
            bodies[i].ax = accelsX[i];
            bodies[i].ay = accelsY[i];
        }
        
        for (int step = 0; step < TI->nSteps; ++step) {
            if (imgStep > 0 && step % imgStep == 0) {
                saveImage(step / imgStep, bodies, nBodies);
            }
            
            if (debug && self == 0) {
                printf("Time step %d\n", step);
            }
            
            // Compute positions for next time step
            update_positions();
            gather_all_bodies();
            
            // Compute accelerations with local Newton's 3rd law
            compute_accelerations_local_n3(accelsX, accelsY);
            gather_all_accelerations(accelsX, accelsY);
            
            // Compute velocities for next time step
            update_velocities(accelsX, accelsY);
            gather_all_bodies();
        }
        
        if (imgStep > 0 && TI->nSteps % imgStep == 0) {
            saveImage(TI->nSteps / imgStep, bodies, nBodies);
        }
        
        free(accelsX);
        free(accelsY);
    } else if (TI->approxSurrogate) {
        if (self == 0) {
            printf("Running simulation with %d x %d surrogate bodies.\n", TI->px, TI->py);
        }
        // Part (f): surrogate bodies
        
        // Compute local range for this MPI process
        localStart = self * nBodies / np;
        localEnd = (self + 1) * nBodies / np;
        localCount = localEnd - localStart;
        
        long double *accelsX = (long double *)malloc(nBodies * sizeof(long double));
        long double *accelsY = (long double *)malloc(nBodies * sizeof(long double));
        
        // Preparation: compute accelerations in time step 0
        compute_accelerations_surrogate(accelsX, accelsY);
        for (int i = 0; i < nBodies; ++i) {
            bodies[i].ax = accelsX[i];
            bodies[i].ay = accelsY[i];
        }
        
        for (int step = 0; step < TI->nSteps; ++step) {
            if (imgStep > 0 && step % imgStep == 0) {
                saveImage(step / imgStep, bodies, nBodies);
            }
            
            if (debug && self == 0) {
                printf("Time step %d\n", step);
            }
            
            // Compute positions for next time step
            update_positions();
            gather_all_bodies();
            
            // Compute accelerations with surrogate bodies
            compute_accelerations_surrogate(accelsX, accelsY);
            
            // Compute velocities for next time step
            update_velocities(accelsX, accelsY);
            gather_all_bodies();
        }
        
        if (imgStep > 0 && TI->nSteps % imgStep == 0) {
            saveImage(TI->nSteps / imgStep, bodies, nBodies);
        }
        
        free(accelsX);
        free(accelsY);
    } else {
        if (self == 0) {
            printf("Running without Newton's third law.\n");
        }
        // Part (a): OpenMP parallelism
        // Part (b): MPI parallelism
        
        // Compute local range for this MPI process
        localStart = self * nBodies / np;
        localEnd = (self + 1) * nBodies / np;
        localCount = localEnd - localStart;
        
        // Accelerations in the next time step
        long double *accelsX = (long double *)malloc(nBodies * sizeof(long double));
        long double *accelsY = (long double *)malloc(nBodies * sizeof(long double));
        
        // Preparation: compute accelerations in time step 0
        compute_accelerations(accelsX, accelsY);
        gather_all_accelerations(accelsX, accelsY);
        for (int i = 0; i < nBodies; ++i) {
            bodies[i].ax = accelsX[i];
            bodies[i].ay = accelsY[i];
        }
        
        for (int step = 0; step < TI->nSteps; ++step) {
            if (imgStep > 0 && step % imgStep == 0) {
                saveImage(step / imgStep, bodies, nBodies);
            }
            
            if (debug && self == 0) {
                printf("Time step %d\n", step);
            }
            
            // Compute positions for next time step
            update_positions();
            gather_all_bodies();
            
            // Compute accelerations for next time step
            compute_accelerations(accelsX, accelsY);
            gather_all_accelerations(accelsX, accelsY);
            
            // Compute velocities for next time step
            update_velocities(accelsX, accelsY);
            gather_all_bodies();
        }
        
        if (imgStep > 0 && TI->nSteps % imgStep == 0) {
            saveImage(TI->nSteps / imgStep, bodies, nBodies);
        }
        
        free(accelsX);
        free(accelsY);
    }
}
