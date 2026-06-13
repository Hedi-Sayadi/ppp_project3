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

static int localStart;
static int localEnd;
static int localCount;

static inline void acceleration_pair(int i, int j, const body *allBodies,
                                      long double *ax, long double *ay) {
    long double dist_x = allBodies[j].x - allBodies[i].x;
    long double dist_y = allBodies[j].y - allBodies[i].y;
    long double h = dist_x * dist_x + dist_y * dist_y;
    if (h < 1e-30L) return;
    long double r = sqrtl(h);
    long double r3 = h * r;
    long double f_x = G * dist_x / r3;
    long double f_y = G * dist_y / r3;
    ax[i] += f_x * allBodies[j].mass;
    ay[i] += f_y * allBodies[j].mass;
}

static inline void acceleration_symmetric(int i, int j, const body *allBodies,
                                          long double *ax, long double *ay) {
    long double dist_x = allBodies[j].x - allBodies[i].x;
    long double dist_y = allBodies[j].y - allBodies[i].y;
    long double h = dist_x * dist_x + dist_y * dist_y;
    if (h < 1e-30L) return;
    long double r = sqrtl(h);
    long double r3 = h * r;
    long double f_x = G * dist_x / r3;
    long double f_y = G * dist_y / r3;
    ax[i] += f_x * allBodies[j].mass;
    ay[i] += f_y * allBodies[j].mass;
    ax[j] -= f_x * allBodies[i].mass;
    ay[j] -= f_y * allBodies[i].mass;
}

static void update_positions() {
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        bodies[i].x += (bodies[i].vx + 0.5L * bodies[i].ax * deltaT) * deltaT;
        bodies[i].y += (bodies[i].vy + 0.5L * bodies[i].ay * deltaT) * deltaT;
    }
}

static void compute_accelerations_no_n3(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
        for (int j = 0; j < nBodies; ++j) {
            if (i != j) {
                acceleration_pair(i, j, bodies, accelsX, accelsY);
            }
        }
    }
}

static void compute_accelerations_local_n3(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
    }
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        for (int j = i + 1; j < localEnd; ++j) {
            acceleration_symmetric(i, j, bodies, accelsX, accelsY);
        }
    }
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        for (int j = 0; j < localStart; ++j) {
            acceleration_pair(i, j, bodies, accelsX, accelsY);
        }
        for (int j = localEnd; j < nBodies; ++j) {
            acceleration_pair(i, j, bodies, accelsX, accelsY);
        }
    }
}

static void compute_accelerations_global_n3(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
    }
    long double *localAccX = (long double *)calloc(nBodies, sizeof(long double));
    long double *localAccY = (long double *)calloc(nBodies, sizeof(long double));
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        for (int j = i + 1; j < nBodies; ++j) {
            acceleration_symmetric(i, j, bodies, localAccX, localAccY);
        }
    }
    MPI_Allreduce(localAccX, accelsX, nBodies, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(localAccY, accelsY, nBodies, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    free(localAccX);
    free(localAccY);
}

static void compute_accelerations_surrogate(long double *accelsX, long double *accelsY) {
    long double myMass = 0.0L;
    long double myCenterX = 0.0L;
    long double myCenterY = 0.0L;
    for (int i = localStart; i < localEnd; ++i) {
        myMass += bodies[i].mass;
        myCenterX += bodies[i].mass * bodies[i].x;
        myCenterY += bodies[i].mass * bodies[i].y;
    }
    if (myMass > 0.0L) {
        myCenterX /= myMass;
        myCenterY /= myMass;
    }
    long double *allSurrogateMasses = (long double *)malloc(np * sizeof(long double));
    long double *allSurrogateX = (long double *)malloc(np * sizeof(long double));
    long double *allSurrogateY = (long double *)malloc(np * sizeof(long double));
    MPI_Allgather(&myMass, 1, MPI_LONG_DOUBLE, allSurrogateMasses, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&myCenterX, 1, MPI_LONG_DOUBLE, allSurrogateX, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgather(&myCenterY, 1, MPI_LONG_DOUBLE, allSurrogateY, 1, MPI_LONG_DOUBLE, MPI_COMM_WORLD);

    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0.0L;
        accelsY[i] = 0.0L;
    }

    for (int i = 0; i < nBodies; ++i) {
        for (int p = 0; p < np; ++p) {
            int pStart = p * nBodies / np;
            int pEnd = (p + 1) * nBodies / np;
            if (p == self) {
                for (int j = pStart; j < pEnd; ++j) {
                    if (i != j) {
                        acceleration_pair(i, j, bodies, accelsX, accelsY);
                    }
                }
            } else {
                if (allSurrogateMasses[p] > 0.0L) {
                    long double dist_x = allSurrogateX[p] - bodies[i].x;
                    long double dist_y = allSurrogateY[p] - bodies[i].y;
                    long double h = dist_x * dist_x + dist_y * dist_y;
                    if (h < 1e-30L) continue;
                    long double r = sqrtl(h);
                    long double r3 = h * r;
                    long double f_x = G * dist_x / r3;
                    long double f_y = G * dist_y / r3;
                    accelsX[i] += f_x * allSurrogateMasses[p];
                    accelsY[i] += f_y * allSurrogateMasses[p];
                }
            }
        }
    }
    free(allSurrogateMasses);
    free(allSurrogateX);
    free(allSurrogateY);
}

static void update_velocities(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = localStart; i < localEnd; ++i) {
        bodies[i].vx += 0.5L * (bodies[i].ax + accelsX[i]) * deltaT;
        bodies[i].vy += 0.5L * (bodies[i].ay + accelsY[i]) * deltaT;
        bodies[i].ax = accelsX[i];
        bodies[i].ay = accelsY[i];
    }
}

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

    localStart = self * nBodies / np;
    localEnd = (self + 1) * nBodies / np;
    localCount = localEnd - localStart;

    long double *accelsX = (long double *)malloc(nBodies * sizeof(long double));
    long double *accelsY = (long double *)malloc(nBodies * sizeof(long double));

    if (TI->newton3) {
        if (self == 0) {
            printf("Running with Newton's third law globally.\n");
        }
        compute_accelerations_global_n3(accelsX, accelsY);
    } else if (TI->newton3local) {
        if (self == 0) {
            printf("Running with Newton's third law locally.\n");
        }
        compute_accelerations_local_n3(accelsX, accelsY);
    } else if (TI->approxSurrogate) {
        if (self == 0) {
            printf("Running simulation with %d x %d surrogate bodies.\n", TI->px, TI->py);
        }
        compute_accelerations_surrogate(accelsX, accelsY);
    } else {
        if (self == 0) {
            printf("Running without Newton's third law.\n");
        }
        compute_accelerations_no_n3(accelsX, accelsY);
    }

    gather_all_bodies();

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

        update_positions();
        gather_all_bodies();

        if (TI->newton3) {
            compute_accelerations_global_n3(accelsX, accelsY);
        } else if (TI->newton3local) {
            compute_accelerations_local_n3(accelsX, accelsY);
        } else if (TI->approxSurrogate) {
            compute_accelerations_surrogate(accelsX, accelsY);
        } else {
            compute_accelerations_no_n3(accelsX, accelsY);
        }

        if (!TI->newton3) {
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

        update_velocities(accelsX, accelsY);
        gather_all_bodies();
    }

    if (imgStep > 0 && TI->nSteps % imgStep == 0) {
        saveImage(TI->nSteps / imgStep, bodies, nBodies);
    }

    free(accelsX);
    free(accelsY);
}
