#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>

#include "mpi.h"
#include "ppp/ppp.h"

static bool debug;
static int np, self;

static long double deltaT;
static body *bodies;
static int nBodies;
static int imgStep;
static int px, py;

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
 * Update the position of each body for the next time step.
 */
static void update_positions() {
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
        bodies[i].x += (bodies[i].vx + 0.5L * bodies[i].ax * deltaT) * deltaT;
        bodies[i].y += (bodies[i].vy + 0.5L * bodies[i].ay * deltaT) * deltaT;
    }
}

/*
 * Compute the accelerations for the next time step.
 */
static void compute_accelerations(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = 0; i < nBodies; ++i) {
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
 * Update the velocities for the next time step.
 */
static void update_velocities(long double *accelsX, long double *accelsY) {
    #pragma omp parallel for
    for (int i = 0; i < nBodies; i++) {
        bodies[i].vx += 0.5L * (bodies[i].ax + accelsX[i]) * deltaT;
        bodies[i].vy += 0.5L * (bodies[i].ay + accelsY[i]) * deltaT;
        bodies[i].ax = accelsX[i];
        bodies[i].ay = accelsY[i];
    }
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
    } else if (TI->newton3local) {
        if (self == 0) {
            printf("Running with Newton's third law locally.\n");
        }
        // Part (c): local Newton's 3rd law
    } else if (TI->approxSurrogate) {
        if (self == 0) {
            printf("Running simulation with %d x %d surrogate bodies.\n", TI->px, TI->py);
        }
        // Part (f): surrogate bodies
    } else {
        if (self == 0) {
            printf("Running without Newton's third law.\n");
        }
        // Part (a): OpenMP parallelism
        // Part (b): MPI parallelism
        
        // Accelerations in the next time step
        long double *accelsX = (long double *)malloc(nBodies * sizeof(long double));
        long double *accelsY = (long double *)malloc(nBodies * sizeof(long double));
        
        // Preparation: compute accelerations in time step 0
        compute_accelerations(accelsX, accelsY);
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
            
            // Compute accelerations for next time step
            compute_accelerations(accelsX, accelsY);
            
            // Compute velocities for next time step
            update_velocities(accelsX, accelsY);
        }
        
        if (imgStep > 0 && TI->nSteps % imgStep == 0) {
            saveImage(TI->nSteps / imgStep, bodies, nBodies);
        }
        
        free(accelsX);
        free(accelsY);
    }
}
