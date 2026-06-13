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
        // implementation with Newton's third law used globally
    } else if (TI->newton3local) {
        if (self == 0) {
            printf("Running with Newton's third law locally.\n");
        }
        // implementation with Newton's third law for
        // local computations
    } else if (TI->approxSurrogate) {
        if (self == 0) {
            printf("Running simulation with %d x %d surrogate bodies.\n", TI->px, TI->py);
        }
        // Master's task:
        // implementation with big surrogate bodies for bodies in other processes
    } else {
        if (self == 0) {
            printf("Running without Newton's third law.\n");
        }
        // implementation without Newton's third law here
    }
}
