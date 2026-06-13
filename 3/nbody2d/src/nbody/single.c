#include "ppp/ppp.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>

static long double deltaT;
static body *bodies;
static int nBodies;

/*
 * Compute the acceleration body j exercises on body i.
 * The acceleration is returned in *ax and *ay.
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
 * Update the position of each body for the next time step from its velocity
 * and its acceleration in the current time step.
 */
static void update_positions() {
    for (int i = 0; i < nBodies; ++i) {
        bodies[i].x += (bodies[i].vx + 0.5 * bodies[i].ax * deltaT) * deltaT;
        bodies[i].y += (bodies[i].vy + 0.5 * bodies[i].ay * deltaT) * deltaT;
    }
}

/*
 * Compute the accelerations in the next time step from the positions of the
 * bodies in the current time step. The accelerations in the next time step
 * are returned in accelsX and accelsY.
 */
static void compute_accelerations(long double *accelsX, long double *accelsY) {
    for (int i = 0; i < nBodies; ++i) {
        accelsX[i] = 0;
        accelsY[i] = 0;
        for (int j = 0; j < nBodies; ++j) {
            if (i != j) {
                acceleration(i, j, accelsX, accelsY);
            }
        }
    }
}

/*
 * Update the velocity of each body for the next time step from its
 * acceleration in the current time step and its acceleration in the next time
 * step. The accelerations for the current time step are in bodies[i].ax and
 * bodies[i].ay; the accelerations for the next time step are in accelsX[i]
 * and accelsY[i]. Also copy the accelerations from accelsX/Y[i] to
 * bodies[i].ax/ay to update the acceleration information for the next time
 * step, too.
 */
static void update_velocities(long double *accelsX, long double *accelsY) {
    for (int i = 0; i < nBodies; i++) {
        bodies[i].vx += 0.5 * (bodies[i].ax + accelsX[i]) * deltaT;
        bodies[i].vy += 0.5 * (bodies[i].ay + accelsY[i]) * deltaT;
        bodies[i].ax = accelsX[i];
        bodies[i].ay = accelsY[i];
    }
}

void compute_single(struct TaskInput *TI) {
    const bool debug = TI->debug;
    const int nSteps = TI->nSteps;
    const int imageStep = TI->imageStep;
    deltaT = TI->deltaT;
    nBodies = TI->nBodies;
    bodies = TI->bodies;

    // Accelerations (in x- and y-directions) of the bodies
    // in the next time step. (Accelerations in the current time
    // step are stored in the body structs.)
    long double accelsX[nBodies];
    long double accelsY[nBodies];

    // Preparation: compute accelerations in time step 0 and store them in
    // the bodies array.
    compute_accelerations(accelsX, accelsY);
    for (int i = 0; i < nBodies; ++i) {
        bodies[i].ax = accelsX[i];
        bodies[i].ay = accelsY[i];
    }

    for (int step = 0; step < nSteps; ++step) {
        if (imageStep > 0 && step % imageStep == 0) {
            saveImage(step / imageStep, bodies, nBodies);
        }

        if (debug) {
            printf("Time step %d\n", step);
        }

        // Compute positions for next time step.
        update_positions();

        // Compute accelerations for next time step (returned in accelsX
        // and accelsY).
        compute_accelerations(accelsX, accelsY);

        // Compute the velocities for the next time step using the
        // accelerations in the current time step (in the body structs)
        // and the next time step (in accelsX and accelsY).
        update_velocities(accelsX, accelsY);
    }

    if (imageStep > 0 && nSteps % imageStep == 0) {
        saveImage(nSteps / imageStep, bodies, nBodies);
    }
}
