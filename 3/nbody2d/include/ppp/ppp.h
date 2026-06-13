#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>

static const long double G = 6.674e-11;

// Representation of bodies
typedef struct {
    int id;             /* unique identifier, do not modify */
    long double mass;   /* in kilograms */
    long double x, y;   /* position in meters */
    long double vx, vy; /* velocity in meters per second */
    long double ax, ay; /* acceleration in meters per second squared */
} body;

struct TaskInput {
    // the number of bodies
    int nBodies;

    // the bodies (also used for output)
    body *bodies;

    // whether to use Newton's third law for local computations
    bool newton3local;

    // whether to use Newton's third law globally
    bool newton3;

    // whether to use approximation with big surrogate bodies
    bool approxSurrogate;

    // number of simulation steps to perform
    int nSteps;

    // the length of a time step of the simulation (in seconds)
    long double deltaT;

    // number of steps between two images;
    // 0 means no image output
    int imageStep;

    // two-dimensional arrangement of processes for
    // computation with surrogate bodies
    int px;
    int py;

    // print some debug outputs during computation
    bool debug;
};

void compute_single(struct TaskInput *TI);
void compute_parallel(struct TaskInput *TI);

// Save a snapshot of the bodies as image number 'imgNum'
void saveImage(int imgNum, const body *bodies, int nBodies);

// Returns the number of seconds since 1970-01-01T00:00:00.
inline static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000;
}
