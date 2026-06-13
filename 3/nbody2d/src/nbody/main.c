#include "mpi.h"
#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Skip comments ("# ...") in ".dat" files.
 */
static void skipComments(FILE *f) {
    int n;
    do {
        n = 0;
        fscanf(f, " #%n", &n);
        if (n > 0) {
            fscanf(f, "%*[^\n]");
            fscanf(f, "\n");
        }
    } while (n > 0);
}

/*
 * Read a ".dat" file containing the description of the bodies
 * (see project sheet for format description).
 *    f: file handle to read from
 *    n: output parameter for the number of bodies read
 * The bodies are returned in an array of "body" structs.
 * In case of error, NULL is returned.
 */
static body *readBodies(FILE *f, int *nBodies) {
    body *bodies;

    skipComments(f);
    if (fscanf(f, " %d", nBodies) != 1) {
        return NULL;
    }
    bodies = (body *)calloc(*nBodies, sizeof(body));
    if (bodies == NULL) {
        return NULL;
    }

    for (int i = 0; i < *nBodies; i++) {
        long double mass, x, y, vx, vy;
        skipComments(f);
        int conv = fscanf(f, " %Lf %Lf %Lf %Lf %Lf", &mass, &x, &y, &vx, &vy);
        bodies[i].id = i;
        bodies[i].mass = mass;
        bodies[i].x = x;
        bodies[i].y = y;
        bodies[i].vx = vx;
        bodies[i].vy = vy;
        if (conv != 5) {
            free(bodies);
            return NULL;
        }
    }
    return bodies;
}

static int cmpByID(const void *a, const void *b) {
    const body *bodyA = (const body *)a;
    const body *bodyB = (const body *)b;
    return bodyA->id - bodyB->id;
}

/*
 * Write 'n' bodies from the array 'bodies' to the file
 * identified by file handle 'f' in ".dat" format.
 */
static void writeBodies(FILE *f, body *bodies, int nBodies) {
    qsort(bodies, nBodies, sizeof(*bodies), cmpByID);
    fprintf(f, "%d\n", nBodies);
    for (int i = 0; i < nBodies; i++) {
        fprintf(f, "% 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg % 10.4Lg\n", (long double)bodies[i].mass,
                (long double)bodies[i].x, (long double)bodies[i].y, (long double)bodies[i].vx,
                (long double)bodies[i].vy);
    }
}

static void usage(const char *progname) {
    fprintf(stderr,
            "USAGE: %s -i input.dat [-o output.dat] [-t step] [-n n_steps]\n"
            "  [-d] [-p] [-3] [-l] [-a] [-W pixels] [-H pixels]\n"
            "  [-w width] [-h width] [-A asteps]\n"
            "   step     time step in seconds\n"
            "   n_steps  number of time steps\n"
            "   pixels   image width/height\n"
            "   width    width of depicted space in meters\n"
            "   height   heigt of depicted space in meters\n"
            "   asteps   number of steps between two images\n"
            "   -p       parallel execution\n"
            "   -3       use Newton's third law globally\n"
            "   -l       use Newton's third law in local computations\n"
            "   -a       use approximation with big surrogate bodies\n"
            "   -d       show some debugging output\n",
            progname);
}

/*
 * Compute the energy (kinetic and potential) and momentum in the system of
 * bodies. Energy is returned in *energy, the momentum vector in *px and *py.
 */
static void momentumAndEnergy(body *bodies, int nBodies, long double *energy, long double *px,
                              long double *py) {
    *energy = 0;
    *px = 0;
    *py = 0;
    for (int i = 0; i < nBodies; ++i) {
        long double velocitySquared = bodies[i].vx * bodies[i].vx + bodies[i].vy * bodies[i].vy;

        // Kinetic energy
        *energy += bodies[i].mass / 2 * velocitySquared;

        // Potential energy (one for every pair of objects). Following the usual
        // convention, objects have 0 potential energy at infinite distance.
        for (int j = 0; j < i; ++j) {
            long double dist = hypotl(bodies[i].x - bodies[j].x, bodies[i].y - bodies[j].y);
            *energy -= G * bodies[i].mass * bodies[j].mass / dist;
        }

        // Momentum
        *px += bodies[i].mass * bodies[i].vx;
        *py += bodies[i].mass * bodies[i].vy;
    }
}

static int imageWidth, imageHeight;
static double width, height;
static const char *imageFilePrefix;

/*
 * Save an image with suffix 'imageNum'.
 */
void saveImage(int imageNum, const body *bodies, int nBodies) {
    uint8_t *img = (uint8_t *)malloc(sizeof(uint8_t) * imageWidth * imageHeight);
    char name[strlen(imageFilePrefix) + 11];

    if (img == NULL) {
        return;
    }

    sprintf(name, "%s-%05d.pbm", imageFilePrefix, imageNum);
    for (int i = 0; i < imageWidth * imageHeight; ++i) {
        img[i] = 0;
    }

    for (int i = 0; i < nBodies; ++i) {
        int x = imageWidth / 2 + (int)roundl(bodies[i].x * imageWidth / width);
        int y = imageHeight / 2 - (int)roundl(bodies[i].y * imageHeight / height);

        if (x >= 0 && x < imageWidth && y >= 0 && y < imageHeight) {
            img[y * imageWidth + x] = 1;
        }
    }

    ppp_pnm_write(name, PNM_KIND_PBM, imageHeight, imageWidth, 1, img);
    free(img);
}

int main(int argc, char *argv[]) {
    FILE *f;
    char *filename, *outfilename;
    bool parallel;
    int retCode;

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }
    int self, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    filename = NULL;
    outfilename = NULL;
    parallel = false;
    imageWidth = 100;
    imageHeight = 100;
    width = 1;
    height = 1;

    struct TaskInput TI;
    TI.nSteps = 0;
    TI.deltaT = 1;
    TI.newton3 = false;
    TI.newton3local = false;
    TI.approxSurrogate = false;
    TI.imageStep = 0;
    TI.debug = false;

    /* Compute a 2-dimensional arrangement for the processors
     * (px processors in the x-dimension, py processors in the y-dimension)
     * such that px * py equals np and py is maximal but py <= px;
     */
    for (int py = 1; py * py <= np; py++) {
        if (np % py == 0) {
            TI.px = np / py;
            TI.py = py;
        }
    }

    int option;
    while ((option = getopt(argc, argv, "i:o:t:n:A:w:h:W:H:p3lad")) != -1) {
        switch (option) {
        case 'i':
            filename = strdup(optarg);
            break;
        case 'o':
            outfilename = strdup(optarg);
            break;
        case 't':
            TI.deltaT = atof(optarg);
            break;
        case 'n':
            TI.nSteps = atoi(optarg);
            break;
        case 'A':
            TI.imageStep = atoi(optarg);
            break;
        case 'w':
            width = atof(optarg);
            break;
        case 'h':
            height = atof(optarg);
            break;
        case 'W':
            imageWidth = atoi(optarg);
            break;
        case 'H':
            imageHeight = atoi(optarg);
            break;
        case 'p':
            parallel = true;
            break;
        case '3':
            TI.newton3 = true;
            break;
        case 'l':
            TI.newton3local = true;
            break;
        case 'a':
            TI.approxSurrogate = true;
            break;
        case 'd':
            TI.debug = true;
            break;
        default:
            usage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    if (filename == NULL || (TI.imageStep > 0 && outfilename == NULL)) {
        if (self == 0) {
            usage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    imageFilePrefix = outfilename;

    retCode = 0;
    f = fopen(filename, "r");
    if (f != NULL) {
        int nBodies;
        body *bodies = readBodies(f, &nBodies);
        TI.nBodies = nBodies;
        TI.bodies = bodies;
        fclose(f);
        if (bodies != NULL) {
            long double energyStart, pxStart, pyStart;
            momentumAndEnergy(bodies, nBodies, &energyStart, &pxStart, &pyStart);
            double simTime = seconds();
            if (parallel) {
                compute_parallel(&TI);
            } else if (self == 0) {
                compute_single(&TI);
            }
            simTime = seconds() - simTime;
            long double energyEnd, pxEnd, pyEnd;
            momentumAndEnergy(bodies, nBodies, &energyEnd, &pxEnd, &pyEnd);
            if (self == 0) {
                if (outfilename != NULL) {
                    f = fopen(outfilename, "w");
                    if (f != NULL) {
                        writeBodies(f, bodies, nBodies);
                        fclose(f);
                    } else {
                        fprintf(stderr, "Could not open '%s' for writing.", outfilename);
                    }
                } else
                    writeBodies(stdout, bodies, nBodies);
                free(bodies);
                long double energyRel = (energyEnd - energyStart) / energyStart;
                long double pAbs = hypotl(pxStart, pyStart);
                long double pxRel = (pxEnd - pxStart) / pAbs;
                long double pyRel = (pyEnd - pyStart) / pAbs;
                double interactionRate =
                    (double)nBodies * (double)(nBodies - 1) * (double)TI.nSteps / simTime;
                const char *pMsg =
                    fabsl(pxRel) <= 1e-6 && fabsl(pyRel) <= 1e-6 ? "OK" : "deviation";
                printf("Simulation time: %.6g s\n"
                       "Interaction rate: %g s^-1\n"
                       "Energy: %Lg vs. %Lg, relative change %Lg\n"
                       "Momentum: (%Lg,%Lg) vs. (%Lg,%Lg), relative change (%Lg,%Lg): %s\n",
                       simTime, interactionRate, energyStart, energyEnd, energyRel, pxStart,
                       pyStart, pxEnd, pyEnd, pxRel, pyRel, pMsg);
            }
        } else {
            fprintf(stderr, "Error reading data form file '%s'.\n", filename);
            retCode = 1;
        }
    } else {
        fprintf(stderr, "Error opening file '%s'.\n", filename);
        retCode = 1;
    }

    MPI_Finalize();
    return retCode;
}
