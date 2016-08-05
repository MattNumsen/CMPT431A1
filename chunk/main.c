#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
//#define CHUNK_DEBUG
//#define DEBUG
struct arg_struct {
    int nsize;
    int first;
};
#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}
pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
// Current number of threads
volatile int numThreads;
// Minimum rows to process per thread
volatile int minimumRowsPerThread = 1;
// Flag to launch threads waiting on the start barrier
volatile int threadsGo;
volatile int threadsDone;
volatile int globalRowStart;

int end_barrier ()
{
    int last = 0;
    pthread_mutex_lock (&mut);	//lock

    threadsDone++;
    if (threadsDone == numThreads) {
        last = 1;
        threadsGo = 0; // every thread has completed work so clear flag
        threadsDone = 0;
        pthread_cond_broadcast(&cond);
    } else {
        pthread_cond_wait(&cond, &mut);
    }
    pthread_mutex_unlock (&mut);	//unlock

    return last;
}

start_barrier(int first, int totalRows, int currentNorm, int *startRow, int *endRow)
{
    pthread_mutex_lock(&mut);

    if (first) {
        // We are the first thread - launch the others
        threadsGo = 1;
        globalRowStart = currentNorm + 1;
        pthread_cond_broadcast(&cond);
    } else {
        // Not the first thread, just wait until we have to start
        while (!threadsGo) {
            pthread_cond_wait(&cond, &mut);
        }
    }
    // Determine what portion of rows we shall process
    int rowsRemain = totalRows - globalRowStart;
    if (!rowsRemain) {
        // No more work as we are near the end 
        *startRow = *endRow = totalRows;
    } else {
        // This code calculates the chunk per thread in the current iteration
        int perThread = (totalRows - currentNorm - 1) / numThreads;
        int remainder = (totalRows - currentNorm - 1) % numThreads;
        if (!perThread) {
            perThread = 1;
        }
        // If we have a constraint about minimum rows per thread
        perThread = MAX(perThread, minimumRowsPerThread);
        int numRowsToProcess = MIN(rowsRemain, perThread);
        rowsRemain -= numRowsToProcess;
        if (first && remainder && (rowsRemain >= remainder)) {
            // Process the remainder
            numRowsToProcess += remainder;
        }
        *startRow = globalRowStart;
        *endRow = *startRow + numRowsToProcess;
        globalRowStart += numRowsToProcess;
    }
#ifdef CHUNK_DEBUG
    printf("We shall process from %d to %d\n", *startRow, *endRow);
#endif
    pthread_mutex_unlock(&mut);
}
/* Solve the equation:
 *   matrix * X = R
 */
double **matrix, *X, *R;
/* Pre-set solution. */
double *X__;
/* Initialize the matirx. */

int initMatrix(const char *fname)
{
    FILE *file;
    int l1, l2, l3;
    double d;
    int nsize;
    int i, j;
    double *tmp;
    char buffer[1024];

    if ((file = fopen(fname, "r")) == NULL) {
	fprintf(stderr, "The matrix file open error\n");
        exit(-1);
    }

    /* Parse the first line to get the matrix size. */
    fgets(buffer, 1024, file);
    sscanf(buffer, "%d %d %d", &l1, &l2, &l3);
    nsize = l1;
#ifdef DEBUG
    fprintf(stdout, "matrix size is %d\n", nsize);
#endif

    /* Initialize the space and set all elements to zero. */
    matrix = (double**)malloc(nsize*sizeof(double*));
    assert(matrix != NULL);
    tmp = (double*)malloc(nsize*nsize*sizeof(double));
    assert(tmp != NULL);
    for (i = 0; i < nsize; i++) {
        matrix[i] = tmp;
        tmp = tmp + nsize;
    }
    for (i = 0; i < nsize; i++) {
        for (j = 0; j < nsize; j++) {
            matrix[i][j] = 0.0;
        }
    }

    /* Parse the rest of the input file to fill the matrix. */
    for (;;) {
	fgets(buffer, 1024, file);
	sscanf(buffer, "%d %d %lf", &l1, &l2, &d);
	if (l1 == 0) break;

	matrix[l1-1][l2-1] = d;
#ifdef DEBUG
	fprintf(stdout, "row %d column %d of matrix is %e\n", l1-1, l2-1, matrix[l1-1][l2-1]);
#endif
    }

    fclose(file);
    return nsize;
}

/* Initialize the right-hand-side following the pre-set solution. */

void initRHS(int nsize)
{
    int i, j;

    X__ = (double*)malloc(nsize * sizeof(double));
    assert(X__ != NULL);
    for (i = 0; i < nsize; i++) {
	X__[i] = i+1;
    }

    R = (double*)malloc(nsize * sizeof(double));
    assert(R != NULL);
    for (i = 0; i < nsize; i++) {
	R[i] = 0.0;
	for (j = 0; j < nsize; j++) {
	    R[i] += matrix[i][j] * X__[j];
	}
    }
}

/* Initialize the results. */

void initResult(int nsize)
{
    int i;

    X = (double*)malloc(nsize * sizeof(double));
    assert(X != NULL);
    for (i = 0; i < nsize; i++) {
	X[i] = 0.0;
    }
}

/* Get the pivot - the element on column with largest absolute value. */

void getPivot(int nsize, int currow)
{
    int i, pivotrow;

    pivotrow = currow;
    for (i = currow+1; i < nsize; i++) {
	if (fabs(matrix[i][currow]) > fabs(matrix[pivotrow][currow])) {
	    pivotrow = i;
	}
    }

    if (fabs(matrix[pivotrow][currow]) == 0.0) {
        fprintf(stderr, "The matrix is singular\n");
        exit(-1);
    }

    if (pivotrow != currow) {
#ifdef DEBUG
	fprintf(stdout, "pivot row at step %5d is %5d\n", currow, pivotrow);
#endif
        for (i = currow; i < nsize; i++) {
            SWAP(matrix[pivotrow][i],matrix[currow][i]);
        }
        SWAP(R[pivotrow],R[currow]);
    }
}

void *computeGauss(void *data)
{
    int i, j, k;
    double pivotval;
    int fromRow = 0;
    int toRow = 0;
    int nsize;
    int first;
    struct arg_struct *args = (struct arg_struct *)data;

    nsize = args->nsize;
    first = args->first;

    for (i = 0; i < nsize; i++) {
        if (first) { // Only one thread shall determine the pivot
#ifdef CHUNK_DEBUG
            printf("We are the first thread at iteration %d\n", i);
#endif
            getPivot(nsize, i);
            //once you get the pivot
            //scale that row in a thread
            /* Scale the main row. */
            //this code just gets the pivot
            pivotval = matrix[i][i];

            //if the pivot isn't 1..make it 1
            if (pivotval != 1.0) {
                matrix[i][i] = 1.0;
                for (j = i + 1; j < nsize; j++) {
                    matrix[i][j] /= pivotval;
                }
                R[i] /= pivotval;
            }
        }
        // Get the needed range, and wait if necessary for the pivot thread
        start_barrier(first, nsize, i, &fromRow, &toRow);

        if (fromRow != toRow) {
            //once pivot is known we can row reduce the rest of matrix
            /* Factorize the rest of the matrix. */
            for (j = fromRow; j < toRow; j++) {
                pivotval = matrix[j][i];
                matrix[j][i] = 0.0;
                for (k = i + 1; k < nsize; k++) {
                    matrix[j][k] -= pivotval * matrix[i][k];
                }
                R[j] -= pivotval * R[i];
            }
        }

        // Wait for all threads to finish work then Last thread to finish becomes first to find pivot of next iteration
        first = end_barrier();
    }

    return NULL;
}
/* Solve the equation. */
void solveGauss(int nsize)
{
    int i, j;

    X[nsize-1] = R[nsize-1];
    for (i = nsize - 2; i >= 0; i --) {
        X[i] = R[i];
        for (j = nsize - 1; j > i; j--) {
            X[i] -= matrix[i][j] * X[j];
        }
    }

#ifdef DEBUG
    fprintf(stdout, "X = [");
    for (i = 0; i < nsize; i++) {
        fprintf(stdout, "%.6f ", X[i]);
    }
    fprintf(stdout, "];\n");
#endif
}
int main(int argc, char *argv[])
{
    int i;
    struct timeval start, finish;
    int nsize = 0;
    double error;
    if (argc < 2) {
	fprintf(stderr, "usage: %s <matrixfile> [num threads] [min rows per thread (def 1)]\n", argv[0]);
	exit(-1);
    }
    numThreads = 1;
    if (argc > 2){
        numThreads = atoi(argv[2]);
    }

    minimumRowsPerThread = 1;
    if (argc > 3) {
        minimumRowsPerThread = atoi(argv[3]);
    }

    nsize = initMatrix(argv[1]);
    initRHS(nsize);
    initResult(nsize);

    pthread_t threads[numThreads];
    printf("N size: %d \n", nsize);
    printf("Num Threads: %d \n", numThreads);

    gettimeofday(&start, 0);

    if (numThreads > 0){
        pthread_t threads[numThreads];
        struct arg_struct argsFirst, argsRest;
        int error_code;
        argsFirst.nsize = nsize;
        argsFirst.first = 1;
        argsRest.nsize = nsize;
        argsRest.first = 0;

        void * argsCurrent = &argsFirst;
        int i = 0;

        for (i = 0; i < numThreads; i++) {

#ifdef CHUNK_DEBUG
            printf("Creating thread number %ld\n", i);
#endif

            if (i > 0) {
                argsCurrent = &argsRest;
            }
            error_code = pthread_create(&threads[i], NULL, computeGauss , argsCurrent);
            if (error_code){
                printf("error code pthread_create(): %d\n", error_code);
                exit(-1);
            }
        }
        for (i = 0; i < numThreads; i++) {
            pthread_join(threads[i], NULL);
        }
    }

    gettimeofday(&finish, 0);

    solveGauss(nsize);

    fprintf(stdout, "Time:  %f seconds\n", (finish.tv_sec - start.tv_sec) + (finish.tv_usec - start.tv_usec)*0.000001);

    error = 0.0;
    for (i = 0; i < nsize; i++) {
	double error__ = (X__[i]==0.0) ? 1.0 : fabs((X[i]-X__[i])/X__[i]);
	if (error < error__) {
	    error = error__;
	}
    }
    fprintf(stdout, "Error: %e\n", error);
    pthread_exit(NULL);

    return 0;
}
