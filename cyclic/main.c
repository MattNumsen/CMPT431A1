/*	Parallel version of gauss.c as provided by Arrvindh Shriramen for CMPT 431 - FALL2015
 * 	Modified by Matt Numsen
 */
#ifndef _REENTRANT
#define _REENTRANT
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>

#define SWAP(a,b)       {double tmp; tmp = a; a = b; b = tmp;}


/* Solve the equation:
 *   matrix * X = R
 */

double **matrix, *X, *R;

/* Pre-set solution. */

double *X__;

/* Initialize the matirx. */

    int numThreads = 1; //default value for numThreads is 1, global variable used for barrier
/* Barrier function used for pthread timing */
pthread_mutex_t mut = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
void
barrier (int expect)
{
    static int arrived = 0;

    pthread_mutex_lock (&mut);	//lock

    arrived++;
    if (arrived < expect)
        pthread_cond_wait (&cond, &mut);
    else {
        arrived = 0;		// reset the barrier before broadcast is important
        pthread_cond_broadcast (&cond);
    }

    pthread_mutex_unlock (&mut);	//unlock
}

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

/* For all the rows, get the pivot and eliminate all rows and columns
 * for that particular pivot row. */
struct scaleRowArgs {
	int id_arg;
	int nsize_arg;
};

void *computeGauss(void *arguments)
{

    struct scaleRowArgs *args = (struct scaleRowArgs*)arguments;
	int id = args->id_arg;
	int nsize = args->nsize_arg;
    int i, j, k;
    double pivotval;

    for (i = 0; i < nsize; i++) {
       if (id == 0){
	      getPivot(nsize,i);
	      /* Scale the main row.  */
            pivotval = matrix[i][i];
            if (pivotval != 1.0) {
	           matrix[i][i] = 1.0;
   	           for (j = i + 1; j < nsize; j++) {
                  matrix[i][j] /= pivotval;
	           }
               R[i] /= pivotval;
	       }
	       barrier(numThreads);
	   } else {
	      barrier(numThreads);
	   }

	/* Factorize the rest of the matrix. */
        for (j = i + id + 1; j < nsize; j= j + numThreads) {
            pivotval = matrix[j][i];
            matrix[j][i] = 0.0;
            for (k = i + 1; k < nsize; k++) {
                matrix[j][k] -= pivotval * matrix[i][k];
            }
            R[j] -= pivotval * R[i];
        }
    }
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
    int i, q;
    struct timeval start, finish;
    int nsize = 0;
    double error;
    pthread_attr_t attr;
    pthread_t *tid;
    struct scaleRowArgs *args;

    if (argc != 3) {
	fprintf(stderr, "usage: %s <matrixfile> <threadCount>\n", argv[0]);
	exit(-1);
    }

    numThreads = atoi (argv[2]);
    nsize = initMatrix(argv[1]);
    initRHS(nsize);
    initResult(nsize);
    gettimeofday(&start, 0);
/* 	This is the function that should be parallellized
	The Parallelization strategy will be to split the rows of the Matrix into however many threads exist.
*/
    args = (struct scaleRowArgs *) malloc (sizeof (struct scaleRowArgs) * numThreads);
    tid = (pthread_t *) malloc (sizeof (pthread_t) * numThreads);


    pthread_attr_init (&attr);
    pthread_attr_setscope (&attr, PTHREAD_SCOPE_SYSTEM);

    for (q = 1; q < numThreads; q++) {
       args[q].id_arg = q;
	   args[q].nsize_arg = nsize;
       pthread_create (&tid[q], &attr, computeGauss, (void *)&args[q]);
    }
    args[0].id_arg = 0;
    args[0].nsize_arg = nsize;
    computeGauss(&args[0]);
    for (i = 1; i < numThreads; i++)
        pthread_join (tid[i], NULL);

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

    return 0;
}
