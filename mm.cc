#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <x86intrin.h>

#include "timer.c"

#define N_ 4096
#define K_ 4096
#define M_ 4096

#define MIN(a,b) ( ( a < b) ? a : b )

typedef double dtype;

void verify(dtype *C, dtype *C_ans, int N, int M)
{
  int i, cnt;
  cnt = 0;
  for(i = 0; i < N * M; i++) {
    if(abs (C[i] - C_ans[i]) > 1e-6) cnt++;
  }
  if(cnt != 0) printf("ERROR\n"); else printf("SUCCESS\n");
}

// naive
void mm_serial (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
  int i, j, k;
  for(int i = 0; i < N; i++) {
    for(int j = 0; j < M; j++) {
      for(int k = 0; k < K; k++) {
        C[i * M + j] += A[i * K + k] * B[k * M + j];
      }
    }
  }
}

// cache-blocked matrix-matrix multiply
void mm_cb (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{
	int b = 256;	// block size
	int i, j, k;
	int j_inner, k_inner;
	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j += b) {
			for(int k = 0; k < K; k += b) {

				// iterate through the blocks
				for (j_inner = j; j_inner < MIN(j + b, M); j_inner++) {
					for (k_inner = k; k_inner < MIN(k + b, K); k_inner++) {
						C[i * M + j_inner] += A[i * K + k_inner] * B[k_inner * M + j_inner];
					}
				}
			}
		}
	}
}

// SIMD-vectorized matrix-matrix multiply
void mm_sv (dtype *C, dtype *A, dtype *B, int N, int K, int M)
{

	int b = 256;	// block size
	int i, j, k;
	int j_inner, k_inner;
	
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j += b) {
			for(int k = 0; k < K; k += b) {

				__m128d Avec, Bvec, Cvec, mult_vec; 

				// iterate through the blocks
				for (j_inner = j; j_inner < MIN(j + b, M); j_inner++) {
					for (k_inner = k; k_inner < MIN(k + b, K); k_inner += 2) {

						// have this if statement if to make sure that it only does SIMD if it can do so without accessing outside the array
						// HOWEVER, this did not work
						if ((i * K + k_inner) < (N*K)-1 && (k_inner * M + j_inner) < (K*M)-1) {
							Avec = _mm_load_pd(A + (i * K + k_inner));
							Bvec = _mm_load_pd(B + (k_inner * M + j_inner));
							Cvec = _mm_mul_pd(Avec,Bvec);

							double c[2];
							_mm_store_pd(c, Cvec);
							C[i * M + j_inner] += c[0] + c[1];
						} 
						else {
							C[i * M + j_inner] += A[i * K + k_inner] * B[k_inner * M + j_inner];
						}
					}
				}
			}
		}
	}
}

int main(int argc, char** argv)
{
  int i, j, k;
  int N, K, M;

  if(argc == 4) {
    N = atoi (argv[1]);		
    K = atoi (argv[2]);		
    M = atoi (argv[3]);		
    printf("N: %d K: %d M: %d\n", N, K, M);
  } else {
    N = N_;
    K = K_;
    M = M_;
    printf("N: %d K: %d M: %d\n", N, K, M);	
  }

  dtype *A = (dtype*) malloc (N * K * sizeof (dtype));
  dtype *B = (dtype*) malloc (K * M * sizeof (dtype));
  dtype *C = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_cb = (dtype*) malloc (N * M * sizeof (dtype));
  dtype *C_sv = (dtype*) malloc (N * M * sizeof (dtype));
  assert (A && B && C);

  /* initialize A, B, C */
  srand48 (time (NULL));
  for(i = 0; i < N; i++) {
    for(j = 0; j < K; j++) {
      A[i * K + j] = drand48 ();
    }
  }
  for(i = 0; i < K; i++) {
    for(j = 0; j < M; j++) {
      B[i * M + j] = drand48 ();
    }
  }
  bzero(C, N * M * sizeof (dtype));
  bzero(C_cb, N * M * sizeof (dtype));
  bzero(C_sv, N * M * sizeof (dtype));

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create ();
  assert (timer);
  long double t;

  printf("Naive matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_serial (C, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for naive implementation: %Lg seconds\n\n", t);


  printf("Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_cb (C_cb, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_cb, C, N, M);

  printf("SIMD-vectorized Cache-blocked matrix multiply\n");
  stopwatch_start (timer);
  /* do C += A * B */
  mm_sv (C_sv, A, B, N, K, M);
  t = stopwatch_stop (timer);
  printf("Done\n");
  printf("time for SIMD-vectorized cache-blocked implementation: %Lg seconds\n", t);

  /* verify answer */
  verify (C_sv, C, N, M);

  return 0;
}
