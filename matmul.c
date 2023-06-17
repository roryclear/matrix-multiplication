#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>

#define N 2048
float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));
float ans[N*N] __attribute__ ((aligned (64)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

float Bf[N*N] __attribute__ ((aligned (64))); //no idea why geohot does this?
__m256 *Bfm = (__m256*)Bf;

int main() {
   // printf() displays the string inside quotation
   printf("Hello, World!\n");

   for(int i = 0; i < N*N; i++) {
      float max = 1;
      A[i] = (float)rand()/(float)(RAND_MAX/max);
      B[i] = (float)rand()/(float)(RAND_MAX/max);
      C[i] = 0;
   }

   matmul();

   for(int i = 0; i < N*N; i++) {
      ans[i] = C[i];
      C[i] = 0;
   }

   printf("HERE");

   matmul2();

   for(int i = 0; i < N*N; i++) {
      if(ans[i] != C[i]) {
         printf("WRONG %f != %f",ans[i],C[i]);
         return;
      }
   }

   printf("\nDONE\n");
   return 0;
}


void matmul() {
   
   for(int y = 0; y < N; y++) {
      for(int k = 0; k < N; k++) {
         for(int x = 0; x < N; x++) {
            C[y*N + x] += A[y*N + k] * B[k * N + y];
         }
      }
   }
   
   /*
   for(int y = 0; y < N; y++) {
      for(int x = 0; x < N; x++) {
         for(int k = 0; k < N; k++) {
            C[y*N + x] += A[y*N + k] * B[x + k*N];
         }
      }
   }
   */
}

void matmul2() {
   int sy = 0;
   int ey = N;

   int BLOCK_X=2;
   int BLOCK_Y=4;
   int BLOCK=8;

   for (int y = sy; y < ey; y+=BLOCK_Y) {
    for (int x = 0; x < N; x+=BLOCK*BLOCK_X) {

      __m256 acc[BLOCK_Y][BLOCK_X] = {};
      for (int k = 0; k < N; k++) {
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          __m256 ta = _mm256_broadcast_ss(&A[(y+iy)*N + k]);
          for (int ix = 0; ix < BLOCK_X; ix++) {
            acc[iy][ix] = _mm256_fmadd_ps(ta, Bm[((x+ix*BLOCK)*N + k*8)/8], acc[iy][ix]);
          }
        }
      }

      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          Cm[((y+iy)*N + x + ix * BLOCK)/8] = acc[iy][ix];
        }
      }
   }
}









}