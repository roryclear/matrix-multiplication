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

//clang -O2 -march=native gemm.c -lpthread

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


   //expriment
   float aa[64] __attribute__ ((aligned (64)));
   float bb[64] __attribute__ ((aligned (64)));
   float cc[64] __attribute__ ((aligned (64)));

   float an[64] __attribute__ ((aligned (64)));

   __m256 *aam = (__m256*)aa;
   __m256 *bbm = (__m256*)bb;
   __m256 *ccm = (__m256*)cc;
   float max =1;
   for(int i = 0; i < 64; i++) {
      aa[i] = (float)rand()/(float)(RAND_MAX/max);
      bb[i] = (float)rand()/(float)(RAND_MAX/max);
      cc[i] = 0;
      an[i] = 0;
      printf("%f %f %f %f\n",aa[i],bb[i],cc[i],an[i]);
   }


   for(int i = 0; i < 8; i++) {
      an[i] = (aa[0] * bb[i]) + an[i];
      an[i] = (aa[0] * bb[i]) + an[i];
   }

   __m256 taa = _mm256_broadcast_ss(&aa[0]);
   ccm[0] = _mm256_fmadd_ps(taa, bbm[0], ccm[0]);
   ccm[0] = _mm256_fmadd_ps(taa, bbm[0], ccm[0]);

   for(int i = 0; i < 8; i++) {
      printf("%f -> %f\n",cc[i],an[i]);
      if(cc[i] != an[i]) {
         printf("\nWRONG\n");
         return;
      }
   }

   printf("\n_mm256_broadcast_ss works\n");

   int dim = 8;

   for(int i = 0; i < dim*dim; i++) {
      aa[i] = (float)rand()/(float)(RAND_MAX/max);
      bb[i] = (float)rand()/(float)(RAND_MAX/max);
      cc[i] = 0;
      an[i] = 0;
      //printf("%f %f %f %f\n",aa[i],bb[i],cc[i],an[i]);
   }
   
   for(int y = 0; y < dim; y++) {
      for(int k = 0; k < dim; k++) {
         for(int x = 0; x < dim; x++) {
            an[y*dim + x] += aa[y*dim + k] * bb[x + k*dim];
         }
      }
   }
   
   printf("DONE\n");

   printf("{");
   for(int y = 0; y < 8; y++) {
      printf("{");
      for(int x = 0; x < 7; x++) {
         printf("%f,",bb[y*8 + x]);
      }
      printf("%f},",bb[y*8 + 7]);
   }
   printf("\n\n\n");

   int BLOCK = 8;
   for(int y = 0; y < dim; y++) {
      for(int k = 0; k < dim; k++) {
         __m256 ta = _mm256_broadcast_ss(&aa[(y*dim) + k]);
         for(int x = 0; x < dim; x+=8) {
            ccm[(y*dim + x)/8] = _mm256_fmadd_ps(ta, bbm[((k*dim) + x)/8], ccm[(y*dim + x)/8]);
         }
      }
   }

   for(int i = 0; i < dim*dim; i++) {
      printf("FFS %d %f -> %f\n",i,cc[i],an[i]);
      if(cc[i] != an[i]) {
         printf("\nWRONG avx ! %f -> %f\n",cc[i],an[i]);
         return;
      }
   } 

   for(int i = 0; i < N*N; i++) {
      float max = 1;
      A[i] = (float)rand()/(float)(RAND_MAX/max);
      B[i] = (float)rand()/(float)(RAND_MAX/max);
      C[i] = 0;
   }

   matmul();

   for(int i = 0; i < N*N; i++) {
      ans[i] = C[i];

      //experiment
      //B[i] = 1;
      //A[i] = 1;
      //

      C[i] = 0;
   }

   printf("HERE\n");

   matmul2();

   for(int i = 0; i < N*N; i++) {
      if(ans[i] != C[i]) {
         //printf("%d WRONG %.20f != %.20f\n",i,ans[i],C[i]);
      //   return;
      } else{
         printf("CORRECT!!!!!!!!!!!! %d\n",i);
      }
   }

   printf("\nDONE\n");
   return 0;
}


void matmul() {
   
   for(int y = 0; y < N; y++) {
      for(int k = 0; k < N; k++) {
         for(int x = 0; x < N; x++) {
            C[y*N + x] += A[y*N + k] * B[x + (N * k)];
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