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

//gcc -O2 -march=native matmul.c -o c

#define N 2048
float max =1;
float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));
float ans[N*N] __attribute__ ((aligned (64)));
atomic_int done = 0;

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

float Bf[N*N] __attribute__ ((aligned (64))); //no idea why geohot does this?
__m256 *Bfm = (__m256*)Bf;

void matmulAvx() {
      for(int y = 0; y < N; y++) {
         for(int k = 0; k < N; k++) {
            __m256 ta = _mm256_broadcast_ss(&A[(y*N) + k]);
            for(int x = 0; x < N; x+=8) {
               Cm[(y*N + x)/8] = _mm256_fmadd_ps(ta, Bm[((k*N) + x)/8], Cm[(y*N + x)/8]);
            }
         }
      }
   }

void matmulAvxTiled() {
   int xs = 2;
   int ys = 4;
   for(int yt = 0; yt < N; yt+=ys){
      //printf("yt = %d\n",yt);
      for(int xt = 0; xt < N; xt+=xs*8){
         //printf("xt = %d\n",xt);
         __m256 tile[4][2] = {};
         for(int y = yt; y < yt+ys; y++) {
            for(int k = 0; k < N; k++) {
               __m256 ta = _mm256_broadcast_ss(&A[(y*N) + k]);
               for(int x = xt; x < xt+xs*8; x+=8) {
                  tile[(y-yt)][(x-xt)/8] = _mm256_fmadd_ps(ta, Bm[((k*N) + x)/8], tile[(y-yt)][(x-xt)/8]);
                  //printf("done\n");
               }
            }
         }

      for(int y = 0; y < 4; y++) {
         for(int x = 0; x < 2*8; x+=8) {
            //Cm[((yt+y)*N + x + xt)/8] = Cm[((yt+y)*N + x + xt)/8];//tile[0][0];
            //printf("here %d %d\n",y,x);a
            Cm[((yt+y)*N + x + xt)/8] = tile[y][x/8];
            //printf("crash??\n");
         }
      }
      //printf("HERE\n");
      }
   }
}

void checkAndReset() {
   for(int i = 0; i < N*N; i++) {
      //printf("FFS %d %f -> %f\n",i,C[i],ans[i]);
      if(fabsf(C[i] - ans[i]) > fabsf(C[i] / 1000000)) {
         printf("\nWRONG! %f -> %f\n",C[i],ans[i]);
         return;
      }
   }

   for(int i = 0; i < N*N; i++) {
       A[i] = (float)rand()/(float)(RAND_MAX/max);
       B[i] = (float)rand()/(float)(RAND_MAX/max);
       C[i] = 0;
       ans[i] = 0;
      //printf("%f %f %f %f\n",aa[i],bb[i],cc[i],an[i]);
   }
   return;
}

void matmulSection(start,end) {
   printf("section %d\n",end);
   for(int y = start; y < end; y++) {
      for(int k = 0; k < N; k++) {
         for(int x = 0; x < N; x++) {
            C[y*N + x] += A[y*N + k] * B[x + (N * k)];
         }
      }
   }
   done++;
}

void *matmulThread(void *n) {
   int start = (N/4) * (int)(int64_t)n;
   int end = start + (N/4);

   matmulSection(start,end);
}

void matmul() {
   for(int y = 0; y < N; y++) {
      for(int k = 0; k < N; k++) {
         for(int x = 0; x < N; x++) {
            ans[y*N + x] += A[y*N + k] * B[x + (N * k)];
         }
      }
   }
}

int main() {
   // printf() displays the string inside quotation
   printf("Hello, World!\n");
   
   clock_t begin = clock();
   matmul();
   clock_t end = clock();
   double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("reordered time spent = %f\n",time_spent);

   checkAndReset();
   matmul();

   pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
   pthread_t threads[4];
   
   begin = clock();
   for(int i = 0; i < 4; i++) {
      pthread_create(&threads[i], NULL, matmulThread, (void *)(uint64_t)i);
   }
   while(done < 4) {
      usleep(1);
   }
   end = clock();
   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("threaded time spent = %f\n",time_spent);
   checkAndReset();
   matmul();


   begin = clock();

   matmulAvx();

   end = clock();
   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("reordered + avx time spent = %f\n",time_spent);

   checkAndReset();
   matmul();

   begin = clock();
   matmulAvxTiled();
   end = clock();

   time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
   printf("reordered + avx tiled time spent = %f\n",time_spent);
   checkAndReset();
   return 0;
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
