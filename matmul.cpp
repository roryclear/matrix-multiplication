#include <iostream>
#include <random>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <unistd.h>
//#include "/usr/local/opt/libomp/include/omp.h"

//g++ -O3 -march=native -ffast-math matmul.cpp -o a

//g++ -O3 -march=native -ffast-math -fopenmp matmul.cpp -o a

//g++ -O3 -ffast-math -fopenmp matmul.cpp -o a

// for mac: https://apple.stackexchange.com/questions/99077/how-to-set-gcc-4-8-as-default-gcc-compiler/99157#99157
// alias g++=/usr/local/bin/g++-13

#define by 4
#define bx 2

const int dim = 512;

float *left =  new float[dim*dim];
float *right =  new float[dim*dim];
float *rightr =  new float[dim*dim];
float *rightr2 =  new float[dim*dim];
float *resultA =  new float[dim*dim];
float *resultB =  new float[dim*dim];
float *resultC =  new float[dim*dim];

inline void matmulImplNaive() {
  for (int y = 0; y < dim; y++) {
    for (int x = 0; x < dim; x++) {
      for (int k = 0; k < dim; k++) {
        resultA[y * dim + x] += left[y * dim + k] * right[k * dim + x];
      } 
    } 
  } 
}


inline void matmulReordered() {
  for (int y = 0; y < dim; y++) {
    for (int k = 0; k < dim; k++) {
      float lnum = left[y * dim + k];
      for (int x = 0; x < dim; x++) {
        resultA[y * dim + x] += lnum * right[k * dim + x];
      } 
    } 
  } 
}

inline void matmulSwizzle() {
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx) {
      float acc[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          float lnum = left[(y+iy)*dim + k];
          for(int ix = 0; ix < bx; ix++) {
              acc[iy][ix] += lnum * rightr[(x+ix)*dim + k];
          }
        }
      }

      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultC[(y+iy)*dim + x + ix] = acc[iy][ix];
        }
      }

    } 
  } 
}

inline void matmulSwizzleAvx() {
  __m256 *rightm = (__m256*)rightr2;
  __m256 *resultm = (__m256*)resultC;
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx*8) {
      __m256 accm[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          __m256 am = _mm256_broadcast_ss(&left[(y+iy)*dim + k]);
          for(int ix = 0; ix < bx; ix++) {
            accm[iy][ix] = _mm256_fmadd_ps(am,rightm[((x + ix * 8)*dim + k*8)/8],accm[iy][ix]);
          }
        }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultm[((y+iy)*dim + x + ix * 8)/8] = accm[iy][ix];
        }
      }
    } 
  } 
}

inline void matmulAvx() {
  __m256 *rightm = (__m256*)right;
  __m256 *resultm = (__m256*)resultC;
  for(int x = 0; x < dim; x+=bx*8) {
    for(int y = 0; y < dim; y+=by) {
      __m256 accm[by][bx] = {};
      for(int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
            __m256 am = _mm256_broadcast_ss(&left[((y+iy)*dim) + k]);
            for(int ix = 0; ix < bx; ix++) {
              accm[iy][ix] = _mm256_fmadd_ps(am, rightm[((k*dim) + x + ix*8)/8],accm[iy][ix]);
            }
         }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultm[((y+iy)*dim + x + ix*8)/8] = accm[iy][ix];
        }
      }
    }
  }
}

inline void matmulAvxMutlti() {
  __m256 *rightm = (__m256*)right;
  __m256 *resultm = (__m256*)resultC;
  #pragma omp parallel for
  for(int x = 0; x < dim; x+=bx*8) {
    for(int y = 0; y < dim; y+=by) {
      __m256 accm[by][bx] = {};
      for(int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
            __m256 am = _mm256_broadcast_ss(&left[((y+iy)*dim) + k]);
            for(int ix = 0; ix < bx; ix++) {
              accm[iy][ix] = _mm256_fmadd_ps(am, rightm[((k*dim) + x + ix*8)/8],accm[iy][ix]);
            }
         }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultm[((y+iy)*dim + x + ix*8)/8] = accm[iy][ix];
        }
      }
    }
  }
}

inline void matmulSwizzleAvxMulti() {
  __m256 *rightm = (__m256*)rightr2;
  __m256 *resultm = (__m256*)resultC;
  #pragma omp parallel for
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx*8) {
      __m256 accm[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          __m256 am = _mm256_broadcast_ss(&left[(y+iy)*dim + k]);
          for(int ix = 0; ix < bx; ix++) {
            accm[iy][ix] = _mm256_fmadd_ps(am,rightm[((x + ix * 8)*dim + k*8)/8],accm[iy][ix]);
          }
        }
      }
      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultm[((y+iy)*dim + x + ix * 8)/8] = accm[iy][ix];
        }
      }
    } 
  } 
}

inline void matmulSwizzleMulti() {
  #pragma omp parallel for
  for (int y = 0; y < dim; y+=by) {
    for (int x = 0; x < dim; x+=bx) {
      float acc[by][bx] = {};
      for (int k = 0; k < dim; k++) {
        for(int iy = 0; iy < by; iy++) {
          float lnum = left[(y+iy)*dim + k];
          for(int ix = 0; ix < bx; ix++) 
          {
            acc[iy][ix] += lnum * rightr[(x+ix)*dim + k];
          }
        }
      }

      for(int iy = 0; iy < by; iy++) {
        for(int ix = 0; ix < bx; ix++) {
          resultC[(y+iy)*dim + x + ix] = acc[iy][ix];
        }
      }

    } 
  } 
}


inline void matmulTiling() {
  int xs = 256;
  int ys = 16;
  int ks = 8;
  for(int yt = 0; yt < dim; yt+=ys) {
    for (int kt = 0; kt < dim; kt+=ks) {
      for(int xt = 0; xt < dim; xt+=xs) {
        for (int y = yt; y < yt+ys; y++) {
          for(int k = kt; k < kt+ks; k++) {
            float lnum = left[y * dim + k];
            for (int x = xt; x < xt+xs; x++) {
              resultC[y * dim + x] +=
                  lnum * right[k * dim + x];
            }
          }
        }
      }
    }
  }
}


//1024 4 4 best on xps
// 256 16 8 best on macbook
inline void matmulTilingMulti() {
  int xs = 256;
  int ys = 16;
  int ks = 4;
  #pragma omp parallel for
  for(int yt = 0; yt < dim; yt+=ys) {
    for (int kt = 0; kt < dim; kt+=ks) {
      for(int xt = 0; xt < dim; xt+=xs) {
        for (int y = yt; y < yt+ys; y++) {
          for(int k = kt; k < kt+ks; k++) {
            float lnum = left[y * dim + k];
            for (int x = xt; x < xt+xs; x++) {
              resultC[y * dim + x] +=
                  lnum * right[k * dim + x];
            }
          }
        }
      }
    }
  }
}

inline void checkOutput() {
  for(int i = 0; i < dim*dim; i++) {
    if(abs(resultC[i] - resultA[i]) > abs(resultA[i]*0.00001)) {
        float diff = abs(resultC[i] - resultA[i]);
        printf("ffs %d %f -> %f %f\n",i,resultC[i],resultA[i],diff);
        exit(0);
    }
  }
}



int main() {
  for(int i = 0; i < dim*dim; i++) {
      left[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
      right[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
  }

  double t = 0;
  double startTime = 0;

  for(int i = 0; i < 10; i++) {
    startTime = omp_get_wtime();
    matmulImplNaive();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (Naive): %.2fs\n", t);
    usleep(5000000);
    //printf("done one");
  }
  printf("avg Time taken (Naive): %.2fs\n", t/10);

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultA =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulReordered();
    t+= (double)(omp_get_wtime() - startTime);
    //printf("Time taken (reorder): %.2fs\n", t);
    usleep(1000000);
  }
  printf("avg Time taken (reorder): %.4fs\n", t/10);


  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulTiling();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (reorder + tiling): %.2fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (reorder + tiling): %.4fs\n", t/10);

  for(int y = 0; y < dim; y++) {
    for(int x = 0; x < dim; x++) {
      rightr[y * dim + x] = right[x * dim + y];
    }
  }

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulAvx();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (avx + tiling): %.5fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (avx + tiling): %.5fs\n", t/10);

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulSwizzle();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (swizzle): %.5fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (swizzle): %.5fs\n", t/10);

  for(int y = 0; y < dim; y+=8) {
    for(int x = 0; x < dim; x++) {
      for(int iy = 0; iy < 8; iy++) {
        rightr2[y*dim + x*8 + iy] = rightr[(y+iy)*dim + x]; //each 256 contains y,x -> y+7,x
      }
    }
  }
  
  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulSwizzleAvx();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (swizzle + avx + tiling): %.5fs\n", t);
    checkOutput();
    usleep(10000000);
  } 
  printf("avg Time taken (swizzle + avx + tiling): %.5fs\n", t/10);

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulAvxMutlti();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (avx + tiling + multi): %.5fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (avx + tiling + multi): %.5fs\n", t/10);

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulTilingMulti();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (reorder + tiling + multi): %.5fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (reorder + tiling + multi): %.5fs\n", t/10);

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulSwizzleMulti();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (swizzle + multi): %.5fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (swizzle + multi): %.5fs\n", t/10);

  t = 0;
  for(int i = 0; i < 10; i++) {
    resultC =  new float[dim*dim];
    startTime = omp_get_wtime();
    matmulSwizzleAvxMulti();
    t += (double)(omp_get_wtime() - startTime);
    //printf("Time taken (swizzle + avx + tiling + multi): %.5fs\n", t);
    checkOutput();
    usleep(5000000);
  }
  printf("avg Time taken (swizzle + avx + tiling + multi): %.5fs\n", t/10);

  return 0;
}