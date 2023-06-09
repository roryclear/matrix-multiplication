#include <iostream>
#include <random>
#include <time.h>
#include <omp.h>
//#include "/usr/local/opt/libomp/include/omp.h"

//g++ -O3 -march=native -ffast-math matmul.cpp -o a

//g++ -O3 -march=native -ffast-math -fopenmp matmul.cpp -o a

//g++ -O3 -ffast-math -fopenmp matmul.cpp -o a

// for mac: https://apple.stackexchange.com/questions/99077/how-to-set-gcc-4-8-as-default-gcc-compiler/99157#99157

inline void matmulImplNaive(const float *left, const float *right,
                            float *result, int dim) {
int rows = dim;
int columns = dim;
int inners = dim;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }


inline void matmulFaster(const float *left, const float *right,
                            float *result, int dim) {
  for (int row = 0; row < dim; row++) {
    for (int inner = 0; inner < dim; inner++) {
      for (int col = 0; col < dim; col++) {
        result[row * dim + col] +=
            left[row * dim + inner] * right[inner * dim + col];
} } } }


inline void matmulTiling(const float *left, const float *right,
                            float *result, int dim, int tileSize) {
  for(int rowTile = 0; rowTile < dim; rowTile+=tileSize) {
    for (int innerTile = 0; innerTile < dim; innerTile+=tileSize) {
      for(int colTile = 0; colTile < dim; colTile+=tileSize) {
        for (int row = rowTile; row < rowTile+tileSize; row++) {
          for(int inner = innerTile; inner < innerTile+tileSize; inner++) {
            for (int col = colTile; col < colTile+tileSize; col++) {
              result[row * dim + col] +=
                  left[row * dim + inner] * right[inner * dim + col];
            }
          }
        }
      }
    }
  }
}

inline void matmulTilingMulti(const float *left, const float *right,
                            float *result, int dim, int tileSize) {
  #pragma omp parallel for
  for(int rowTile = 0; rowTile < dim; rowTile+=tileSize) {
    for (int innerTile = 0; innerTile < dim; innerTile+=tileSize) {
      for(int colTile = 0; colTile < dim; colTile+=tileSize) {
        for (int row = rowTile; row < rowTile+tileSize; row++) {
          for(int inner = innerTile; inner < innerTile+tileSize; inner++) {
            for (int col = colTile; col < colTile+tileSize; col++) {
              result[row * dim + col] +=
                  left[row * dim + inner] * right[inner * dim + col];
            }
          }
        }
      }
    }
  }
}

inline void matmulTilingMulti2(const float *left, const float *right,
                            float *result, int dim, int tileSize, int tileY, int tileZ) {
  #pragma omp parallel for
  for(int rowTile = 0; rowTile < dim; rowTile+=tileY) {
    for (int innerTile = 0; innerTile < dim; innerTile+=tileZ) {
      for(int colTile = 0; colTile < dim; colTile+=tileSize) {
        for (int row = rowTile; row < rowTile+tileY; row++) {
          for(int inner = innerTile; inner < innerTile+tileZ; inner++) {
            for (int col = colTile; col < colTile+tileSize; col++) {
              result[row * dim + col] +=
                  left[row * dim + inner] * right[inner * dim + col];
            }
          }
        }
      }
    }
  }
}



int main() {
    const int dim = 2048;
    float *left =  new float[dim*dim];
    float *right =  new float[dim*dim];
    float *resultA =  new float[dim*dim];
    float *resultB =  new float[dim*dim];
    float *resultC =  new float[dim*dim];
    for(int i = 0; i < dim*dim; i++) {
        left[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        right[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    clock_t tStart;
    matmulImplNaive(left,right,resultA,dim);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    matmulFaster(left,right,resultB,dim);
    printf("Time taken (reorder): %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    int tileSize = 8;


    while(tileSize < dim) {
      resultC =  new float[dim*dim];
      tStart = clock();
      matmulTiling(left,right,resultC,dim,tileSize);
      printf("Time taken (reorder + tiling): %.2fs tileSize = %d \n", (double)(clock() - tStart)/CLOCKS_PER_SEC, tileSize);
      for(int i = 0; i < dim*dim; i++) {
          if(resultA[i] != resultC[i]) {
              printf("ffs %d",i);
              return 0;
          }
      }
      tileSize += tileSize;
    }

    tileSize = 8;

    while(tileSize < dim) {
      resultC =  new float[dim*dim];
      double startTime = omp_get_wtime();
      matmulTilingMulti(left,right,resultC,dim,tileSize);
      printf("Time taken (reorder + tiling + multi): %.2fs tileSize = %d \n", (double)(omp_get_wtime() - startTime), tileSize);
      for(int i = 0; i < dim*dim; i++) {
          if(resultA[i] != resultC[i]) {
              printf("ffs %d",i);
              return 0;
          }
      }
      tileSize += tileSize;
    }

    int fx = 0;
    int fy = 0;
    int fz = 0;
    tileSize = 2;
    int tileY = 2;
    int tileZ = 2;
    double lowest = 420;
    for(tileSize = 256; tileSize < dim; tileSize*=2) {
      for(tileY = 4; tileY < dim; tileY*=2) {
        for(tileZ = 4; tileZ < dim; tileZ*=2) {
          resultC =  new float[dim*dim];
          double startTime = omp_get_wtime();
          matmulTilingMulti2(left,right,resultC,dim,tileSize,tileY,tileZ);
          double timeTaken = omp_get_wtime() - startTime;
          if(timeTaken < lowest) {
            lowest = timeTaken;
            fx = tileSize;
            fy = tileY;
            fz = tileZ;
            printf("fastest = %.2fs\n",lowest);
          }
          printf("Time taken (reorder + tiling + multi 2.0): %.2fs tileSize = %d tileY = %d tileZ = %d \n", (double)(timeTaken), tileSize, tileY, tileZ);
          for(int i = 0; i < dim*dim; i++) {
              if(resultA[i] != resultC[i]) {
                  printf("ffs %d",i);
                  return 0;
              }
          }
      }
      }
    }
    printf("fastest values: %d %d %d -> %.2fs\n",fx,fy,fz,lowest);


    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultB[i]) {
            return 0;
        }
    }
    printf("reorder outputs equal\n");

    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultC[i]) {
            return 0;
        }
    }

    printf("tiling outputs equal");

    return 0;
}