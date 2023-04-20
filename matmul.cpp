#include <iostream>
#include <random>
#include <time.h>

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
int rows = dim;
int columns = dim;
int inners = dim;
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

int main() {
    const int dim = 512;
    float left[dim*dim] = {};
    float right[dim*dim] = {};
    float resultA[dim*dim] = {};
    float resultB[dim*dim] = {};
    for(int i = 0; i < dim*dim; i++) {
        left[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        right[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    clock_t tStart = clock();
    matmulImplNaive(left,right,resultA,dim);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    tStart = clock();
    matmulFaster(left,right,resultB,dim);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);

    for(int i = 0; i < dim*dim; i++) {
        if(resultA[i] != resultB[i]) {
            return 0;
        }
    }
    std::cout << "outputs equal";
    return 0;
}