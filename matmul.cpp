#include <iostream>
#include <random>
#include <time.h>

inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
int dim = 512;
int rows = dim;
int columns = dim;
int inners = dim;
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * columns + inner] * right[inner * columns + col];
} } } }

int main() {
    const int dim = 512;
    float left[dim*dim] = {};
    float right[dim*dim] = {};
    float result[dim*dim] = {};
    for(int i = 0; i < dim*dim; i++) {
        left[i] = rand();
        right[i] = rand();
    }
    clock_t tStart = clock();
    matmulImplNaive(left,right,result);
    printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    //for(int i = 0; i < dim*dim; i++) {
    //    std::cout << result[i];
    //}
    std::cout << "matmul??";
    return 0;
}