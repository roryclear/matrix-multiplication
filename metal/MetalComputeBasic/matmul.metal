/*
See LICENSE folder for this sample’s licensing information.

*/

#include <metal_stdlib>
using namespace metal;
/// This is a Metal Shading Language (MSL) function equivalent to the add_arrays() C function, used to perform the calculation on a GPU.
kernel void add_arrays(device const float* inA,
                       device const float* inB,
                       device float* result,
                       uint index [[thread_position_in_grid]])
{
    // the for-loop is replaced with a collection of threads, each of which
    // calls this function.
    result[index] = inA[index] + inB[index];
}

kernel void matmul(device const float* inA,
                       device const float* inB,
                       device float* result,
                        device int& length,
                       uint index [[thread_position_in_grid]])
{
    int row = index / length; // unhardcode
    int col = index % length;
    
    float total = 0;
    if(row < length && col < length)
    {
        for(int i = 0; i < length; i++)
        {
            total += inA[(row * length) + i] * inB[col + (i * length)];
        }
        result[(row * length) + col] = total;
    }
}
