// https://developer.apple.com/library/archive/samplecode/OpenCL_Hello_World_Example/Listings/hello_c.html
// https://jameshfisher.com/2017/04/19/macos-opencl-hello-world/
//clang -framework OpenCL main.c
// Simple compute kernel which computes the square of an input array
//
#include <stdio.h>
#include <OpenCL/opencl.h>
#define DATA_SIZE (256)

const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

const char *KernelSource2 = "\n" \
"__kernel void matmul(                                                       \n" \
"   __global float* a,                                              \n" \
"   __global float* b,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   int row = get_global_id(0) / count;                                           \n" \
"   int col = get_global_id(0) % count;                                           \n" \
"   if(row < count && col < count) {                                                      \n" \
    "float total = 0;\n"\
    "for(int j = 0; j < count; j++)\n" \
    "{  \n" \
    "    total += a[(row * count) + j] * b[(j * count) + col];  \n" \
    "}\n" \
"       output[(row * count) + col] = total;                                \n" \
"    }                             \n" \
"}                                                                      \n" \
"\n";
 
////////////////////////////////////////////////////////////////////////////////
 
int main(void)
{
    int err;                            // error code returned from api calls

    printf("workgroup size = %d\n",CL_KERNEL_WORK_GROUP_SIZE);

    cl_device_id device_id;             // compute device id
    clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    cl_command_queue commands = clCreateCommandQueue(context, device_id, 0, &err);
    cl_program program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource2, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matmul", &err);
    
    cl_mem inputA = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * DATA_SIZE, NULL, NULL);
    cl_mem inputB = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * DATA_SIZE, NULL, NULL);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);
    
    float data[DATA_SIZE];

    for(int i = 0; i < DATA_SIZE; i++) {
        //data[i] = i;
        //data[i] = 10 * (rand() / (float)RAND_MAX);
        data[i] = 1;
        //printf("%f\n",data[i]);
    }
    
    err = clEnqueueWriteBuffer(commands, inputA, CL_TRUE, 0, sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(commands, inputB, CL_TRUE, 0, sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &output);
    unsigned int count = DATA_SIZE;
    unsigned int rows = 16;
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &rows);
    size_t local;
    
    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    size_t global = count;
    clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    clFinish(commands);
    
    float results[DATA_SIZE];
    clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL);
    unsigned int correct = 0;
    
    // 
    float results2[DATA_SIZE];
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < rows; j++) {
            float total = 0;
            for(int k = 0; k < rows; k++)
            {
                total += data[(i * rows) + k] * data[j + (k * rows)];
            }
            results2[(i * rows) + j] = total;
        }
    }
    //

    for (int i = 0; i < count; i++) {
        printf("value at %d = %f    %f\n",i,results[i],results2[i]);
        if (results[i] == data[i] * data[i]) { correct++; }
        if (results2[i] == results[i]) { correct++; }
    }
    printf("Computed '%d/%d' correct values!\n", correct, count);
    clReleaseMemObject(inputA);
    clReleaseMemObject(inputB);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    return 0;
}