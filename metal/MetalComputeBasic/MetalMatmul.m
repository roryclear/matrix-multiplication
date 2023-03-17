/*
See LICENSE folder for this sample’s licensing information.

Abstract:
A class to manage all of the Metal objects this app creates.
*/

#import "MetalMatmul.h"

// The number of floats in each array, and the size of the arrays in bytes.
//const unsigned int arrayLength = 1 << 24;  // this is 1 followed by 24 zeros in binary
const unsigned int arrayLength = 1024*1024; // matrix
const unsigned int length = 1024;
const unsigned int bufferSize = arrayLength * sizeof(float);


@implementation MetalMatmul
{
    id<MTLDevice> _mDevice;

    // The compute pipeline generated from the compute kernel in the .metal shader file.

    id<MTLComputePipelineState> _mMatmulFunctionPSO;
    
    // The command queue used to pass commands to the device.
    id<MTLCommandQueue> _mCommandQueue;

    // Buffers to hold data.
    id<MTLBuffer> _mBufferA;
    id<MTLBuffer> _mBufferB;
    id<MTLBuffer> _mBufferResult;
    id<MTLBuffer> _mBufferLength;

}

- (instancetype) initWithDevice: (id<MTLDevice>) device
{
    self = [super init];
    if (self)
    {
        _mDevice = device;

        NSError* error = nil;

        // Load the shader files with a .metal file extension in the project

        id<MTLLibrary> defaultLibrary = [_mDevice newDefaultLibrary];
        
        id<MTLFunction> matmulFunction = [defaultLibrary newFunctionWithName:@"matmul"];

        // Create a compute pipeline state object.
        
        _mMatmulFunctionPSO = [_mDevice newComputePipelineStateWithFunction: matmulFunction error:&error];

        _mCommandQueue = [_mDevice newCommandQueue];
    }

    return self;
}

- (void) sendComputeCommand
{
    // Allocate three buffers to hold our initial data and the result.
    _mBufferA = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferB = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferResult = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    _mBufferLength = [_mDevice newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];

    [self generateRandomFloatData:_mBufferA];
    [self generateRandomFloatData:_mBufferB];
    
    int* dataPtr = _mBufferLength.contents;

    dataPtr[0] = (int) length;
    
    // Create a command buffer to hold commands.
    id<MTLCommandBuffer> commandBuffer = [_mCommandQueue commandBuffer];
    assert(commandBuffer != nil);

    // Start a compute pass.
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    assert(computeEncoder != nil);

    [self encodeMatmulCommand:computeEncoder];

    // End the compute pass.
    [computeEncoder endEncoding];

    // Execute the command.
    NSDate *start = [NSDate date];
    [commandBuffer commit];

    // Normally, you want to do other work in your app while the GPU is running,
    // but in this example, the code simply blocks until the calculation is complete.
    [commandBuffer waitUntilCompleted];
    NSTimeInterval timeInterval = [start timeIntervalSinceNow];
    NSLog(@"time taken metal = %f",timeInterval);

    [self verifyResults];
}

- (void)encodeMatmulCommand:(id<MTLComputeCommandEncoder>)computeEncoder {

    // Encode the pipeline state object and its parameters.
    [computeEncoder setComputePipelineState:_mMatmulFunctionPSO];
    [computeEncoder setBuffer:_mBufferA offset:0 atIndex:0];
    [computeEncoder setBuffer:_mBufferB offset:0 atIndex:1];
    [computeEncoder setBuffer:_mBufferResult offset:0 atIndex:2];
    [computeEncoder setBuffer:_mBufferLength offset:0 atIndex:3];

    
    
    MTLSize gridSize = MTLSizeMake(arrayLength, 1, 1);  // grid size measured in threads not blocks like CUDA

    NSLog(@"arrayLength = %lu",arrayLength);
    
    // Calculate a threadgroup size.
    NSUInteger threadGroupSize = _mMatmulFunctionPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > arrayLength)
    {
        threadGroupSize = arrayLength;
    }
    
    NSLog(@"threadGroupSize = %lu",threadGroupSize);
    
    NSLog(@"gridSize = %lu, %lu, %lu",gridSize.width,gridSize.height,gridSize.depth);

    
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    //MTLSize threadgroupSize = MTLSizeMake(1, 1, 1); this also works but slower
    
    //gridSize = MTLSizeMake((NSUInteger) gw, 1, 1);
    
    NSLog(@"gridSize = %lu, %lu, %lu",gridSize.width,gridSize.height,gridSize.depth);
    NSLog(@"threadGroupSize = %lu, %lu, %lu",threadgroupSize.width,threadgroupSize.height,threadgroupSize.depth);

    // Encode the compute command.
    [computeEncoder dispatchThreads:gridSize
              threadsPerThreadgroup:threadgroupSize];
}

- (void) generateRandomFloatData: (id<MTLBuffer>) buffer
{
    float* dataPtr = buffer.contents;

    for (unsigned long index = 0; index < arrayLength; index++)
    {
        dataPtr[index] = (float)rand()/(float)(RAND_MAX);
    }
}
- (void) verifyResults
{
    float* a = _mBufferA.contents;
    float* b = _mBufferB.contents;
    float* result = _mBufferResult.contents;
    
    id<MTLBuffer> _mBufferResult2 = [_mDevice newBufferWithLength:bufferSize options:MTLResourceStorageModeShared];
    float* resultCPU = _mBufferResult2.contents;
    
    NSDate *start = [NSDate date];
    for (unsigned long i = 0; i < (length*length); i++)
    {
        int row = i / length;
        int col = i % length;
        float t = 0;
        for (unsigned long j = 0; j < length; j++) {
            t += a[(row * length) + j] * b[col + (length * j)];
        }
        resultCPU[i] = t;
    }
    NSTimeInterval timeInterval = [start timeIntervalSinceNow];
    NSLog(@"time taken = %f",timeInterval);
    
    for (unsigned long index = 0; index < arrayLength; index++)
    {
        if (result[index] != resultCPU[index])
        {
            printf("Compute ERROR: index=%lu result=%g vs %g\n",
                   index, result[index], resultCPU[index]);
                    assert(result[index] == resultCPU[index]);
        }

    }
    printf("Compute results as expected\n");
}
@end
