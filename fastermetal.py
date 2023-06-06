import Metal
import numpy as np

length = 2048

device = Metal.MTLCreateSystemDefaultDevice()

na = np.empty([length, length]).astype(np.float32).flatten()
a = device.newBufferWithLength_options_(na.nbytes ,1)

nb = np.random.rand(length,length).astype(np.float32)
nc = np.random.rand(length,length).astype(np.float32)

b = device.newBufferWithLength_options_(nb.nbytes ,1)
c = device.newBufferWithLength_options_(nc.nbytes ,1)

answer = np.empty_like(a)


prg = f"""#include <metal_stdlib>
using namespace metal;
kernel void matmul()
{{     

}}"""


mtl_queue = device.newCommandQueue()
command_buffer = mtl_queue.commandBuffer()
encoder = command_buffer.computeCommandEncoder()

options = Metal.MTLCompileOptions.alloc().init()
library, err = device.newLibraryWithSource_options_error_(prg, options, None)
fxn = library.newFunctionWithName_("matmul")
pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
encoder.setComputePipelineState_(pipeline_state)

threadGroupSize = pipeline_state.maxTotalThreadsPerThreadgroup()
if length*length < threadGroupSize:
    threadGroupSize = length*length
encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(length*length,1,1), Metal.MTLSizeMake(threadGroupSize,1,1))
encoder.endEncoding()
command_buffer.commit()
command_buffer.waitUntilCompleted()


print("done")