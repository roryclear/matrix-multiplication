import objc
import Metal
import Foundation
import numpy as np
import struct

def matmul(a,b,length):
    device = Metal.MTLCreateSystemDefaultDevice()
    mtl_queue = device.newCommandQueue()
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    prg = f"""#include <metal_stdlib>
    using namespace metal;
    kernel void matmul(device float *a,
                        device float *b,
                        device int& length,
                        device float *res,
    uint index [[thread_position_in_grid]])
    {{     
          int row = index / length;
          int col = index % length;
          float total = 0;
          if(row < length && col < length) 
          {{
            for(int i = 0; i < length; i++)
            {{
              total += a[row * length + i] * b[col + i * length];
            }}
            res[row * length + col] = total;
          }}
    }}"""

    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(prg, options, None)
    fxn = library.newFunctionWithName_("matmul")
    pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
    encoder.setComputePipelineState_(pipeline_state)

    a_buffer = device.newBufferWithLength_options_(a.nbytes ,1)
    m = a_buffer.contents().as_buffer(a.nbytes)
    m[:] = bytes(a)

    b_buffer = device.newBufferWithLength_options_(b.nbytes ,1)
    m = b_buffer.contents().as_buffer(b.nbytes)
    m[:] = bytes(b)

    n = np.int32(length)
    n_buffer = device.newBufferWithLength_options_(4 ,1)
    for i in range(4):
        n_buffer.contents().__setitem__(i,n.tobytes()[i].to_bytes(1,'big'))

    res = np.empty_like(b)
    res_buffer = device.newBufferWithLength_options_(res.nbytes ,1)

    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(b_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(n_buffer, 0, 2)
    encoder.setBuffer_offset_atIndex_(res_buffer, 0, 3)
    threadGroupSize = pipeline_state.maxTotalThreadsPerThreadgroup()
    if length < threadGroupSize:
        threadGroupSize = length
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(length*length,1,1), Metal.MTLSizeMake(threadGroupSize,1,1))
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    output = np.asarray(res_buffer.contents().as_buffer(res.nbytes))
    output = np.frombuffer(output, dtype=np.float32)
    return output