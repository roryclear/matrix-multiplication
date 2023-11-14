import objc
import Metal
import Foundation
import numpy as np
import struct

device = Metal.MTLCreateSystemDefaultDevice()

def sum(a):
    dim = len(a)
    mtl_queue = device.newCommandQueue()
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    prg = f"""#include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;
    kernel void sum(device float *res,
                        device const float *a,
                        uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
    {{
        int x = lid.x * 2;
        for(int i = 0; i < {dim}; i+=2048) {{
            res[x+i] = a[x+i] + a[x+i+1];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int i = 2; i <= {dim}/2; i*=2) {{
            for(int j = 0; j < {dim}; j+=2048) {{
            if((x+i+j) < {dim}) {{
            res[x+j] = res[x+j] + res[x+i+j];
                }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
        }}
    }}"""

    options = Metal.MTLCompileOptions.alloc().init()
    library, err = device.newLibraryWithSource_options_error_(prg, options, None)
    fxn = library.newFunctionWithName_("sum")
    pipeline_state, err = device.newComputePipelineStateWithFunction_error_(fxn, None)
    encoder.setComputePipelineState_(pipeline_state)

    size = dim*4
    a_buffer = device.newBufferWithLength_options_(size ,1)
    m = a_buffer.contents().as_buffer(size)
    m[:] = bytes(a)

    res_buffer = device.newBufferWithLength_options_(size ,1)

    encoder.setBuffer_offset_atIndex_(res_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 1)
    threadsPerGrid = Metal.MTLSizeMake(1,1,1)
    threadsPerThreadGroup = Metal.MTLSizeMake(1024,1,1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup)
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    output = np.asarray(res_buffer.contents().as_buffer(size))
    output = np.frombuffer(output, dtype=np.float32)[0]
    return output