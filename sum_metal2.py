import objc
import Metal
import Foundation
import numpy as np
import struct
device = Metal.MTLCreateSystemDefaultDevice()

def sum(a):
    #works on 1024*((N+2)*4)
    dim = len(a)
    mtl_queue = device.newCommandQueue()
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    prg = f"""#include <metal_stdlib>
    #include <metal_simdgroup_matrix>
    using namespace metal;
    kernel void sum(device float *res,
                        device float *a,
                        uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
    {{
        int x = lid.x * {dim}/1024; //max threads
        for(int i = {dim}; i > 2; i*=0.25) {{
            if(x < i) {{
                for(int j = 0; j < {dim}/1024; j++) {{
                    res[x+j] = a[x*2+2*j] + a[x*2+2*j+1];
                }}  
            }}

            if(x < i/2) {{
                for(int j = 0; j < {dim}/1024; j++) {{
                    a[x+j] = res[x*2+2*j] + res[x*2+2*j+1];
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup); //todo add if, dont always need 
        res[0] = a[0];
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