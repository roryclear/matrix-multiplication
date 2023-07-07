import objc
import Metal
import Foundation
import numpy as np
import struct

def matmul(a,b):
    dim = a.shape[0]
    device = Metal.MTLCreateSystemDefaultDevice()
    mtl_queue = device.newCommandQueue()
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    prg = f"""#include <metal_stdlib>
    using namespace metal;
    kernel void matmul(device float *a,
                        device float *b,
                        device int& dim,
                        device float *res,
    uint3 index [[thread_position_in_grid]])
    {{  
        int by = 4;
        int bx = 2;
        for(int x = 0; x < dim; x+=bx) {{
            for(int y = 0; y < dim; y+=by) {{   
                for(int k = 0; k < dim; k++) {{
                    for(int iy = 0; iy < by; iy++) {{
                        float lnum = a[(y + iy)*dim + k];
                        for(int ix = 0; ix < bx; ix++) {{
                            res[(y+iy)*dim + (x + ix)] += lnum * b[k*dim + (x + ix)];  
                        }}
                    }}
                }}
            }}
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

    dim = np.int32(dim)
    dim_buffer = device.newBufferWithLength_options_(4 ,1)

    for i in range(4):
        dim_buffer.contents().__setitem__(i,dim.tobytes()[i].to_bytes(1,'big'))

    res = np.empty([dim, dim]).astype(np.float32).flatten()
    res_buffer = device.newBufferWithLength_options_(res.nbytes ,1)

    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(b_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(dim_buffer, 0, 2)
    encoder.setBuffer_offset_atIndex_(res_buffer, 0, 3)
    threadGroupSize = pipeline_state.maxTotalThreadsPerThreadgroup()
    if dim*dim < threadGroupSize:
        threadGroupSize = dim*dim
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(1,1,1), Metal.MTLSizeMake(1,1,1)) #1thread for now?
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    output = np.asarray(res_buffer.contents().as_buffer(res.nbytes))
    output = np.frombuffer(output, dtype=np.float32)
    return output