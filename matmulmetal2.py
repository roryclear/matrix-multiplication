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
    #include <metal_simdgroup_matrix>
    using namespace metal;
    kernel void matmul(device float *res,
                        device const float *a,
                        device const float *b,
                        uint3 gid [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]])
    {{
      res += gid.x * 8 + lid.y * 8*8*16;
      a += lid.y * 8*8*16;
      b += gid.x * 8;

      simdgroup_float8x8 x[16];
      simdgroup_float8x8 y[16];
      simdgroup_float8x8 acc = simdgroup_float8x8(0);

      for(int i = 0; i < 16; i++) {{
          simdgroup_load(x[i],a+(8*i),128,ulong2(0,0));
          simdgroup_load(y[i],b+(8*8*16*i),128,ulong2(0,0));
          simdgroup_multiply_accumulate(acc, x[i], y[i], acc);
      }}
      simdgroup_store(acc,res,128,ulong2(0,0));
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

    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(b_buffer, 0, 2)
    encoder.setBuffer_offset_atIndex_(res_buffer, 0, 0)
    threadGroupSize = pipeline_state.maxTotalThreadsPerThreadgroup()
    if dim*dim < threadGroupSize:
        threadGroupSize = dim*dim
    print("max threadGroupSize =",pipeline_state.maxTotalThreadsPerThreadgroup())
    threadsPerGrid = Metal.MTLSizeMake(512,32,1)
    threadsPerThreadGroup = Metal.MTLSizeMake(32,32,1)
    encoder.dispatchThreads_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup) #1thread for now?
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    output = np.asarray(res_buffer.contents().as_buffer(res.nbytes))
    output = np.frombuffer(output, dtype=np.float32)
    return output