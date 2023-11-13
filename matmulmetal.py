import objc
import Metal
import Foundation
import numpy as np
import struct

device = Metal.MTLCreateSystemDefaultDevice()

def matmul(a,b):
    dim = a.shape[0]
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
       res += gid.x * 4 * 8 + (lid.y + gid.y * 32) * 8*2*{dim};
       a += (lid.y + gid.y * 32) * 8*2*{dim};
       b += gid.x * 8*4;

      simdgroup_float8x8 x[2];
      simdgroup_float8x8 y[4];
      simdgroup_float8x8 acc[2][4];
      for(int j = 0; j < 2; j++) {{
        for(int i = 0; i < 4; i++) {{
            acc[j][i] = simdgroup_float8x8(0);
        }}
      }}

      for(int i = 0; i < {dim}; i+=8) {{
          simdgroup_load(x[0],a+i,{dim},ulong2(0,0));
          simdgroup_load(x[1],a+i+8*{dim},{dim},ulong2(0,0));
          simdgroup_load(y[0],b+{dim}*i,{dim},ulong2(0,0));
          simdgroup_load(y[1],b+{dim}*i+8,{dim},ulong2(0,0));
          simdgroup_load(y[2],b+{dim}*i+8*2,{dim},ulong2(0,0));
          simdgroup_load(y[3],b+{dim}*i+8*3,{dim},ulong2(0,0));

          simdgroup_multiply_accumulate(acc[0][0], x[0], y[0], acc[0][0]);
          simdgroup_multiply_accumulate(acc[0][1], x[0], y[1], acc[0][1]);
          simdgroup_multiply_accumulate(acc[0][2], x[0], y[2], acc[0][2]);
          simdgroup_multiply_accumulate(acc[0][3], x[0], y[3], acc[0][3]);
          simdgroup_multiply_accumulate(acc[1][0], x[1], y[0], acc[1][0]);
          simdgroup_multiply_accumulate(acc[1][1], x[1], y[1], acc[1][1]);
          simdgroup_multiply_accumulate(acc[1][2], x[1], y[2], acc[1][2]);
          simdgroup_multiply_accumulate(acc[1][3], x[1], y[3], acc[1][3]);
      }}
      simdgroup_store(acc[0][0],res,{dim},ulong2(0,0));
      simdgroup_store(acc[0][1],res+8,{dim},ulong2(0,0));
      simdgroup_store(acc[1][0],res+{dim}*8,{dim},ulong2(0,0));
      simdgroup_store(acc[1][1],res+{dim}*8+8,{dim},ulong2(0,0));
      simdgroup_store(acc[0][2],res+8*2,{dim},ulong2(0,0));
      simdgroup_store(acc[0][3],res+8*3,{dim},ulong2(0,0));
      simdgroup_store(acc[1][2],res+{dim}*8+8*2,{dim},ulong2(0,0));
      simdgroup_store(acc[1][3],res+{dim}*8+8*3,{dim},ulong2(0,0));      
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

    res_buffer = device.newBufferWithLength_options_(a.nbytes ,1)

    encoder.setBuffer_offset_atIndex_(res_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(b_buffer, 0, 2)
    threadsPerGrid = Metal.MTLSizeMake(64,4,1)
    threadsPerThreadGroup = Metal.MTLSizeMake(32,32,1)
    encoder.dispatchThreadgroups_threadsPerThreadgroup_(threadsPerGrid, threadsPerThreadGroup) #1thread for now?
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()
    output = np.asarray(res_buffer.contents().as_buffer(a.nbytes))
    output = np.frombuffer(output, dtype=np.float32)
    return output