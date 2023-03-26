import objc
import Metal
import Foundation
import numpy as np
import struct

def matmul(a,b):
    aRows = a.shape[0]
    aCols = a.shape[1]
    bRows = b.shape[0]
    bCols = b.shape[1]
    device = Metal.MTLCreateSystemDefaultDevice()
    mtl_queue = device.newCommandQueue()
    command_buffer = mtl_queue.commandBuffer()
    encoder = command_buffer.computeCommandEncoder()

    prg = f"""#include <metal_stdlib>
    using namespace metal;
    kernel void matmul(device float *a,
                        device float *b,
                        device int& aRows,
                        device int& aCols,
                        device int& bCols,
                        device float *res,
    uint index [[thread_position_in_grid]])
    {{     
          int row = index / bCols;
          int col = index % bCols;
          float total = 0;
          if(row < aRows && col < bCols) 
          {{
            for(int i = 0; i < aCols; i++)
            {{
              total += a[row * aCols + i] * b[col + i * bCols];
            }}
            res[row * bCols + col] = total;
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

    aRows = np.int32(aRows)
    aCols = np.int32(aCols)
    bCols = np.int32(bCols)
    aRows_buffer = device.newBufferWithLength_options_(4 ,1)
    aCols_buffer = device.newBufferWithLength_options_(4 ,1)
    bCols_buffer = device.newBufferWithLength_options_(4 ,1)
    for i in range(4):
        aRows_buffer.contents().__setitem__(i,aRows.tobytes()[i].to_bytes(1,'big'))
        aCols_buffer.contents().__setitem__(i,aCols.tobytes()[i].to_bytes(1,'big'))
        bCols_buffer.contents().__setitem__(i,bCols.tobytes()[i].to_bytes(1,'big'))

    res = np.empty([aRows, bCols]).astype(np.float32).flatten()
    res_buffer = device.newBufferWithLength_options_(res.nbytes ,1)

    encoder.setBuffer_offset_atIndex_(a_buffer, 0, 0)
    encoder.setBuffer_offset_atIndex_(b_buffer, 0, 1)
    encoder.setBuffer_offset_atIndex_(aRows_buffer, 0, 2)
    encoder.setBuffer_offset_atIndex_(aCols_buffer, 0, 3)
    encoder.setBuffer_offset_atIndex_(bCols_buffer, 0, 4)
    encoder.setBuffer_offset_atIndex_(res_buffer, 0, 5)
    threadGroupSize = pipeline_state.maxTotalThreadsPerThreadgroup()
    if aRows*bCols < threadGroupSize:
        threadGroupSize = aRows*bCols
    encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(aRows*bCols,1,1), Metal.MTLSizeMake(threadGroupSize,1,1))
    encoder.endEncoding()
    command_buffer.commit()
    command_buffer.waitUntilCompleted()

    output = np.asarray(res_buffer.contents().as_buffer(res.nbytes))
    output = np.frombuffer(output, dtype=np.float32)
    return output