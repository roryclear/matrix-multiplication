import objc
import Metal
import Foundation
import numpy as np
import struct

def unwrap(x):
  ret, err = x
  assert err is None, str(err)
  return ret

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
library = unwrap(device.newLibraryWithSource_options_error_(prg, options, None))
fxn = library.newFunctionWithName_("matmul")
pipeline_state = unwrap(device.newComputePipelineStateWithFunction_error_(fxn, None))

print("max thread per threadgroup =",pipeline_state.maxTotalThreadsPerThreadgroup())

encoder.setComputePipelineState_(pipeline_state)

length = 128
size = length*length

a = np.random.randn(size).astype(np.float32)
a_buffer = device.newBufferWithLength_options_(a.nbytes ,1)
m = a_buffer.contents().as_buffer(a.nbytes)
m[:] = bytes(a)

b = np.random.randn(size).astype(np.float32)
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

y = res_buffer.contents()
s = y.__getitem__(0)
for i in range(1,size*4):
    s += y.__getitem__(i)
output = np.asarray(struct.unpack(str(size)+'f',s))
print(output)

answer = np.empty_like(a)
for r in range(length):
  for c in range(length):
    total = 0
    for n in range(length):
      total += a[r * length + n] * b[c + n * length]
    answer[r * length + c] = total

print(answer)

assert np.allclose(output, answer, 0.1) #todo are is cuda and opencl closer?
print("done")