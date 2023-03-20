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
kernel void matmul(device float* a,
uint index [[thread_position_in_grid]])
{{   
    a[0] = 23;
}}"""

options = Metal.MTLCompileOptions.alloc().init()
library = unwrap(device.newLibraryWithSource_options_error_(prg, options, None))
fxn = library.newFunctionWithName_("matmul")
pipeline_state = unwrap(device.newComputePipelineStateWithFunction_error_(fxn, None))


encoder.setComputePipelineState_(pipeline_state)

a = np.random.randn(128).astype(np.float32)
a_buffer = device.newBufferWithLength_options_(a.nbytes ,1)
t = a_buffer.contents()
m = t.as_buffer(a.nbytes)
m[:] = bytes(a)

b = np.random.randn(128).astype(np.float32)
b_buffer = device.newBufferWithLength_options_(b.nbytes ,1)
t = b_buffer.contents()
m = t.as_buffer(b.nbytes)
m[:] = bytes(b)


encoder.setBuffer_offset_atIndex_(a_buffer, 0, 0)

encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(1,1,1), Metal.MTLSizeMake(1,1,1))
encoder.endEncoding()
command_buffer.commit()
command_buffer.waitUntilCompleted()

y = a_buffer.contents()

s = y.__getitem__(0)
for i in range(1,512):
    s += y.__getitem__(i)
output = struct.unpack('128f',s)
output = np.asarray(output)
print(output)


print("done")