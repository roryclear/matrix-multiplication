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
print("HERE")


encoder.setComputePipelineState_(pipeline_state)
print("HERE")
#a = pipeline_state.maxTotalThreadsPerThreadgroup()
#print(a)

a = np.random.rand(128).astype(np.float32)
a[0] = np.float32(420)
a_buffer = device.newBufferWithLength_options_(a.nbytes ,1)

y = a_buffer.contents()
print(y.__getitem__(0),y.__getitem__(1),y.__getitem__(2),y.__getitem__(3))

print("x42??")
a_buffer.contents().__setitem__(0,b'\x42')

y = a_buffer.contents()
print(y.__getitem__(0),y.__getitem__(1),y.__getitem__(2),y.__getitem__(3))

#a_buffer.contents()

encoder.setBuffer_offset_atIndex_(a_buffer, 0, 0)

encoder.dispatchThreads_threadsPerThreadgroup_(Metal.MTLSizeMake(1,1,1), Metal.MTLSizeMake(1,1,1))
encoder.endEncoding()
command_buffer.commit()
command_buffer.waitUntilCompleted()

s = ""
n = 0

x = a_buffer.contents
y = a_buffer.contents()
print(y)
print("after function")
print(y.__getitem__(0),y.__getitem__(1),y.__getitem__(2),y.__getitem__(3))
y.__setitem__(0,b'\x69')
print(y.__getitem__(0),y.__getitem__(1),y.__getitem__(2),y.__getitem__(3))

s = y.__getitem__(0) + y.__getitem__(1) + y.__getitem__(2) + y.__getitem__(3)
print("s =",s)
n = struct.unpack('1f', s)

print("n =",n)

#a_buffer.contents().__setitem__(0,1)

#print(a_buffer.contents())
#for i in a_buffer.contents():
#    if "x00" not in str(i):
#        print(str(i)[2:len(str(i))-1])
#        s += str(i)[2:len(str(i))-1]
#    n+=1
#    #print(n)
#print(s)
#print(type(defaultLibrary))

print("hello!")