import objc
import Metal
import Foundation

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
kernel void matmul(device const float* inA,
                       device const float* inB,
                       device float* result,
                        device int& length,
                       uint index [[thread_position_in_grid]])
{{   
    int row = index / length; // unhardcode
    int col = index % length;
    
    float total = 0;
    if(row < length && col < length)
    {{
        for(int i = 0; i < length; i++)
        {{
            total += inA[(row * length) + i] * inB[col + (i * length)];
        }}
        result[(row * length) + col] = total;
    }}
}}"""

options = Metal.MTLCompileOptions.alloc().init()
library = unwrap(device.newLibraryWithSource_options_error_(prg, options, None))
fxn = library.newFunctionWithName_("matmul")
pipeline_state = device.newComputePipelineStateWithFunction_error_(fxn, None)
encoder.setComputePipelineState_(pipeline_state)

#print(type(defaultLibrary))
matmulFunction = library.newFunctionWithName_("matmul")

print("hello")