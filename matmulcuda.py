import pycuda.compiler as comp
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def matmul(a,b,length):
    a_g = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_g,a)
    b_g = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_g,b)
    res_g = cuda.mem_alloc(a.nbytes)
    res_np = np.empty_like(a)#fix later
    cuda.memcpy_htod(res_g,res_np)

    mod = comp.SourceModule(
      """
    __global__ void matmul(float *a, float *b, int length, float *res)
  {
    const int gid = threadIdx.x + (blockDim.x * blockIdx.x);
    int row = gid / length;
    int col = gid % length;
    float total = 0;
    if(row < length && col < length) 
    {
      for(int i = 0; i < length; i++)
      {
        total += a[row * length + i] * b[col + i * length];
      }
      res[row * length + col] = total;
    }
  }
  """
  )
    matmulcuda = mod.get_function("matmul")
    numberOfThreads = \
        cuda.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)
    matmulcuda(a_g,b_g,np.int32(length),res_g,block=(1024,1,1),grid=(16,1)) #todo fix these sizes
    cuda.memcpy_dtoh(res_np,res_g)
    return res_np