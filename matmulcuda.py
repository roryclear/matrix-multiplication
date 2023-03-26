import pycuda.compiler as comp
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import math

def matmul(a,b):
    aRows = a.shape[0]
    aCols = a.shape[1]
    bCols = b.shape[1]
    a_g = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_g,a)
    b_g = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(b_g,b)
    res_np = np.empty([aRows, bCols]).astype(np.float32).flatten()
    res_g = cuda.mem_alloc(aRows * bCols * 4)
    cuda.memcpy_htod(res_g,res_np)

    mod = comp.SourceModule(
      """
    __global__ void matmul(float *a, float *b, int aRows, int aCols, int bCols, float *res)
  {
    const int gid = threadIdx.x + (blockDim.x * blockIdx.x);
    int row = gid / bCols;
    int col = gid % bCols;
    float total = 0;
    if(row < aRows && col < bCols) 
    {
      for(int i = 0; i < aCols; i++)
      {
        total += a[row * aCols + i] * b[col + i * bCols];
      }
      res[row * bCols + col] = total;
    }
  }
  """
  )
    matmulcuda = mod.get_function("matmul")
    numberOfThreads = \
        cuda.Device(0).get_attribute(pycuda._driver.device_attribute.MAX_THREADS_PER_BLOCK)
    ops = aRows * bCols
    bx = min(ops,numberOfThreads)
    gx = math.ceil(ops / numberOfThreads)
    matmulcuda(a_g,b_g,np.int32(aRows),np.int32(aCols),np.int32(bCols),res_g,block=(bx,1,1),grid=(gx,1))
    cuda.memcpy_dtoh(res_np,res_g)
    return res_np