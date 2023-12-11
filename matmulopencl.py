import numpy as np
import pyopencl as cl

ys = 2;
xs = 16;
platform = cl.get_platforms()
my_gpu_devices = platform[0].get_devices(device_type=cl.device_type.GPU)
ctx = cl.Context(devices=my_gpu_devices)
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags
local_0 = 256

def matmul(a,b):
	dim = len(a)
	a = a.flatten()
	b = b.flatten()
	a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
	b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

	res_np = np.empty([dim, dim]).astype(np.float32).flatten()
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * dim * 4))

	prg = cl.Program(ctx, f"""
	__kernel void matmul(
	    __global const float *a, __global const float *b, __global float *res)
	{{
	int lidx0 = get_local_id(0);
	int startY = lidx0 * ({dim} / {local_0});
	int endY = startY + ({dim} / {local_0});
	for(int y = startY; y < endY; y+={ys}) {{
		for(int x = 0; x < {dim}; x+={xs}) {{
			float acc[2][16];
			acc[0][0] = 0;
			acc[0][1] = 0;
			acc[0][2] = 0;
			acc[0][3] = 0;
			acc[0][4] = 0;
			acc[0][5] = 0;
			acc[0][6] = 0;
			acc[0][7] = 0;
			acc[0][8] = 0;
			acc[0][9] = 0;
			acc[0][10] = 0;
			acc[0][11] = 0;
			acc[0][12] = 0;
			acc[0][13] = 0;
			acc[0][14] = 0;
			acc[0][15] = 0;

			acc[1][0] = 0;
			acc[1][1] = 0;
			acc[1][2] = 0;
			acc[1][3] = 0;
			acc[1][4] = 0;
			acc[1][5] = 0;
			acc[1][6] = 0;
			acc[1][7] = 0;
			acc[1][8] = 0;
			acc[1][9] = 0;
			acc[1][10] = 0;
			acc[1][11] = 0;
			acc[1][12] = 0;
			acc[1][13] = 0;
			acc[1][14] = 0;
			acc[1][15] = 0;

			for(int k = 0; k < {dim}; k++) {{
				acc[0][0] += a[y * {dim} + k] * b[k * {dim} + x];
				acc[0][1] += a[y * {dim} + k] * b[k * {dim} + (x + 1)];
				acc[0][2] += a[y * {dim} + k] * b[k * {dim} + (x + 2)];
				acc[0][3] += a[y * {dim} + k] * b[k * {dim} + (x + 3)];
				acc[0][4] += a[y * {dim} + k] * b[k * {dim} + (x + 4)];
				acc[0][5] += a[y * {dim} + k] * b[k * {dim} + (x + 5)];
				acc[0][6] += a[y * {dim} + k] * b[k * {dim} + (x + 6)];
				acc[0][7] += a[y * {dim} + k] * b[k * {dim} + (x + 7)];
				acc[0][8] += a[y * {dim} + k] * b[k * {dim} + (x + 8)];
				acc[0][9] += a[y * {dim} + k] * b[k * {dim} + (x + 9)];
				acc[0][10] += a[y * {dim} + k] * b[k * {dim} + (x + 10)];
				acc[0][11] += a[y * {dim} + k] * b[k * {dim} + (x + 11)];
				acc[0][12] += a[y * {dim} + k] * b[k * {dim} + (x + 12)];
				acc[0][13] += a[y * {dim} + k] * b[k * {dim} + (x + 13)];
				acc[0][14] += a[y * {dim} + k] * b[k * {dim} + (x + 14)];
				acc[0][15] += a[y * {dim} + k] * b[k * {dim} + (x + 15)];

				acc[1][0] += a[(y + 1) * {dim} + k] * b[k * {dim} + x];
				acc[1][1] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 1)];
				acc[1][2] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 2)];
				acc[1][3] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 3)];
				acc[1][4] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 4)];
				acc[1][5] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 5)];
				acc[1][6] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 6)];
				acc[1][7] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 7)];
				acc[1][8] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 8)];
				acc[1][9] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 9)];
				acc[1][10] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 10)];
				acc[1][11] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 11)];
				acc[1][12] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 12)];
				acc[1][13] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 13)];
				acc[1][14] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 14)];
				acc[1][15] += a[(y + 1) * {dim} + k] * b[k * {dim} + (x + 15)];

			}}
			res[y * {dim} + x] = acc[0][0];
			res[y * {dim} + x + 1] = acc[0][1];
			res[y * {dim} + x + 2] = acc[0][2];
			res[y * {dim} + x + 3] = acc[0][3];
			res[y * {dim} + x + 4] = acc[0][4];
			res[y * {dim} + x + 5] = acc[0][5];
			res[y * {dim} + x + 6] = acc[0][6];
			res[y * {dim} + x + 7] = acc[0][7];
			res[y * {dim} + x + 8] = acc[0][8];
			res[y * {dim} + x + 9] = acc[0][9];
			res[y * {dim} + x + 10] = acc[0][10];
			res[y * {dim} + x + 11] = acc[0][11];
			res[y * {dim} + x + 12] = acc[0][12];
			res[y * {dim} + x + 13] = acc[0][13];
			res[y * {dim} + x + 14] = acc[0][14];
			res[y * {dim} + x + 15] = acc[0][15];

			res[(y + 1) * {dim} + x] = acc[1][0];
			res[(y + 1) * {dim} + x + 1] = acc[1][1];
			res[(y + 1) * {dim} + x + 2] = acc[1][2];
			res[(y + 1) * {dim} + x + 3] = acc[1][3];
			res[(y + 1) * {dim} + x + 4] = acc[1][4];
			res[(y + 1) * {dim} + x + 5] = acc[1][5];
			res[(y + 1) * {dim} + x + 6] = acc[1][6];
			res[(y + 1) * {dim} + x + 7] = acc[1][7];
			res[(y + 1) * {dim} + x + 8] = acc[1][8];
			res[(y + 1) * {dim} + x + 9] = acc[1][9];
			res[(y + 1) * {dim} + x + 10] = acc[1][10];
			res[(y + 1) * {dim} + x + 11] = acc[1][11];
			res[(y + 1) * {dim} + x + 12] = acc[1][12];
			res[(y + 1) * {dim} + x + 13] = acc[1][13];
			res[(y + 1) * {dim} + x + 14] = acc[1][14];
			res[(y + 1) * {dim} + x + 15] = acc[1][15];

		}}
	}}
	}}
	""").build()
	
	knl = prg.matmul
	knl(queue, (local_0,1), (local_0,1), a_g, b_g, res_g) #todo check shape
	cl.enqueue_copy(queue, res_np, res_g)
	return res_np

def matmul2(a,b):
  dim = len(a)
  a = a.flatten()
  b = b.flatten()
  a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
  b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)

  res_np = np.empty([dim, dim]).astype(np.float32).flatten()
  res_g = cl.Buffer(ctx, mf.WRITE_ONLY, (dim * dim * 4))

  prg = cl.Program(ctx, f"""
  __kernel void matmul(
      __global const float *a, __global const float *b, __global float *res)
  {{
    float4 acc0 = (float4)(0.0f,0.0f,0.0f,0.0f);
    float4 acc1 = (float4)(0.0f,0.0f,0.0f,0.0f);
    float4 acc2 = (float4)(0.0f,0.0f,0.0f,0.0f);
    float4 acc3 = (float4)(0.0f,0.0f,0.0f,0.0f);
    a+=get_local_id(1)*4*{dim};
    b+=get_local_id(0)*4;
    res+=get_local_id(1)*4*{dim} + get_local_id(0)*4;
    for(int k = 0; k < {dim}/4; k++) {{
      float4 a0 = (float4)(*((__global float4*)(a+0+k*4)));
      float4 a1 = (float4)(*((__global float4*)(a+{dim}+k*4)));
      float4 a2 = (float4)(*((__global float4*)(a+{dim}*2+k*4)));
      float4 a3 = (float4)(*((__global float4*)(a+{dim}*3+k*4)));
      float4 b0 = (float4)(*((__global float4*)(b+0+k*4*{dim})));
      float4 b1 = (float4)(*((__global float4*)(b+{dim}+k*4*{dim})));
      float4 b2 = (float4)(*((__global float4*)(b+{dim}*2+k*4*{dim})));
      float4 b3 = (float4)(*((__global float4*)(b+{dim}*3+k*4*{dim})));

      (acc0).x = mad((a0).x,(b0).x,(acc0).x);
      (acc0).x = mad((a0).y,(b1).x,(acc0).x);
      (acc0).x = mad((a0).z,(b2).x,(acc0).x);
      (acc0).x = mad((a0).w,(b3).x,(acc0).x);
      (acc0).y = mad((a0).x,(b0).y,(acc0).y);
      (acc0).y = mad((a0).y,(b1).y,(acc0).y);
      (acc0).y = mad((a0).z,(b2).y,(acc0).y);
      (acc0).y = mad((a0).w,(b3).y,(acc0).y);
      (acc0).z = mad((a0).x,(b0).z,(acc0).z);
      (acc0).z = mad((a0).y,(b1).z,(acc0).z);
      (acc0).z = mad((a0).z,(b2).z,(acc0).z);
      (acc0).z = mad((a0).w,(b3).z,(acc0).z);
      (acc0).w = mad((a0).x,(b0).w,(acc0).w);
      (acc0).w = mad((a0).y,(b1).w,(acc0).w);
      (acc0).w = mad((a0).z,(b2).w,(acc0).w);
      (acc0).w = mad((a0).w,(b3).w,(acc0).w);

      (acc1).x = mad((a1).x,(b0).x,(acc1).x);
      (acc1).x = mad((a1).y,(b1).x,(acc1).x);
      (acc1).x = mad((a1).z,(b2).x,(acc1).x);
      (acc1).x = mad((a1).w,(b3).x,(acc1).x);
      (acc1).y = mad((a1).x,(b0).y,(acc1).y);
      (acc1).y = mad((a1).y,(b1).y,(acc1).y);
      (acc1).y = mad((a1).z,(b2).y,(acc1).y);
      (acc1).y = mad((a1).w,(b3).y,(acc1).y);
      (acc1).z = mad((a1).x,(b0).z,(acc1).z);
      (acc1).z = mad((a1).y,(b1).z,(acc1).z);
      (acc1).z = mad((a1).z,(b2).z,(acc1).z);
      (acc1).z = mad((a1).w,(b3).z,(acc1).z);
      (acc1).w = mad((a1).x,(b0).w,(acc1).w);
      (acc1).w = mad((a1).y,(b1).w,(acc1).w);
      (acc1).w = mad((a1).z,(b2).w,(acc1).w);
      (acc1).w = mad((a1).w,(b3).w,(acc1).w);

      (acc2).x = mad((a2).x,(b0).x,(acc2).x);
      (acc2).x = mad((a2).y,(b1).x,(acc2).x);
      (acc2).x = mad((a2).z,(b2).x,(acc2).x);
      (acc2).x = mad((a2).w,(b3).x,(acc2).x);
      (acc2).y = mad((a2).x,(b0).y,(acc2).y);
      (acc2).y = mad((a2).y,(b1).y,(acc2).y);
      (acc2).y = mad((a2).z,(b2).y,(acc2).y);
      (acc2).y = mad((a2).w,(b3).y,(acc2).y);
      (acc2).z = mad((a2).x,(b0).z,(acc2).z);
      (acc2).z = mad((a2).y,(b1).z,(acc2).z);
      (acc2).z = mad((a2).z,(b2).z,(acc2).z);
      (acc2).z = mad((a2).w,(b3).z,(acc2).z);
      (acc2).w = mad((a2).x,(b0).w,(acc2).w);
      (acc2).w = mad((a2).y,(b1).w,(acc2).w);
      (acc2).w = mad((a2).z,(b2).w,(acc2).w);
      (acc2).w = mad((a2).w,(b3).w,(acc2).w);

      (acc3).x = mad((a3).x,(b0).x,(acc3).x);
      (acc3).x = mad((a3).y,(b1).x,(acc3).x);
      (acc3).x = mad((a3).z,(b2).x,(acc3).x);
      (acc3).x = mad((a3).w,(b3).x,(acc3).x);
      (acc3).y = mad((a3).x,(b0).y,(acc3).y);
      (acc3).y = mad((a3).y,(b1).y,(acc3).y);
      (acc3).y = mad((a3).z,(b2).y,(acc3).y);
      (acc3).y = mad((a3).w,(b3).y,(acc3).y);
      (acc3).z = mad((a3).x,(b0).z,(acc3).z);
      (acc3).z = mad((a3).y,(b1).z,(acc3).z);
      (acc3).z = mad((a3).z,(b2).z,(acc3).z);
      (acc3).z = mad((a3).w,(b3).z,(acc3).z);
      (acc3).w = mad((a3).x,(b0).w,(acc3).w);
      (acc3).w = mad((a3).y,(b1).w,(acc3).w);
      (acc3).w = mad((a3).z,(b2).w,(acc3).w);
      (acc3).w = mad((a3).w,(b3).w,(acc3).w);
    }}
    *((__global float4*)(res+0)) = (float4)(float4)((acc0).x,(acc0).y,(acc0).z,(acc0).w);
    *((__global float4*)(res+{dim})) = (float4)(float4)((acc1).x,(acc1).y,(acc1).z,(acc1).w);
    *((__global float4*)(res+{dim}*2)) = (float4)(float4)((acc2).x,(acc2).y,(acc2).z,(acc2).w);
    *((__global float4*)(res+{dim}*3)) = (float4)(float4)((acc3).x,(acc3).y,(acc3).z,(acc3).w);
  }}
  """).build()
  
  knl = prg.matmul
  knl(queue, (4,4), (4,4), a_g, b_g, res_g) #todo check shape
  cl.enqueue_copy(queue, res_np, res_g)
  return res_np