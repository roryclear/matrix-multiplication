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