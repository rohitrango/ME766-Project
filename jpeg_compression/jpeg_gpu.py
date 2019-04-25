import sys
import cv2 as cv
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda

from pycuda.compiler import SourceModule

mod = SourceModule("""
__device__ __constant__ float dct_matrix[8][8];

__global__ void RGBtoYUV(float *R, float *G, float*B){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    float y = 0.299*R[i * gridDim.x *blockDim.x + j] + 0.587*G[i * gridDim.x *blockDim.x + j] + 0.114*B[i * gridDim.x *blockDim.x + j];
    float u = -0.14713*R[i * gridDim.x *blockDim.x + j] - 0.28886*G[i * gridDim.x *blockDim.x + j] + 0.436*B[i * gridDim.x *blockDim.x + j];
    float v = 0.615*R[i * gridDim.x *blockDim.x + j] - 0.51499*G[i * gridDim.x *blockDim.x + j] - 0.10001*B[i * gridDim.x *blockDim.x + j];
    R[i * gridDim.x *blockDim.x + j] = y;
    G[i * gridDim.x *blockDim.x + j] = u;
    B[i * gridDim.x *blockDim.x + j] = v;
}

__global__ void compression(float *image)
{
    __shared__ float patch[8][8];
    __shared__ float dct_intermediate[8][8];
    __shared__ float dct[8][8];
    
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;
    int i_prime = i%8;
    int j_prime = j%8;

    dct[i_prime][j_prime] = 0.f;
    dct_intermediate[i_prime][j_prime] = 0.f;
    patch[i_prime][j_prime] = image[i * gridDim.x *blockDim.x + j];

    __syncthreads();

    for(int k=0; k<8; k++)
        dct_intermediate[i_prime][j_prime] += dct_matrix[i_prime][k] * patch[k][j_prime];

    __syncthreads();
    
    for(int k=0; k<8; k++)
        dct[i_prime][j_prime] += dct_intermediate[i_prime][k] * dct_matrix[j_prime][k];

    __syncthreads();
    
    image[i * gridDim.x *blockDim.x + j] = dct[i_prime][j_prime];

}
""")

dct_matrix = np.zeros((8, 8))
for i in range(8):
    for j in range(8):
        if( i == 0):
            dct_matrix[i, j] = 1/np.sqrt(8)
        else:
            dct_matrix[i, j] = .5*np.cos((np.pi*(2*j+1)*i)/16)
dct_matrix = dct_matrix.astype(np.float32)
dct_matrix_device, _ = mod.get_global("dct_matrix")
cuda.memcpy_htod(dct_matrix_device, dct_matrix)

image_raw = cv.imread('4k.jpg')
rows_raw, cols_raw, channels = image_raw.shape
rows = int(np.ceil(rows_raw/8)*8)
cols = int(np.ceil(cols_raw/8)*8)
image = np.ones((rows, cols, channels))
image[:rows_raw, :cols_raw, :] = image_raw
image = image.astype(np.float32)
R_device = cuda.mem_alloc(image[:, :, 2].nbytes)
G_device = cuda.mem_alloc(image[:, :, 1].nbytes)
B_device = cuda.mem_alloc(image[:, :, 0].nbytes)
R_doubled = np.copy(image[:, :, 2])
G_doubled = np.copy(image[:, :, 1])
B_doubled = np.copy(image[:, :, 0])
cuda.memcpy_htod(R_device, R_doubled)
cuda.memcpy_htod(G_device, G_doubled)
cuda.memcpy_htod(B_device, B_doubled)

RGBtoYUV = mod.get_function("RGBtoYUV");
RGBtoYUV(R_device, G_device, B_device, grid=(int(rows/8), int(cols/8)),  block=(8, 8, 1))

compress = mod.get_function("compression")
compress(R_device, grid=(int(rows/8), int(cols/8)),  block=(8, 8, 1))
compress(G_device, grid=(int(rows/8), int(cols/8)),  block=(8, 8, 1))
compress(B_device, grid=(int(rows/8), int(cols/8)),  block=(8, 8, 1))

Y_doubled = np.empty_like(R_doubled)
U_doubled = np.empty_like(G_doubled)
V_doubled = np.empty_like(B_doubled)

cuda.memcpy_dtoh(Y_doubled, R_device)
cuda.memcpy_dtoh(U_doubled, G_device)
cuda.memcpy_dtoh(V_doubled, B_device)