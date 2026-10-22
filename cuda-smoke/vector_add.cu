// Minimal CUDA smoke test: vector add + device info + arch check.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);
    printf("Device: %s, SM count: %d, CC %d.%d\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    vecAdd<<<(n + 255) / 256, 256>>>(a, b, c, n);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); return 1; }

    int bad = 0;
    for (int i = 0; i < n; i++) if (c[i] != 3.0f) bad++;
    printf("%s (%d wrong of %d)\n", bad == 0 ? "PASS" : "FAIL", bad, n);
    return bad == 0 ? 0 : 1;
}
