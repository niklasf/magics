#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define PERIOD (UINT64_C(1) << UINT64_C(42)) // h2

#define CHUNKS (1 << 19)
#define THREADS_PER_BLOCK 32

#include "generated.inc"

__global__ void test(uint64_t *result) {
    const uint64_t chunk_size = PERIOD / CHUNKS;
    assert(chunk_size * CHUNKS == PERIOD);
    int chunk = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    uint64_t magic = chunk_size * chunk;

    result[chunk] = 0;

    while (magic < chunk_size * (chunk + 1)) {
        if (check_magic(magic)) {
            result[chunk] = magic;
            return;
        }
        magic++;
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    printf("parallel search ...\n");

    uint64_t h_result[CHUNKS] = { 0 };
    uint64_t *d_result;
    gpuErrchk(cudaMalloc(&d_result, sizeof(uint64_t) * CHUNKS));
    test<<<CHUNKS / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_result);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_result, d_result, sizeof(uint64_t) * CHUNKS, cudaMemcpyDeviceToHost));

    printf("search complete.\n");

    for (int i = 0; i < CHUNKS; i++) {
        if (h_result[i]) printf("magic: 0x%lx\n", h_result[i]);
    }
}
