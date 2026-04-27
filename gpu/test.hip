// Copyright (c) 2018 Niklas Fiekas <niklas.fiekas@backscattering.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <stdbool.h>

#define CHUNKS (1 << 19)
#define THREADS_PER_BLOCK 64

#define HIP_CHECK(ret) if ((ret) != hipSuccess) { abort(); }

#include "generated.inc"

__global__ void test_kernel(uint64_t *result) {
    const uint64_t chunk_size = PERIOD / CHUNKS;
    assert(chunk_size * CHUNKS == PERIOD);
    int chunk = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    result[chunk] = 0;

    uint64_t magic = chunk_size * chunk;
    const uint64_t end_magic = magic + chunk_size;

    while (magic < end_magic) {
        if (check_magic(magic)) {
            result[chunk] = magic;
            return;
        }
        magic++;
    }
}

int main() {
    printf("parallel search ...\n");

    uint64_t h_result[CHUNKS] = { 0 };
    uint64_t *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t) * CHUNKS));
    test_kernel<<<CHUNKS / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, hipStreamDefault>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_result, d_result, sizeof(uint64_t) * CHUNKS, hipMemcpyDeviceToHost));

    printf("search complete.\n");

    for (int i = 0; i < CHUNKS; i++) {
        if (h_result[i]) printf("magic: 0x%lx\n", h_result[i]);
    }
}
