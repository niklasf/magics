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
    assert(chunk_size % 8 == 0);
    int chunk = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;

    result[chunk] = 0;

    uint64_t magic = chunk_size * chunk;
    const uint64_t end_magic = magic + chunk_size;

    while (magic < end_magic) {
        bool r0 = check_magic(magic);
        bool r1 = check_magic(magic + 1);
        bool r2 = check_magic(magic + 2);
        bool r3 = check_magic(magic + 3);
        bool r4 = check_magic(magic + 4);
        bool r5 = check_magic(magic + 5);
        bool r6 = check_magic(magic + 6);
        bool r7 = check_magic(magic + 7);
        if (r0) { result[chunk] = magic; return; }
        if (r1) { result[chunk] = magic + 1; return; }
        if (r2) { result[chunk] = magic + 2; return; }
        if (r3) { result[chunk] = magic + 3; return; }
        if (r4) { result[chunk] = magic + 4; return; }
        if (r5) { result[chunk] = magic + 5; return; }
        if (r6) { result[chunk] = magic + 6; return; }
        if (r7) { result[chunk] = magic + 7; return; }
        magic += 8;
    }
}

int main() {
    printf("parallel search of %s with shift %d ...\n", SQUARE_NAME, SHIFT);

    uint64_t h_result[CHUNKS] = { 0 };
    uint64_t *d_result;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t) * CHUNKS));
    test_kernel<<<CHUNKS / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, hipStreamDefault>>>(d_result);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(h_result, d_result, sizeof(uint64_t) * CHUNKS, hipMemcpyDeviceToHost));

    printf("search of %s with shift %d complete.\n", SQUARE_NAME, SHIFT);

    for (int i = 0; i < CHUNKS; i++) {
        if (h_result[i]) printf("magic: 0x%lx\n", h_result[i]);
    }
}
