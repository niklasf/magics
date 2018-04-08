#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>

#ifdef BISHOP
static const int DELTAS[] = { 9, 7, -9, -7, 0 };
static const char PCH = 'b';
#define FIXED_SHIFT 9
#else
static const int DELTAS[] = { 8, 1, -8, -1, 0 };
static const char PCH = 'r';
#define FIXED_SHIFT 12
#endif

#ifndef SQUARE
#error SQUARE not defined
#endif

#ifdef EASY_SHIFT
#define HARD_SHIFT ((EASY_SHIFT) - 1)
#else
#error EASY_SHIFT not defined
#endif

#define CHUNK_SIZE 36 // 2^36
#define SLICE_SIZE 30 // 2^30

static long get_nanos() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long) ts.tv_sec * 1000000000L + ts.tv_nsec;
}

static long get_millis() {
    return get_nanos() / 1000000L;
}

static const uint64_t BB_RANKS[8] = {
    UINT64_C(0xff) << (8 * 0),
    UINT64_C(0xff) << (8 * 1),
    UINT64_C(0xff) << (8 * 2),
    UINT64_C(0xff) << (8 * 3),
    UINT64_C(0xff) << (8 * 4),
    UINT64_C(0xff) << (8 * 5),
    UINT64_C(0xff) << (8 * 6),
    UINT64_C(0xff) << (8 * 7),
};

static const uint64_t BB_FILES[8] = {
    UINT64_C(0x101010101010101) << 0,
    UINT64_C(0x101010101010101) << 1,
    UINT64_C(0x101010101010101) << 2,
    UINT64_C(0x101010101010101) << 3,
    UINT64_C(0x101010101010101) << 4,
    UINT64_C(0x101010101010101) << 5,
    UINT64_C(0x101010101010101) << 6,
    UINT64_C(0x101010101010101) << 7,
};

static inline int popcount(uint64_t bb) {
    return __builtin_popcountll(bb);
}

static inline int lsb(uint64_t b) {
    return __builtin_ctzll(b);
}

static inline int max(int a, int b) {
    return a > b ? a : b;
}

static inline int square_rank(int square) {
    return square >> 3;
}

static inline int square_file(int square) {
    return square & 7;
}

static int square_distance(int a, int b) {
    return max(abs(square_rank(a) - square_rank(b)),
               abs(square_file(a) - square_file(b)));
}

static uint64_t sliding_attack(const int deltas[], int square, uint64_t occupied) {
    uint64_t attack = 0;

    for (int i = 0; deltas[i]; i++) {
        for (int s = square + deltas[i];
             0 <= s && s < 64 && square_distance(s, s - deltas[i]) == 1;
             s += deltas[i])
        {
            attack |= UINT64_C(1) << s;

            if (occupied & (UINT64_C(1) << s))
                break;
        }
    }

    return attack;
}

static uint64_t square_mask(int square) {
    uint64_t edges = (((BB_RANKS[0] | BB_RANKS[7]) & ~BB_RANKS[square_rank(square)]) |
                      ((BB_FILES[0] | BB_FILES[7]) & ~BB_FILES[square_file(square)]));

    uint64_t range = sliding_attack(DELTAS, square, 0);
    return range & ~edges;
}

typedef struct {
    uint64_t occupied;
    uint64_t attack;
    uint64_t failures;
} reference_t;

static uint64_t magic = 0;

static reference_t refs[1 << EASY_SHIFT];

void init_references() {
    const uint64_t mask = square_mask(SQUARE);
    uint64_t b = 0;
    int size = 0;
    do {
        refs[size].occupied = b;
        refs[size].attack = sliding_attack(DELTAS, SQUARE, b);
        refs[size].failures = 0;
        size++;
        b = (b - mask) & mask;
    } while (b);
    assert(size == 1 << EASY_SHIFT);
}

static int cmp_failures(const void *a, const void *b) {
    reference_t *ra = (reference_t *) a, *rb = (reference_t *) b;
    if (ra->failures < rb->failures) return 1;
    else if (ra->failures == rb->failures) return 0;
    else return -1;
}

static void reorder_references() {
    // Order references that fail often at the top to fail candidates as
    // early as possible.
    const int size = 1 << EASY_SHIFT;
    qsort(refs, size, sizeof(reference_t), cmp_failures);
    for (int i = 0; i < size; i++) refs[i].failures = 0;
}

void signal_handler(int sig) {
    if (sig != SIGUSR1) return;
    fprintf(stderr, "Got SIGUSR1 at 0x%lx\n", magic);
    signal(SIGUSR1, signal_handler);
}

int main(int argc, char *argv[]) {
    // Initialize.
    init_references();

    const int PERIOD = 64 - lsb(square_mask(SQUARE));
    const int LOWER = 64 - lsb(square_mask(SQUARE)) - HARD_SHIFT;

    const int NUM_CHUNKS = UINT64_C(1) << (PERIOD - CHUNK_SIZE);
    const int NUM_SLICES = UINT64_C(1) << (CHUNK_SIZE - SLICE_SIZE);

    // Parse chunk and compute range to test.
    if (argc != 2) {
        fprintf(stderr, "usage: ./magics <chunk>");
        return 64;
    }

    int chunk = atoi(argv[1]);
    bool profile = false;

    if (strcmp(argv[1], "--profile") == 0) {
        chunk = NUM_CHUNKS - 1;
        profile = true;
    }

    assert(chunk < NUM_CHUNKS);
    magic = ((uint64_t) chunk) << CHUNK_SIZE;

    const uint64_t CHUNK_END = magic + (UINT64_C(1) << CHUNK_SIZE);

    // Send SIGUSR1 to check the status.
    signal(SIGUSR1, signal_handler);

    // Start search.
    fprintf(stderr, "Searching %c-%d chunk %d/%d (0x%lx to 0x%lx, L = 2^%d, P = 2^%d) in %d slices ...\n",
            PCH, SQUARE, chunk, NUM_CHUNKS - 1, magic, CHUNK_END, LOWER, PERIOD, NUM_SLICES);

    // Exit early if no magics can be in range.
    if (CHUNK_END <= (UINT64_C(1) << LOWER)) {
        fprintf(stderr, "sq: %c-%d chunk: %d/%d skipped for lower bound: 0x%lx < 0x%lx\n",
                PCH, SQUARE, chunk, NUM_CHUNKS - 1, magic, UINT64_C(1) << LOWER);
        return 0;
    }

    // Search for magics.
    int slice = 0;

    uint64_t table[1 << FIXED_SHIFT] = { 0 };
    uint64_t age[1 << FIXED_SHIFT] = { 0 };

    while (magic < CHUNK_END) {
        long start = get_millis();
        const uint64_t slice_start = magic;
        const uint64_t slice_end = magic + (UINT64_C(1) << SLICE_SIZE);

        for (; magic < slice_end; magic++) {
            int ref;

            for (ref = 0; ref < (1 << EASY_SHIFT); ref++) {
                uint64_t product = magic * refs[ref].occupied;
                unsigned idx = product >> (64 - EASY_SHIFT);

                if (age[idx] != magic) {
                    age[idx] = magic;
                    table[idx] = refs[ref].attack;
                } else if (table[idx] != refs[ref].attack) {
                    refs[ref].failures++;
                    break;
                }
            }

            // Found a magic.
            if (ref == 1 << EASY_SHIFT) printf("0x%lx\n", magic);
        }

        reorder_references();

        long duration = get_millis() - start;
        fprintf(stderr, "sq: %c-%d chunk: %d/%d slice: %d/%d (0x%lx to 0x%lx) t: %lu ms.\n",
                PCH, SQUARE, chunk, NUM_CHUNKS - 1, slice, NUM_SLICES - 1, slice_start, slice_end, duration);
        slice++;

        if (profile) {
            fprintf(stderr, "Profiling done: Finished one slice.\n");
            return 0;
        }
    }

    // Done.
    fprintf(stderr, "Chunk %c-%d %d/%d completed.\n", PCH, SQUARE, chunk, NUM_CHUNKS - 1);
    return 0;
}
