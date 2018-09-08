#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#ifndef SQUARE
#error SQUARE not defined
#endif

#ifndef SHIFT
#error SHIFT not defined
#endif

#if defined(BISHOP)
const int DELTAS[] = { 7, -7, 9, -9, 0 };
#elif defined(ROOK)
const int DELTAS[] = { 8, 1, -8, -1, 0 };
#else
#error Neither BISHOP nor ROOK defined
#endif

#define MAX_SQUARES 12

#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

typedef struct {
    uint64_t occupied;
    uint64_t attack;
} reference_t;

static inline int lsb(uint64_t b) {
    assert(b);
    return __builtin_ctzll(b);
}

static inline int msb(uint64_t b) {
    assert(b);
    return 63 ^ __builtin_clzll(b);
}

static inline int square_rank(int square) {
    return square >> 3;
}

static inline int square_file(int square) {
    return square & 7;
}

static int square_distance(int a, int b) {
    return MAX(abs(square_rank(a) - square_rank(b)),
               abs(square_file(a) - square_file(b)));
}

static uint64_t sliding_attack(const int deltas[], const int square, uint64_t occupied) {
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

static size_t init_references(const uint64_t mask, reference_t * const refs) {
    uint64_t b = 0;
    int size = 0;
    do {
        refs[size].occupied = b;
        refs[size].attack = sliding_attack(DELTAS, SQUARE, b);
        size++;
        b = (b - mask) & mask;
    } while (b);
    return size;
}

static int attack_sets(const reference_t * refs, int size) {
    int sets = 0;
    for (int i = 0; i < size; i++) {
        sets++;
        for (int j = 0; j < i; j++) {
            if (refs[i].attack == refs[j].attack) {
                sets--;
                break;
            }
        }
    }
    return sets;
}

static uint64_t square_mask(const int deltas[], int square) {
    uint64_t edges = (((BB_RANKS[0] | BB_RANKS[7]) & ~BB_RANKS[square_rank(square)]) |
                      ((BB_FILES[0] | BB_FILES[7]) & ~BB_FILES[square_file(square)]));

    uint64_t range = sliding_attack(deltas, square, 0);
    return range & ~edges;
}

typedef struct {
    uint64_t mask;

    int bits;
    int prefix_bits;
    bool last;

    int unique_attacks;

    uint64_t min;
    uint64_t step;
    uint64_t max;

    reference_t refs[1 << MAX_SQUARES];
    int size;

    uint64_t age[1 << SHIFT];

    uint64_t stats;
} stack_t;

static stack_t stack[MAX_SQUARES];
static int unique_attacks;
static uint64_t num_magics = 0;

static void init_stack(uint64_t max_occupied) {
    uint64_t mask = 0;

    for (int i = 0; i < MAX_SQUARES; i++) {
        stack[i].prefix_bits = mask ? 64 - lsb(mask) : 0;

        mask ^= UINT64_C(1) << msb(max_occupied ^ mask);
        stack[i].mask = mask;
        stack[i].bits = 64 - lsb(mask);
        stack[i].last = mask == max_occupied;
        stack[i].min = UINT64_C(1) << MAX(0, 64 - SHIFT - lsb(mask));
        stack[i].step = UINT64_C(1) << stack[i].prefix_bits;
        stack[i].max = UINT64_C(1) << stack[i].bits;
        stack[i].size = init_references(mask, stack[i].refs);
        stack[i].unique_attacks = attack_sets(stack[i].refs, stack[i].size);
        memset(stack[i].age, 0xff, sizeof(uint64_t) * (1 << SHIFT));
        stack[i].stats = 0;

        if (stack[i].min <= stack[i].step) {
            stack[i].min = 0;
        }

        fprintf(stderr, "stack[%d]: mask=0x%lx min=0x%lx step=0x%lx max=0x%lx unique=%d\n", i, mask, stack[i].min, stack[i].step, stack[i].max, stack[i].unique_attacks);

        if (stack[i].last) {
            unique_attacks = stack[i].unique_attacks;
            break;
        }
    }

    assert(mask == max_occupied);
}

static void print_stats() {
    uint64_t total = 0;
    for (int i = 0; i < MAX_SQUARES; i++) {
        fprintf(stderr, "stats[%d]: %ld\n", i, stack[i].stats);
        total += stack[i].stats;
        if (stack[i].last) break;
    }
    fprintf(stderr, "total tests for square %d with shift %d: %ld\n", SQUARE, SHIFT, total);
}

static void print_prefix(uint64_t magic, int depth) {
    int cur_depth = 0;
    for (; cur_depth < MAX_SQUARES; cur_depth++) {
        if (stack[cur_depth].last) break;
    }

    char binary[128] = { 0 };
    int ch = 0;

    for (int i = stack[cur_depth].bits - 1; i >= 0; i--) {
        if (cur_depth >= 0 && stack[cur_depth].bits == i + 1) {
          binary[ch++] = '_';
          cur_depth--;
        }
        binary[ch++] = (magic & (UINT64_C(1) << i)) ? '1' : '0';
    }

    fprintf(stderr, "0b%s (sq %d, shift %d, depth %d, %d bits of 0x%lx, num magics: %ld)\n", binary, SQUARE, SHIFT, depth, stack[depth].bits, magic, num_magics);
}

static void divide_and_conquer(uint64_t prefix, int depth) {
    static uint64_t table[1 << SHIFT];

    stack_t * const frame = stack + depth;

    for (uint64_t magic = prefix | frame->min; magic < frame->max; magic += frame->step) {
        frame->stats++;

        int used_buckets = 0;

        size_t ref = 0;
        for (; ref < frame->size; ref++) {
            uint64_t idx = (magic * frame->refs[ref].occupied) >> (64 - SHIFT);
            if (frame->age[idx] != magic) {
                frame->age[idx] = magic;
                table[idx] = frame->refs[ref].attack;
                used_buckets++;
            } else if (table[idx] != frame->refs[ref].attack) {
                break;
            }
        }

        if (used_buckets + unique_attacks - frame->unique_attacks > (1 << SHIFT)) continue;

        if (ref == frame->size) {
            if (frame->last) {
                num_magics++;
                printf("0x%lx\n", magic);
                if (num_magics < 200) print_prefix(magic, depth);
            } else {
                if (depth <= 0) print_prefix(magic, depth);
                divide_and_conquer(magic, depth + 1);
            }
        }
    }
}

int main() {
    uint64_t max_occupied = square_mask(DELTAS, SQUARE);
    init_stack(max_occupied);
    fprintf(stderr, "searching square=%d shift=%d mask=0x%lx ...\n", SQUARE, SHIFT, max_occupied);
    divide_and_conquer(0, 0);
    print_stats();
    return 0;
}
