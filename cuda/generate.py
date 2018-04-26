import chess
import itertools

deltas = [-7, 7, -9, 9]
square = chess.D7
shift = 4

refs = {}
normalized = {}

auto_increment = itertools.count(1)
attack_id = 0

mask = chess._sliding_attacks(square, 0, deltas) & ~chess._edges(square)
subset = 0
while True:
    attack = chess._sliding_attacks(square, subset, deltas)
    if attack not in normalized:
        attack_id = next(auto_increment)
        normalized[attack] = attack_id
    else:
        attack_id = normalized[attack]

    refs[subset] = attack_id

    subset = (subset - mask) & mask
    if not subset:
        break

print(r"""
__device__ bool check_magic(uint64_t magic) {
    char table[1 << SHIFT] = { 0 };
    int idx;
""".replace("SHIFT", str(shift)))

for subset, attack_id in refs.items():
    print("    idx = (magic * UINT64_C({})) >> (64 - {});".format(subset, shift))
    print("    if (table[idx] && table[idx] != {}) return false;".format(attack_id))
    print("    table[idx] = {};".format(attack_id))

print("""
    return true;
}
""")
