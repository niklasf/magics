# Copyright (c) 2018 Niklas Fiekas <niklas.fiekas@backscattering.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import chess
import itertools

PIECE_TYPE = chess.ROOK
SQUARE = chess.H3
SHIFT = 11
DELTAS = [-7, 7, -9, 9] if PIECE_TYPE == chess.BISHOP else [-8, 8, -1, 1]

mask = chess._sliding_attacks(SQUARE, 0, DELTAS) & ~chess._edges(SQUARE)

print(r"""
#define SQUARE_NAME "{}"
#define SHIFT {}
#define PERIOD (UINT64_C(1) << {}) // {}
""".format(chess.SQUARE_NAMES[SQUARE], SHIFT, 64 - chess.lsb(mask), chess.SQUARE_NAMES[SQUARE]))

auto_increment = itertools.count(1)
refs = {}
normalized = {}

subset = 0
while True:
    attack = chess._sliding_attacks(SQUARE, subset, DELTAS)
    if attack not in normalized:
        attack_id = next(auto_increment)
        normalized[attack] = attack_id
    else:
        attack_id = normalized[attack]

    refs[subset] = attack_id

    subset = (subset - mask) & mask
    if not subset:
        break

max_attack_id = max(refs.values())
packed = (1 << SHIFT) * SHIFT <= 64 and max_attack_id < (1 << SHIFT)
if packed:
    assert max_attack_id < (1 << SHIFT)
else:
    assert max_attack_id < 256, f"attack_id {max_attack_id} does not fit in uint8_t"

by_attack = {}
for subset, attack_id in refs.items():
    if subset:
        by_attack.setdefault(attack_id, []).append(subset)

# Interleave round-robin, rarest attack_id first, so consecutive checks
# always have different attack_ids and less frequent ids are placed early.
groups = sorted(by_attack.items(), key=lambda x: len(x[1]))

if packed:
    print("""
__device__ bool check_magic(uint64_t magic) {{
    uint64_t table = {zero}; // slot 0 pre-set for subset 0
    int shift;
    uint64_t slot;
""".format(zero=refs[0]))

    for i in range(max(len(subsets) for _, subsets in groups)):
        for attack_id, subsets in groups:
            if i < len(subsets):
                subset = subsets[i]
                print("    shift = (int)((magic * UINT64_C({})) >> {}) * {};".format(subset, 64 - SHIFT, SHIFT))
                print("    slot = (table >> shift) & UINT64_C(0x{:x});".format((1 << SHIFT) - 1))
                print("    if (slot && slot != UINT64_C({})) return false;".format(attack_id))
                print("    table |= UINT64_C({}) << shift;".format(attack_id))
                print()
else:
    print("""
__device__ bool check_magic(uint64_t magic) {{
    uint8_t table[1 << {shift}] = {{ {zero} }}; // slot 0 pre-set for subset 0
    int idx;
""".format(shift=SHIFT, zero=refs[0]))

    for i in range(max(len(subsets) for _, subsets in groups)):
        for attack_id, subsets in groups:
            if i < len(subsets):
                subset = subsets[i]
                print("    idx = (int)((magic * UINT64_C({})) >> {});".format(subset, 64 - SHIFT))
                print("    if (table[idx] && table[idx] != {}) return false;".format(attack_id))
                print("    table[idx] = {};".format(attack_id))
                print()

print("""
    return true;
}
""")
