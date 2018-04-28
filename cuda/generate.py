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

DELTAS = [-7, 7, -9, 9]
SQUARE = chess.B8
SHIFT = 4

mask = chess._sliding_attacks(SQUARE, 0, DELTAS) & ~chess._edges(SQUARE)

print(r"""
#define PERIOD (UINT64_C(1) << {}) // {}
""".format(64 - chess.lsb(mask), chess.SQUARE_NAMES[SQUARE]))

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

print(r"""
__device__ bool check_magic(uint64_t magic) {
    char table[1 << SHIFT] = { 0 };
    int idx;
""".replace("SHIFT", str(SHIFT)))

for subset, attack_id in refs.items():
    print("    idx = (magic * UINT64_C({})) >> (64 - {});".format(subset, SHIFT))
    print("    if (table[idx] && table[idx] != {}) return false;".format(attack_id))
    print("    table[idx] = {};".format(attack_id))

print("""
    return true;
}
""")
