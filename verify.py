#!/usr/bin/env python
#
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

import sys
import chess

DELTAS = {
    "bishop": [7, -7, 9, -9],
    "rook": [1, -1, 8, -8],
}

FIXED_SHIFT = {
    "bishop": 9,
    "rook": 12
}

def carry_rippler(mask):
    subset = 0
    while True:
        yield subset
        subset = (subset - mask) & mask
        if not subset:
            break

def verify_magic(magic, square, role, shift):
    deltas = DELTAS[role]
    mask = chess._sliding_attacks(square, 0, deltas) & ~chess._edges(square)

    max_idx = 0

    table = [None for _ in range(1 << shift)]

    for occupied in carry_rippler(mask):
        attack = chess._sliding_attacks(square, occupied, deltas)
        idx = ((occupied * magic) & chess.BB_ALL) >> (64 - shift)
        max_idx = max(max_idx, idx)

        if table[idx] is None:
            table[idx] = attack
        elif table[idx] != attack:
            return

    return max_idx

def verify(magic):
    for role in ["bishop", "rook"]:
        for square, name in enumerate(chess.SQUARE_NAMES):
            for shift in range(1, FIXED_SHIFT[role] + 1):
                max_idx = verify_magic(magic, square, role, shift)
                if max_idx is not None:
                    print("{} magic: {}, square: {} ({}) shift: {} max idx: {}".format(
                          role, hex(magic), name, square, shift, max_idx))
                    break

def parse(arg):
    if arg.startswith("0x"):
        return int(arg, 16)
    else:
        return int(arg)

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage: {} <CANDIATE>".format(sys.argv[0]))
        print("Verifies a magic candidate in hex (0xf00) or decimal notation")

    for arg in args:
        magic = parse(arg)
        print("checking {}".format(hex(magic)))
        verify(magic)
