Finding magic factors for bitboard based move generation
========================================================

Tool to find factors for [Magic bitboards](https://www.chessprogramming.org/Magic_Bitboards).

Results
-------

All magics for a given square are equivalent to a magic in the range
`[0, period]` [1].

`c` is the occupancy of the attack mask for the given square (the "easy" shift).

Tables contain magics for fixed shift and shift `c` with proven minimal
table size (attacks mapped to indexes from in `[0, range]`), as well as magics
with reduced shift `c - 1`.

The latter include [results from](https://www.chessprogramming.org/Best_Magics_so_far)
Grant Osborne, Peter Österlund, Volker Annuss, Gerd Isenberg, and Richard Pijl.
New results in **bold**.

### Rooks

| | s | period | c | magic 12 | range 12 | magic c | range c | magic c - 1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | 0 | 2^63 | 12 | | | | | |
| b1 | 1 | 2^62 | 11 | | | | | |
| c1 | 2 | 2^63 | 11 | | | | | |
| d1 | 3 | 2^63 | 11 | | | | | |
| e1 | 4 | 2^63 | 11 | | | | | |
| f1 | 5 | 2^63 | 11 | | | | | |
| g1 | 6 | 2^63 | 11 | | | | | |
| h1 | 7 | 2^63 | 12 | | | | | |
| a2 | 8 | 2^55 | 11 | | | | | |
| b2 | 9 | 2^54 | 10 | | | | | |
| c2 | 10 | 2^55 | 10 | | | | | |
| d2 | 11 | 2^55 | 10 | | | | | |
| e2 | 12 | 2^55 | 10 | | | | | |
| f2 | 13 | 2^55 | 10 | | | | | |
| g2 | 14 | 2^55 | 10 | | | | | |
| h2 | 15 | 2^55 | 11 | | | | | |
| a3 | 16 | 2^56 | 11 | | | | | **disproved** |
| b3 | 17 | 2^55 | 10 | | | | | **disproved** |
| c3 | 18 | 2^54 | 10 | | | | | **disproved** |
| d3 | 19 | 2^53 | 10 | | | | | **disproved** |
| e3 | 20 | 2^52 | 10 | | | | | **disproved** |
| f3 | 21 | 2^51 | 10 | | | | | **disproved** |
| g3 | 22 | 2^50 | 10 | | | | | **disproved** |
| h3 | 23 | 2^49 | 11 | | | | | **disproved** |
| a4 | 24 | 2^56 | 11 | | | | | **disproved** |
| b4 | 25 | 2^55 | 10 | | | | | **disproved** |
| c4 | 26 | 2^54 | 10 | | | | | **disproved** |
| d4 | 27 | 2^53 | 10 | | | | | **disproved** |
| e4 | 28 | 2^52 | 10 | | | | | **disproved** |
| f4 | 29 | 2^51 | 10 | | | | | **disproved** |
| g4 | 30 | 2^50 | 10 | | | | | **disproved** |
| h4 | 31 | 2^49 | 11 | | | **0x137894006eced** | 2044 | **disproved** |
| a5 | 32 | 2^56 | 11 | | | | | **disproved** |
| b5 | 33 | 2^55 | 10 | | | | | **disproved** |
| c5 | 34 | 2^54 | 10 | | | | | **disproved** |
| d5 | 35 | 2^53 | 10 | | | | | **disproved** |
| e5 | 36 | 2^52 | 10 | | | | | **disproved** |
| f5 | 37 | 2^51 | 10 | | | | | **disproved** |
| g5 | 38 | 2^50 | 10 | | | | | **disproved** |
| h5 | 39 | 2^49 | 11 | | | **0x4242136fff00** | 2045 | **disproved** |
| a6 | 40 | 2^56 | 11 | | | | | **disproved** |
| b6 | 41 | 2^55 | 10 | | | | | **disproved** |
| c6 | 42 | 2^54 | 10 | | | | | **disproved** |
| d6 | 43 | 2^53 | 10 | | | | | **disproved** |
| e6 | 44 | 2^52 | 10 | | | | | **disproved** |
| f6 | 45 | 2^51 | 10 | | | | | **disproved** |
| g6 | 46 | 2^50 | 10 | | | | | **disproved** |
| h6 | 47 | 2^49 | 11 | | | | | **disproved** |
| a7 | 48 | 2^56 | 11 | | | | | 0x48fffe99fecfaa00 |
| b7 | 49 | 2^55 | 10 | | | | | 0x48fffe99fecfaa00 |
| c7 | 50 | 2^54 | 10 | | | | | 0x497fffadff9c2e00 |
| d7 | 51 | 2^53 | 10 | | | | | 0x613fffddffce9200 |
| e7 | 52 | 2^52 | 10 | | | | | 0xffffffe9ffe7ce00 |
| f7 | 53 | 2^51 | 10 | | | | | 0xfffffff5fff3e600 |
| g7 | 54 | 2^50 | 10 | | | | | **0x3ff95e5e6a4c0** |
| h7 | 55 | 2^49 | 11 | | | | | 0x510ffff5f63c96a0 |
| a8 | 56 | 2^56 | 12 | | | | | 0xebffffb9ff9fc526 |
| b8 | 57 | 2^55 | 11 | | | | | 0x61fffeddfeedaeae |
| c8 | 58 | 2^54 | 11 | | | | | 0x53bfffedffdeb1a2 |
| d8 | 59 | 2^53 | 11 | | | | | 0x127fffb9ffdfb5f6 |
| e8 | 60 | 2^52 | 11 | | | | | 0x411fffddffdbf4d6 |
| f8 | 61 | 2^51 | 11 | | | | | **disproved** |
| g8 | 62 | 2^50 | 10 | | | | | 0x3ffef27eebe74 |
| h8 | 63 | 2^49 | 12 | | | | | 0x7645fffecbfea79e |

### Bishops

| | s | period | c | magic 9 | range 9 | magic c | range c | magic c - 1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | 0 | 2^55 | 6 | | | | | 0xffedf9fd7cfcffff |
| b1 | 1 | 2^54 | 5 | | | | | 0xfc0962854a77f576 |
| c1 | 2 | 2^55 | 5 | | | | | |
| d1 | 3 | 2^54 | 5 | | | | | |
| e1 | 4 | 2^53 | 5 | | | | | **disproved** |
| f1 | 5 | 2^52 | 5 | | | | | **disproved** |
| g1 | 6 | 2^51 | 5 | | | | | 0xfc0a66c64a7ef576 |
| h1 | 7 | 2^50 | 6 | | | | | 0x7ffdfdfcbd79ffff |
| a2 | 8 | 2^47 | 5 | | | | | 0xfc0846a64a34fff6 |
| b2 | 9 | 2^46 | 5 | | | | | 0xfc087a874a3cf7f6 |
| c2 | 10 | 2^47 | 5 | | | | | **disproved** |
| d2 | 11 | 2^46 | 5 | | | | | **disproved** |
| e2 | 12 | 2^45 | 5 | **0x4402000000** | 62 | **0x8f24760b248** | 30 | **disproved** |
| f2 | 13 | 2^44 | 5 | **0x21c1007bff** | 59 | **0xa21427af62** | 28 | **disproved** |
| g2 | 14 | 2^43 | 5 | **0x1041040040** | 61 | **0x52e9060be9** | 22 | 0x820a13ffe0 |
| h2 | 15 | 2^42 | 5 | **0x820820020** | 61 | **0x29748305f5** | 22 | **0x410509fff0** |
| a3 | 16 | 2^55 | 5 | | | | | 0x73c01af56cf4cffb |
| b3 | 17 | 2^54 | 5 | | | | | 0x41a01cfad64aaffc |
| c3 | 18 | 2^55 | 7 | | | | | **disproved** |
| d3 | 19 | 2^54 | 7 | | | | | **disproved** |
| e3 | 20 | 2^53 | 7 | | | | | **disproved** |
| f3 | 21 | 2^52 | 7 | | | | | **disproved** |
| g3 | 22 | 2^51 | 5 | | | | | 0x7c0c028f5b34ff76 |
| h3 | 23 | 2^50 | 5 | | | | | 0xfc0a028e5ab4df76 |
| a4 | 24 | 2^54 | 5 | | | | | **disproved** |
| b4 | 25 | 2^53 | 5 | | | | | **disproved** |
| c4 | 26 | 2^52 | 7 | | | | | **disproved** |
| d4 | 27 | 2^55 | 9 | | | | | **disproved** |
| e4 | 28 | 2^54 | 9 | | | | | **disproved** |
| f4 | 29 | 2^53 | 7 | | | | | **disproved** |
| g4 | 30 | 2^52 | 5 | | | | | **disproved** |
| h4 | 31 | 2^51 | 5 | | | | | **disproved** |
| a5 | 32 | 2^53 | 5 | | | | | **disproved** |
| b5 | 33 | 2^52 | 5 | | | | | **disproved** |
| c5 | 34 | 2^51 | 7 | | | | | **disproved** |
| d5 | 35 | 2^50 | 9 | **0x20080080080** | 511 | **0x20080080080** | 511 | **disproved** |
| e5 | 36 | 2^55 | 9 | | | | | **disproved** |
| f5 | 37 | 2^54 | 7 | | | | | **disproved** |
| g5 | 38 | 2^53 | 5 | | | | | **disproved** |
| h5 | 39 | 2^52 | 5 | | | | | **disproved** |
| a6 | 40 | 2^52 | 5 | | | | | 0xdcefd9b54bfcc09f |
| b6 | 41 | 2^51 | 5 | | | | | 0xf95ffa765afd602b |
| c6 | 42 | 2^50 | 7 | | | | | **disproved** |
| d6 | 43 | 2^42 | 7 | **0x8840200040** | 132 | **0xa44000800** | 127 | **disproved** |
| e6 | 44 | 2^47 | 7 | | | | | **disproved** |
| f6 | 45 | 2^55 | 7 | | | | | **disproved** |
| g6 | 46 | 2^54 | 5 | | | | | 0x43ff9a5cf4ca0c01 |
| h6 | 47 | 2^53 | 5 | | | | | 0x4bffcd8e7c587601 |
| a7 | 48 | 2^51 | 5 | | | | | 0xfc0ff2865334f576 |
| b7 | 49 | 2^50 | 5 | | | | | 0xfc0bf6ce5924f576 |
| c7 | 50 | 2^42 | 5 | **0x820806000** | 35 | **0x50c34179e6** | 29 | **disproved** |
| d7 | 51 | 2^34 | 5 | **0x8403000** | 60 | **0x20880000** | 31 | **disproved** |
| e7 | 52 | 2^39 | 5 | **0x100202000** | 31 | **0x4f68bcb86d** | 29 | **disproved** |
| f7 | 53 | 2^47 | 5 | | | | | **disproved** |
| g7 | 54 | 2^55 | 5 | | | | | 0xc3ffb7dc36ca8c89 |
| h7 | 55 | 2^54 | 5 | | | | | 0xc3ff8a54f4ca2c89 |
| a8 | 56 | 2^50 | 6 | | | | | 0xfffffcfcfd79edff |
| b8 | 57 | 2^42 | 5 | **0x820820020** | 61 | **0x58c328c2ee** | 22 | 0x1ec04eae595 |
| c8 | 58 | 2^34 | 5 | **0x8208060** | 35 | **0x50c3417a** | 29 | **disproved** |
| d8 | 59 | 2^26 | 5 | **0x84030** | 60 | **0x208800** | 31 | **disproved** |
| e8 | 60 | 2^31 | 5 | **0x1002020** | 31 | **0x4f68bcb9** | 29 | **disproved** |
| f8 | 61 | 2^39 | 5 | **0x40408020** | 31 | **0x74486419f** | 26 | **disproved** |
| g8 | 62 | 2^47 | 5 | | | | | 0xfc087e8e4bb2f736 |
| h8 | 63 | 2^55 | 6 | | | | | 0x43ff9e4ef4ca2c89 |

Tools
-----

### `verify.py <CANDIDATE>`

Verifies a magic candidate.

### `v2/daq`

Select `BISHOP` or `ROOK`, `SQUARE` and `SHIFT` in `v2/Makefile`,
`make` and run `./daq`. Counts all magics with the specified settings.

### `gpu/test`

Select square, piece type and shift in `generate.py`.
`make` and run `./test` to find magic factors with the specified
shift or disprove their existence. Works best for bishop squares (with small
shifts and small table sizes).

References
----------

1. [Disproving the existence of some magics](http://www.talkchess.com/forum/viewtopic.php?t=65187). 2017.
2. [No bishop magics with fixed shift 8](http://www.talkchess.com/forum/viewtopic.php?t=67051). 2018.
