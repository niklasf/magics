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

### Rooks

| | s | period | c | magic 12 | range 12 | magic c | range c | magic c - 1 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| a1 | 0 | 2^63 | 12 | | | | |
| b1 | 1 | 2^62 | 11 | | | | |
| c1 | 2 | 2^63 | 11 | | | | |
| d1 | 3 | 2^63 | 11 | | | | |
| e1 | 4 | 2^63 | 11 | | | | |
| f1 | 5 | 2^63 | 11 | | | | |
| g1 | 6 | 2^63 | 11 | | | | |
| h1 | 7 | 2^63 | 12 | | | | |
| h3 | 23 | 2^49 | 11 | | | | | **disproved** |

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
