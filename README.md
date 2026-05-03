Finding magic factors for bitboard based move generation
========================================================

Tool to find factors for [Magic bitboards](https://www.chessprogramming.org/Magic_Bitboards).

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
