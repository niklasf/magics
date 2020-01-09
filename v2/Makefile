CFLAGS = -std=gnu99 -march=native -Ofast -flto -fwhole-program -Wall

all: daq

daq: daq.c Makefile
	gcc $(CFLAGS) -D ROOK -D SQUARE=27 -D SHIFT=9 daq.c -o daq
