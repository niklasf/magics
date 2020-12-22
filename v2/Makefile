CFLAGS = -std=gnu99 -march=native -Ofast -flto -fwhole-program -Wall

all: daq

daq: daq.c Makefile
	gcc $(CFLAGS) -D ROOK -D SQUARE=7 -D SHIFT=11 daq.c -o daq
