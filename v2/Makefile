CFLAGS = -std=gnu99 -march=native -Ofast -flto -fwhole-program -Wall

all: daq

daq: daq.c Makefile
	gcc $(CFLAGS) -D ROOK -D SQUARE=24 -D SHIFT=10 daq.c -o daq
