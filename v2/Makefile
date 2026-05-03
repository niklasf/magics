CFLAGS = -std=gnu99 -Ofast -flto -fwhole-program -Wall

all: daq

daq: daq.c Makefile
	gcc $(CFLAGS) -D ROOK -D SQUARE=22 -D SHIFT=10 -D REPORT_DEPTH=1 daq.c -o daq
