CFLAGS = -std=gnu99 -Ofast -flto -fwhole-program -Wall

all: daq

daq: daq.c Makefile
	gcc $(CFLAGS) -D ROOK -D SQUARE=39 -D SHIFT=11 -D REPORT_DEPTH=0 daq.c -o daq
