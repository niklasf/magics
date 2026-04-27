CFLAGS = -std=gnu99 -march=native -Ofast -flto -fwhole-program -Wall

all: daq

daq: daq.c Makefile
	gcc $(CFLAGS) -D BISHOP -D SQUARE=2 -D SHIFT=4 -D REPORT_DEPTH=0 daq.c -o daq
