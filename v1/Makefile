CFLAGS = -D BISHOP -D SQUARE=5 -D EASY_SHIFT=5 -std=gnu99 -march=native -Ofast -flto -fwhole-program -Wall

all: magics

magics: magics.c
	rm -f magics.gcda
	gcc $(CFLAGS) -fprofile-generate magics.c -o magics
	./magics --profile
	gcc $(CFLAGS) -fprofile-use -fno-tracer magics.c -o magics
