CFLAGS = -D BISHOP -D SQUARE=35 -D EASY_SHIFT=9 -std=gnu99 -march=native -Ofast -Wall

all: magics

magics: magics.c magics.gcda
	gcc $(CFLAGS) -fprofile-use=magics.gcda -fno-tracer magics.c -o magics

magics.gcda: magics-instrumented
	rm -f magics.gcda
	./magics-instrumented --profile

magics-instrumented: magics.c
	gcc $(CFLAGS) -fprofile-generate=magics.gcda magics.c -o magics-instrumented
