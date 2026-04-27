HIPCC=hipcc

test: test.hip generated.inc
	$(HIPCC) test.hip --offload-arch=gfx1201 -O3 -o test

generated.inc: generate.py
	python generate.py > generated.inc
