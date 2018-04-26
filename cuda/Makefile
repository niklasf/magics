NVCC=nvcc

test: test.cu generated.inc
	$(NVCC) test.cu -use_fast_math -o test

generated.inc: generate.py
	python generate.py > generated.inc
