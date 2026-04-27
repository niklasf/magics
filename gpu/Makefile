HIPCC=hipcc
HIPCC_FLAGS=--offload-arch=gfx1201 -O3

test: test.hip generated.inc
	$(HIPCC) $< -o $@

test.s: test.hip generated.inc
	$(HIPCC) $< $(HIPCC_FLAGS) -S -o $@

generated.inc: generate.py
	python $< > $@
