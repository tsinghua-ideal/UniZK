DEPS=Makefile gen_trace.cpp

default: gen_trace

gen_trace: $(DEPS) gen_trace.cpp
	g++ -O3 -g -o gen_trace gen_trace.cpp

run: gen_trace
	./gen_trace 1 cpu.bin
	./gen_trace 8 gpu.bin

clean:
	rm -f gen_trace *.bin
