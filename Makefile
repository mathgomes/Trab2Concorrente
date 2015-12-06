##Makefile Smooth Paralelo CUDA##
all: Main
Main: main.cu
	nvcc `pkg-config --cflags opencv ` main.cu `pkg-config --libs opencv` -o Cuda

clean:
	rm -rf *.o

mrproper: clean
	rm -rf Cuda
