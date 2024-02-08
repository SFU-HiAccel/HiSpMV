CC = g++
CFLAGS = -O2 -fopenmp 
LIBRARIES = -ltapa -lfrt -lglog -lgflags -lOpenCL -lpthread -I"$(XILINX_HLS)/include" -std=c++17
DEFINES = -DTAPA_BUFFER_SUPPORT
platform = xilinx_u280_xdma_201920_3

spmv: src/
	$(CC) -o spmv $(CFLAGS) src/* $(DEFINES) $(LIBRARIES)

clean:
	rm -rf spmv

