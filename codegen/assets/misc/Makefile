CC = g++
CFLAGS = -O2 -fopenmp 
LIBRARIES = -ltapa -lfrt -lglog -lgflags -lOpenCL -lpthread -I"$(XILINX_HLS)/include" -std=c++17
DEFINES = -DTAPA_BUFFER_SUPPORT
platform = xilinx_u280_xdma_201920_3

spmv: src/
	$(CC) -o spmv $(CFLAGS) src/* $(DEFINES) $(LIBRARIES)

tapa:
	tapac -o spmv.$(platform).hw.xo src/spmv.cpp \
  --connectivity link_config.ini \
  --platform $(platform) \
  --top SpMV \
  --read-only-args "A*" \
  --read-only-args "b" \
  --read-only-args "c_in*" \
  --write-only-args "c_out*" \
  --work-dir spmv.$(platform).hw.xo.tapa \
  --enable-hbm-binding-adjustment \
  --enable-floorplan \
  --enable-buffer-support \
  --enable-synth-util \
  --max-parallel-synth-jobs 16 \
  --floorplan-output constraint.tcl \
  --clock-period 4.25

hw:
	sh spmv.$(platform).hw_generate_bitstream.sh

clean:
	rm -rf spmv

