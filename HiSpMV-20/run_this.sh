platform=xilinx_u280_xdma_201920_3
for file in ../matrices/*/*.mtx
do
#   ./main ./"$file" &> "$file".out
    filename=$(basename -- "$file")
    filename="${filename%.*}"

  # Generate a unique log file name based on the current directory
    log_file="${filename}.log"
    # echo "./main $file |& tee $log_file"
    ./spmv --bitstream=SpMV_$platform.xclbin $file 10000 |& tee $log_file

done