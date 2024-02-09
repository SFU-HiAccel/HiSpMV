# HiSpMV
Sparse-matrix vector Multiplication is one of the fundamental kernels used in various applications like Scientific Computing, Machine Learning, Graph Analytics, and Circuit Simulation. HiSpMV is an open-source SpMV Accelerator for FPGA built using Vitis HLS and [PASTA](https://github.com/SFU-HiAccel/pasta) This repo consists of a **Code Generator** and two main designs **HiSpMV-16** and **HiSpMV-20** with host code and bitstream for *Alveo U280 board*. 

## Requirements:
The following needs to be installed 
1. Vitis 2021.2 (or newer version)
2. TAPA + Autobridge from PASTA (see [here](https://github.com/SFU-HiAccel/pasta/blob/main/tapa/README.md#installation-process) for installation details)
3. [Xilinx Runtime](https://github.com/Xilinx/XRT)
4.  Alveo U280 HBM FPGA
   
## Set Environment:
Source Vitis and Xilinx Runtime, replace VITIS_PATH and XRT_PATH with the actual path to your Vitis and Xilinx Runtime installation
```
source VITIS_PATH/2021.2/settings64.sh
source XRT_PATH/xrt/setup.sh
```

Add TAPA library to the PATH if installed in non-root, replace TAPA_PATH with the actual path to where you installed TAPA
```
export PATH="TAPA_PATH/bin":$PATH
export CPATH="TAPA_PATH/include":$CPATH
export LD_LIBRARY_PATH="TAPA_PATH/lib":$LD_LIBRARY_PATH
```

*NOTE:* Install and set up Gurobi (recommended if you are building hardware). See [here](https://tapa.readthedocs.io/en/release/installation.html#install-gurobi-recommended) for details.

## Run HiSpMV design on benchmark matrices:
1. Download the benchmark matrices used in the paper by running the python script "get_tb_matrices.py"
```
python3 get_tb_matrices.py
```
2. Switch to HiSpMV-16 or HiSpMV-20 directory
```
cd HiSpMV-16
```
or
```
cd HiSpMV-20
```

3. Compile and run the design for all the matrices
```
make spmv
bash run_this.sh
```

## Use the code generator to build your hardware

### Generate Source Code
This script generates TAPA code for HiSpMV design.

```
python3 codegen/scripts/main.py <home_directory> <build_directory> [--num-ch-A <value>] [--num-ch-y <value>] [--config <path_to_config_file>] [--no-tree-adder] [--hybrid-buffer] [--matrices <path_to_matrices_directory>]
```


#### Arguments

- `home_directory`: Path to the home directory, which is the CodeGen Folder.
- `build_directory`: The path to where the generated source code should be placed, can be any location, new or existing. **WARNING:** Anything inside this directory will be erased.
- `--num-ch-A <value>`: Number of HBM channels to read sparse matrix A (optional).
- `--num-ch-y <value>`: Number of HBM channels to read dense vector y (optional).
- `--config <path_to_config_file>`: The path to the configuration file containing available resources on the hardware, must specify bram, uram, dsp, lut, ff, and hbm (optional).
- `--no-tree-adder`: Build hardware without reduction tree adder represented as Adder Chain Group in the paper(optional).
- `--hybrid-buffer`: Build hardware with hybrid buffering capability (optional).
- `--matrices <path_to_matrices_directory>`: Directory containing matrices in .mtx format (optional).


#### Note
Ensure to provide either `--num-ch-A <value>` or `--config <path_to_config_file>`, if you provide only a `--config` file, the code generator will automatically use the best value for `num-ch-A` based on the resources. If you specify only `--num-ch-A` or both, the code generator will use the value specified.

#### Example
In this example, HiSpMV design with 2 channels for streaming in sparse matrix A, 1 channel for output dense vector y, with hybrid buffer and without tree adder is specified.
```
python3 codegen/scripts/main.py codegen/ HiSpMV-2 --num-ch-A 2 --num-ch-y 1 --config codegen/u280_config.json --hybrid-buffer --no-tree-adder
```
The output will look something like this
```
Home Directory: codegen/
Build Directory: HiSpMV-2
Config File: codegen/u280_config.json
Build with Tree Adder: False
Build with Hybrid Buffer: True
Num Channels for Dense Vector y: 1

Number of Channels Specified, skipping computing optimum num_ch...

Resource Estimation [2 Channels]
  bram: 329 [16.32%]
  DSP: 263 [2.91%]
  uram: 48 [5.0%]
  lut: 177641 [13.63%]
  ff: 222828 [8.55%]
  hbm: 5 [15.62%]

Number of Channels: 2
Total PEs: 16
```

### Build Hardware
Use the following commands to build the hardware from the generated source code

1. Change the current work directory to the path you specified as the build directory, in the previous example this was `HiSpMV-2`
```
cd HiSpMV-2
```
3. Compile host code and synthesize kernel using TAPA
```
make spmv
make tapa
```
4. Generate Bitstream
```
make hw
```
5. Finally test the built hardware on benchmark matrices
```
bash run_this.sh
```
 
## Publications
The HiSpMV work is published in FPGA 2024.
> Manoj B. Rajashekar, Xingyu Tian, and Zhenman Fang. 2024. HiSpMV: Hybrid Row Distribution and Vector Buffering for Imbalanced SpMV Acceleration on FPGAs. In Proceedings of the 2024 ACM/SIGDA International Symposium on Field Programmable Gate Arrays (FPGA ’24), March 3–5, 2024, Monterey, CA, USA. ACM, New York, NY, USA, 12 pages, https://doi.org/10.1145/3626202.3637557

The PASTA work has been published at FCCM 2023.
> M. Khatti, X. Tian, Y. Chi, L. Guo, J. Cong and Z. Fang, "PASTA: Programming and Automation Support for Scalable Task-Parallel HLS Programs on Modern Multi-Die FPGAs," 2023 IEEE 31st Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM), Marina Del Rey, CA, USA, 2023, pp. 12-22, doi: 10.1109/FCCM57271.2023.00011.
