# HiSpMV
Sparse-matrix vector Multiplication is one of the fundamental kernels used in various applications like Scientific Computing, Machine Learning, Graph Analytics, and Circuit Simulation. HiSpMV is an open-source SpMV Accelerator for FPGA built using Vitis HLS and [PASTA](https://github.com/SFU-HiAccel/pasta) This repo consists of a **Code Generator** and two main designs **HiSpMV-16** and **HiSpMV-20** with host code and bitstream for *Alveo U280 board*. 

## Usage
The following sections will explain in detail how to build/use the HiSpMV designs and Code Generator.
### Requirements:
The following needs to be installed 
1. Vitis 2021.2 (or newer version)
2. TAPA + Autobridge from PASTA (see [here](https://github.com/SFU-HiAccel/pasta/blob/main/tapa/README.md#installation-process) for installation details)
3. [Xilinx Runtime](https://github.com/Xilinx/XRT)
4.  Alveo U280 HBM FPGA
   
### Setup Environment:
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

### Run HiSpMV design on benchmark matrices:
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

### Use code generator to build your own hardware


## Publications
The HiSpMV work is published in FPGA 2024.
> Manoj B. Rajashekar, Xingyu Tian, and Zhenman Fang. 2024. HiSpMV: Hybrid Row Distribution and Vector Buffering for Imbalanced SpMV Acceleration on FPGAs. In Proceedings of the 2024 ACM/SIGDA International Symposium on Field Programmable Gate Arrays (FPGA ’24), March 3–5, 2024, Monterey, CA, USA. ACM, New York, NY, USA, 12 pages, https://doi.org/10.1145/3626202.3637557

The PASTA work has been published at FCCM 2023.
> M. Khatti, X. Tian, Y. Chi, L. Guo, J. Cong and Z. Fang, "PASTA: Programming and Automation Support for Scalable Task-Parallel HLS Programs on Modern Multi-Die FPGAs," 2023 IEEE 31st Annual International Symposium on Field-Programmable Custom Computing Machines (FCCM), Marina Del Rey, CA, USA, 2023, pp. 12-22, doi: 10.1109/FCCM57271.2023.00011.
