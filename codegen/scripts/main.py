from spmvcodegen import SpMVCodeGen
from resource import compute_optimum_num_ch, print_resource
from runtime_estimate import estimate_runtimes

import argparse

def main():
    parser = argparse.ArgumentParser(description="Script to Generate TAPA code for SpMV")
    
    parser.add_argument("home_directory", help="Path to home directory, that is the CodeGen Folder")
    parser.add_argument("build_directory", help="Path to build directory, can be any location, new or existing [WARNING: anything inside this directory will be erased]")
    parser.add_argument("--num-ch-A", type=int, help="Number of HBM channels to read sparse matrix A")
    parser.add_argument("--num-ch-y", type=int, help="Number of HBM channels to read dense vector y")
    parser.add_argument("--config", help="Path to configuration file containing available resources on the hardware, must specify bram, uram, dsp, lut, ff, and hbm")
    parser.add_argument("--no-tree-adder", action="store_false", help="Build hardware without reduction tree adder")
    parser.add_argument("--hybrid-buffer", action="store_true", help="Build hardware with hybrid capability")
    parser.add_argument("--matrices", help="Directory containing matrices in .mtx format")
    
    args = parser.parse_args()
    
    home_dir = args.home_directory
    build_dir = args.build_directory
    mat_dir = args.matrices

    num_ch_A = args.num_ch_A
    num_ch_C = args.num_ch_y
    build_with_tree_adder = args.no_tree_adder
    build_with_hybrid_buffer = args.hybrid_buffer
    config_file = args.config

    print(f"Home Directory: {home_dir}")
    print(f"Build Directory: {build_dir}")
    if config_file is not None:
        print(f"Config File: {config_file}")
    print(f"Build with Tree Adder: {build_with_tree_adder}")
    print(f"Build with Hybrid Buffer: {build_with_hybrid_buffer}")
    
    if num_ch_C is None:
        num_ch_C = 2
    
    print(f"Num Channels for Dense Vector y: {num_ch_C}")

    print()

    if num_ch_A is None:
        
        if config_file is None:
            print("Neither the Number of Channels nor the Config File is provided, at least one is required")
            return
        
        print("Computing Optimum Number of Channels, based on the configuration file...")
        num_ch_A = compute_optimum_num_ch(config_file, build_with_tree_adder, build_with_hybrid_buffer, num_ch_C)

    else:
        if config_file is not None:
            print("Number of Channels Specified, skipping computing optimum num_ch...")
        num_ch_A = int((num_ch_A // (2*num_ch_C)) * (2*num_ch_C))
        print_resource(config_file, build_with_tree_adder, build_with_hybrid_buffer, num_ch_A, num_ch_C)
    
    print()
    total_processors = num_ch_A * 8

    print(f"Number of Channels: {num_ch_A}")
    print(f"Total PEs: {total_processors}")

    myGen = SpMVCodeGen(total_processors, num_ch_C, build_with_tree_adder, build_with_hybrid_buffer, build_dir, home_dir)
    myGen.generateAll()

    if mat_dir is not None:
        print(f"Matrix Directory Specified : {mat_dir}\n")
        print(f"Estimating Clock Cycles\n")
        estimate_runtimes(home_dir, build_dir, mat_dir, num_ch_A, num_ch_C, build_with_hybrid_buffer, build_with_tree_adder)

if __name__ == "__main__":
    main()