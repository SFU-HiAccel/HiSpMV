import os
import csv
import subprocess

def list_mtx_files(directory):
    mtx_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".mtx"):
                mtx_files.append(os.path.join(root, filename))
    return mtx_files

def convert_to_csv(data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Writing header row
        headers = ['Filename', 'loadX', 'initY', 'compAx', "updateY", "Total", "runtime (ms)"]
        writer.writerow(headers)

        # Writing data rows
        for filename, values in data.items():
            row_data = [
                filename.split(".")[0],
                values.get('loadX', ''),
                values.get('initY', ''),
                values.get('compAx', ''),
                values.get('updateY', ''),
                values.get('Total', ''),
                values.get('runtime', '')
            ]
            writer.writerow(row_data)

def compile_and_run_cpp(home_dir, build_dir, files, num_ch_A, num_ch_y, hb, ta):
    # Compile C++ code
    bin_path = os.path.join(home_dir, "assets/misc/main") 
    source_path = os.path.join(home_dir, "assets/misc/main.cpp") 
    csv_path = os.path.join(build_dir, "runtime_est.csv")

 
        
    if os.path.exists(bin_path):
        os.remove(bin_path)

    HB = ""
    TA = ""

    command = ["g++", source_path, "-o", bin_path, "-O2", "-fopenmp"]
    if hb:
        command.append("-DHYBRID_DESIGN")
    
    if ta:
        command.append("-DBUILD_TREE_ADDER")

    compile_process = subprocess.run(command, capture_output=True, text=True)
    if compile_process.returncode != 0:
        print(compile_process.stderr)
    
    else:
    
    # Run compiled executable
        stats = {}
        for file in files:
            filename = file.split("/")[-1].split(".")[0]
            print(filename)
            stats[filename] = {}
            run_process = subprocess.run([bin_path, file, str(num_ch_A), str(num_ch_y)], capture_output=True, text=True)
            if run_process.returncode != 0:
                print(run_process.stderr)
            else:
                for line in run_process.stdout.split("\n"):
                    line.strip()
                    print(line)
                    if line.startswith("loadX") or line.startswith("initY") or line.startswith("compAx") or line.startswith("updateY") or line.startswith("Total"):
                        parameter, value = line.split(":")
                        parameter = parameter.strip()
                        value = int(value.strip())
                        stats[filename][parameter] = value
                runtime =  stats[filename]["Total"]/225/1000
                stats[filename]["runtime"] = runtime
        convert_to_csv(stats, csv_path)
        
        return run_process.stdout


def estimate_runtimes(home_dir, build_dir, matrix_dir, num_ch_A, num_ch_y, has_hb, has_ta):
    files = list_mtx_files(os.path.abspath(matrix_dir))
    print("Found following matrices:")
    for file in files:
        print(file)
    print()
    compile_and_run_cpp(os.path.abspath(home_dir), os.path.abspath(build_dir), files, num_ch_A, num_ch_y, has_hb, has_ta)


# def __main__():
# estimate_runtimes("./new_codegen/SpMV", "/localhdd/mba151/", "./matrices", 16, 1 , True, True)