#include "helper_functions.h"
#include "spmv.h"

using namespace std;

#define USE_TREE_ADDER 1
#define USE_ROW_SHARE 1

DEFINE_string(bitstream, "", "path to bitstream file, run csim if empty");


void cpu_spmv(const CSRMatrix& A, const vector<float> B, vector<float>& Cin, const float alpha, const float beta, vector<float>& Cout) {  // Initialize result vector C with zeros

    // Perform matrix-vector multiplication
    for (int i = 0; i < A.row_offsets.size() - 1; ++i) {
        for (int j = A.row_offsets[i]; j < A.row_offsets[i + 1]; ++j) {
            int colIndex = A.col_indices[j];
            float value = A.values[j];
            Cout[i] += value * B[colIndex];
        }
        Cout[i] = (alpha*Cout[i]) + (beta*Cin[i]);
    }
}

double computePrecisionLoss(const vector<float>& vectorA, const vector<aligned_vector<float>>& vectorB) 
{
  if (vectorA.size() != vectorB[0].size()*NUM_C_CH) {
      cout << "Error: Vector sizes do not match!" << endl;
      return -1;
  }

  double diffSum = 0.0;
  double refSum = 0.0;
  double maxRelativeError = 0.0;
  int maxRelativeErrorIdx = 0;

  for (size_t i = 0; i < vectorA.size(); ++i) {
    int ch = (i / 16) % NUM_C_CH;
    int addr = (i / (16* NUM_C_CH)) * 16 + (i % 16);
    double diff = fabs(vectorA[i] - vectorB[ch][addr]);
    double ref = min(fabs(vectorA[i]), (float)fabs(vectorB[ch][addr]));
    double relativeError = 0.0;

    if ((vectorA[i] != 0.0 ) && (vectorB[ch][addr] != 0.0)) {
        relativeError = (diff / ref);
        if (relativeError > maxRelativeError) {
          maxRelativeErrorIdx = i;
          maxRelativeError = relativeError;
        }
    }

    diffSum += diff;
    refSum += ref;   
  }

  clog << "Max Relative Error: " << maxRelativeError <<  " CPU: " << vectorA[maxRelativeErrorIdx] << " FPGA: " << vectorB[(maxRelativeErrorIdx / 16) % NUM_C_CH][(maxRelativeErrorIdx/ (16* NUM_C_CH)) * 16 + (maxRelativeErrorIdx % 16)] << "\t i: " << maxRelativeErrorIdx << endl;

  return diffSum/refSum;
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);

  if (argc < 2){
    clog << "Invalid Arguments" << endl;
    return 0; 
  }

  char* filename = argv[1];
  uint16_t rp_time = 1;

  if (argc == 3) 
    rp_time = (uint16_t)atoi(argv[2]);


  clog << "Reading A" << filename << endl;

  vector<float> cscValues;
  vector<int> cscRowIndices;
  vector<int> cscColOffsets;
  int rows, cols, nnz;
  readMatrixCSC(filename, cscValues, cscRowIndices, cscColOffsets, rows, cols, nnz);


  CSRMatrix cpuAmtx;
  convertCSCtoCSR(cscValues, cscRowIndices, cscColOffsets, cpuAmtx.values, cpuAmtx.col_indices, cpuAmtx.row_offsets, rows, cols, nnz);


  int Window = W;
  int Depth = D;

  int numTilesCols = (cols - 1)/Window + 1;
  int numTilesRows = (rows - 1)/Depth + 1;

  cout << "Rows: " << rows << "\t Cols: " << cols << "\t NNZ: " << nnz << endl;
  cout << "Window: " << Window << "\t Depth: " << Depth << endl;
  cout << "Numtiles: " << numTilesRows << ", " << numTilesCols << endl << endl; 
  cout << "Preparing A Mtx... " << endl; 

  auto start_gen = std::chrono::steady_clock::now();
  vector<vector<CSRMatrix>> tiledMatrices = tileCSRMatrix(cpuAmtx, rows, cols, Depth, Window, numTilesRows, numTilesCols);

  vector<aligned_vector<uint64_t>> fpgaAmtx = prepareAmtx(tiledMatrices, numTilesRows, numTilesCols, Depth, Window, rows, cols,  nnz, USE_ROW_SHARE, USE_TREE_ADDER);

  auto end_gen = std::chrono::steady_clock::now();
  double time_gen = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen - start_gen).count();
  time_gen *= 1e-9;
  cout << "Pre-processing Time: " << time_gen*1000 << " msec\n";

  uint32_t numRowsPerPE = (rows - 1) / NUM_PES + 1;
  uint32_t numCols16 = (cols - 1) / 16 + 1;
  uint32_t numTiles = (numTilesRows * numTilesCols);
  
  cout << "Rows Per PE: " << numRowsPerPE << "\t Num Cols 16: " << numCols16 << "\t NNZ: " << nnz << endl;
  cout << "Numtiles: " << numTiles << endl << endl; 

  vector<float>cpuBvect(numCols16 * 16, 0);
  vector<float>cpuCinVect(numRowsPerPE * NUM_PES, 0);
  vector<float>cpuCoutVect(numRowsPerPE * NUM_PES, 0);

  aligned_vector<float>fpgaBvect(numCols16 * 16, 0);
  vector<aligned_vector<float>>fpgaCinVect(NUM_C_CH, aligned_vector<float>(numRowsPerPE * NUM_PES/NUM_C_CH, 0));
  vector<aligned_vector<float>>fpgaCoutVect(NUM_C_CH, aligned_vector<float>(numRowsPerPE * NUM_PES/NUM_C_CH, 0));

  const float alpha = 0.85;
  const float beta = -2.06;

  clog << "Generating C Vec..."  << endl;
  for (int i = 0; i < rows; i++) {
    int ch = (i / 16) % NUM_C_CH;
    int addr = (i / (16* NUM_C_CH)) * 16 + (i % 16); 
    cpuCinVect[i] = -2.0 * (i+1) / (i+2);
    fpgaCinVect[ch][addr] = -2.0 * (i+1) / (i+2);
  }

  cout << endl << "Generating B Vec..."  << endl;
  for (int i = 0; i < (numCols16 * 16); i++) {
    cpuBvect[i] = 1.0 * (i+1) / (i+2);
    fpgaBvect[i] = 1.0 * (i+1) / (i+2);
  }

  cout <<  endl << "Computing CPU SpMV... "  << endl;

  auto start_cpu = std::chrono::steady_clock::now();
  cpu_spmv(cpuAmtx, cpuBvect, cpuCinVect, alpha, beta, cpuCoutVect);
  auto end_cpu = std::chrono::steady_clock::now();
  double time_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count();
  time_cpu *= 1e-9;
  cout << "done (" << time_cpu*1000 << " msec)\n";
  cout <<"CPU GFLOPS: " << 2.0 * (nnz + rows) / 1e+9 / time_cpu << "\n";
  // cout <<  endl << "Computing FPGA SpMV... "  << endl;
  // cpu_spmv_tiled(tiledMatrices, fpgaBvect, fpgaCoutVect, numTilesRows, numTilesCols, Depth, Window);

  cout <<  endl << "Computing FPGA SpMV... "  << endl;
  // fpga_spmv(fpgaAmtx, fpgaBvect, fpgaCinVect, fpgaCoutVect, alpha, beta, numTilesRows, numTilesCols, Depth, Window);
  
  uint32_t run_len = fpgaAmtx[0].size()/ PES_PER_CH;
  // vector<aligned_vector<uint64_t>> DEBUGmtx(NUM_CH, aligned_vector<uint64_t>(PES_PER_CH*run_len));

  cout << "Run Length:" << run_len << endl;

  bool USE_DOUBLE_BUFFER = false;
#ifdef HYBRID_DESIGN
  USE_DOUBLE_BUFFER = (run_len <= (numCols16 * numTilesRows));
#endif
  printf("Double Buffering:%d\n", USE_DOUBLE_BUFFER);

  double time_taken = tapa::invoke(
    SpMV, FLAGS_bitstream, 
    tapa::read_only_mmaps<uint64_t, NUM_CH>(fpgaAmtx).reinterpret<uint64_v>(),
    tapa::read_only_mmap<float>(fpgaBvect).reinterpret<float_v16>(),
    tapa::read_only_mmaps<float, NUM_C_CH>(fpgaCinVect).reinterpret<float_v16>(),
    tapa::write_only_mmaps<float, NUM_C_CH>(fpgaCoutVect).reinterpret<float_v16>(),
    alpha, beta, 
    (uint32_t)numRowsPerPE, (uint32_t)numCols16,
    (uint16_t)numTilesRows, (uint16_t)numTilesCols,
    (uint32_t)(numTiles*rp_time), (uint32_t)run_len, (uint16_t)rp_time, USE_DOUBLE_BUFFER);
  // clog << "kernel time: " << kernel_time_ns * 1e-9 << " s" << endl;

  time_taken *= (1e-9); // total time in second
  time_taken /= rp_time;
    printf("Kernel time:%f\n", time_taken*1000);
  float gflops =
    2.0 * (nnz + rows)
    / 1e+9
    / time_taken
    ;
  printf("GFLOPS:%f\n", gflops);



  cout <<  endl << "Comparing Results... "  << endl;
  double precisionLoss = computePrecisionLoss(cpuCoutVect, fpgaCoutVect);
  cout << "Precision Loss: " << precisionLoss << endl;
  return 0;

}