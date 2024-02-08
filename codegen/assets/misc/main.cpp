#include "../host/mmio.h"
#include <iostream>
#include <vector>
#include <algorithm>

#define II_DIST 5
#define PADDING 1
#define FIFO_DEPTH 2
#define FIFO_LARGE_DEPTH 8
#define PES_PER_CH 8

int NUM_C_CH = 0;
int NUM_CH = 0;
int NUM_PES = 0;

struct CSRMatrix {
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;
};

struct rcv{
    int r;
    int c;
    float v;
};

std::vector<std::vector<std::vector<int>>> balanceWorkload(std::vector<std::vector<CSRMatrix>> tiledMatrices, 
    const int numTilesRows, const int numTilesCols, const int Depth, const int SHARED_ROW_LIMIT, const int nnz,
    std::vector<std::vector<int>>& numSharedRows, std::vector<std::vector<int>>& maxPEload1_p, float& original_imb, float& improved_imb, int& totalPEload)
{

    std::vector<std::vector<int>> maxPEload0_p(numTilesRows, std::vector<int>(numTilesCols, 0));
    std::vector<std::vector<std::vector<int>>> sharedRows(numTilesRows, std::vector<std::vector<int>>(numTilesCols, std::vector<int>(SHARED_ROW_LIMIT, 0)));

    int maxPEload0 = 0;
    int maxPEload1 = 0;

    #pragma omp parallel for collapse(2) 
    for (int i = 0; i < numTilesRows; i++) {
        for (int j = 0; j < numTilesCols; j++) {
            
            std::vector<std::vector<int>>rowCounts(Depth, std::vector<int>(2, 0));
            std::vector<std::vector<int>>peWorkloads(NUM_PES, std::vector<int>(2, 0));

            #ifdef DEBUGGING
                std::cout << "i= "<< i << "\tj= " << j << std::endl;
            #endif

            for(int ii = 0; ii < Depth; ii++) {
                rowCounts[ii][0] = ii;
                rowCounts[ii][1] = tiledMatrices[i][j].row_offsets[ii+1] - tiledMatrices[i][j].row_offsets[ii];
                peWorkloads[ii % NUM_PES][0] = ii % NUM_PES;
                peWorkloads[ii % NUM_PES][1] += rowCounts[ii][1];

            }      

            sort(rowCounts.begin(), 
                rowCounts.end(),
                [] (const std::vector<int> &a, const std::vector<int> &b)
                {
                    return a[1] > b[1];
                });

            int maxVal = 0;
            for (int p = 0; p < NUM_PES; p++)
                if ( peWorkloads[p][1] > maxVal) maxVal = peWorkloads[p][1];
                    

            int total = tiledMatrices[i][j].row_offsets[Depth];
            int scheduled = (maxVal + PADDING) * NUM_PES;

            // #pragma omp critical
            maxPEload0_p[i][j] = maxVal + PADDING;
            maxPEload0 += maxPEload0_p[i][j];

            float imb = (float)(scheduled-total)/total;
            // std::cout << "Max Load 0 = " << peWorkloads[0][1] << std::endl;
            // std::cout << "Imbalance Ratio = " << imb << "\tMax Val: " << maxVal << std::endl;
            int extraCycles = 0;
            std::vector<int> removedRows;

            for(int ii = 0; ii < SHARED_ROW_LIMIT ; ii++) {
                int newTotal = total - rowCounts[ii][1];
                std::vector<std::vector<int>> newPeWorkloads = peWorkloads;
                // 
                if (maxVal < 2)
                    break;

                int newMaxVal = 0;

                for (int p = 0; p < NUM_PES; p++)
                {
                    if (newPeWorkloads[p][0] == (rowCounts[ii][0] % NUM_PES))
                        newPeWorkloads[p][1] -= rowCounts[ii][1];

                    if ( newPeWorkloads[p][1] > newMaxVal) 
                        newMaxVal = newPeWorkloads[p][1];
                    
                }
            
                scheduled = NUM_PES * newMaxVal;
                float new_imb = (float)(scheduled-newTotal)/newTotal;

                //if imbalance ratio decreases only then remove the row
                if (((imb - new_imb) > 0) || (ii < 2)) {
                    // printf("Tile [%d][%d] Removed Row: %d Row Count: %d\n", i, j, i*Depth + rowCounts[ii][0], rowCounts[ii][1]);
                    total = newTotal;
                    peWorkloads = newPeWorkloads;
                    maxVal = newMaxVal;
                    extraCycles += ((rowCounts[ii][1] - 1)/NUM_PES) + 1;
                    removedRows.push_back(rowCounts[ii][0]);
                }

                // std::cout << "Imbalance Ratio = " << new_imb << "\tMax Val: " << maxVal << std::endl; 

                //if the change in imbalance is negligible then exit loop
                if ((std::abs(imb - new_imb) <= 0.01) && (new_imb < 2))
                    break;
                
                imb = new_imb;
            }

            std::sort(removedRows.begin(), removedRows.end());
            
            for (int k = 0; k < removedRows.size(); k++ )
                sharedRows[i][j][k] = removedRows[k];
            numSharedRows[i][j] = removedRows.size();
            
            int load_size = maxVal + extraCycles + PADDING;

            #ifdef DEBUGGING
                printf("Max PE Load: %d\t Extra Cycles: %d\t Estimated Load Size: %d\n", maxVal, extraCycles, load_size);
                // std::cout << "Max PE Load:" << maxVal << "\t Extra Cycles: " << extraCycles << "\t Estimated Load Size: " << load_size << std::endl;
            #endif
            maxPEload1_p[i][j] = load_size;
            maxPEload1 += maxPEload1_p[i][j]; 
        }
    }
    original_imb = (float)((maxPEload0*NUM_PES) - nnz)/nnz;
    improved_imb = (float)((maxPEload1*NUM_PES) - nnz)/nnz;
    totalPEload = maxPEload1;
    return sharedRows;
    
}

std::vector<std::vector<int>> computePEloads1(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, int& totalSize)
{
    std::vector<std::vector<int>> tileSizes(numTilesRows, std::vector<int>(numTilesCols, 0));

    #pragma omp parallel for collapse(2)
    for(int i = 0; i < numTilesRows; i++) {
        for(int j = 0; j < numTilesCols; j++) {
            CSRMatrix* curr_tile = &tiledMatrices[i][j];
            int num_rows = curr_tile->row_offsets.size() - 1;
            std::vector<std::vector<int>> Loads(NUM_PES, std::vector<int>(II_DIST, 0));
            std::vector<std::vector<int>> sorted_rows(num_rows, std::vector<int>(2, 0));

            for(int row = 0; row < num_rows; row++) {
                sorted_rows[row][0] = row;
                sorted_rows[row][1] = curr_tile->row_offsets[row+1] - curr_tile->row_offsets[row];
            }

            sort(sorted_rows.begin(), 
                sorted_rows.end(),
                [] (const std::vector<int> &a, const std::vector<int> &b)
                {
                    return a[1] > b[1];
                });


            for(int row = 0; row < num_rows; row++) {
                int rowSize = sorted_rows[row][1];
                int pe = sorted_rows[row][0] % NUM_PES;
                int min = Loads[pe][0];
                int min_idx = 0;
                for(int ii = 0; ii < II_DIST; ii++) {
                    if(Loads[pe][ii] < min){
                        min = Loads[pe][ii];
                        min_idx = ii;
                    }
                }
                Loads[pe][min_idx] += rowSize;
            }

            int max_load = Loads[0][0];

            for(int p = 0; p < NUM_PES; p++) 
                for(int ii = 0; ii < II_DIST; ii++) 
                    if(Loads[p][ii] > max_load) 
                        max_load = Loads[p][ii];

            tileSizes[i][j] = (max_load+ PADDING) * II_DIST;
        }
    }

    for(int i = 0; i < numTilesRows; i++) 
        for(int j = 0; j < numTilesCols; j++) 
            totalSize += tileSizes[i][j];

    return tileSizes;
}

std::vector<std::vector<int>> computePEloads2(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, 
    std::vector<std::vector<int>> numSharedRows, std::vector<std::vector<std::vector<int>>> sharedRows, int& totalSize) 
{
    std::vector<std::vector<int>> tileSizes(numTilesRows, std::vector<int>(numTilesCols, 0));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < numTilesRows; i++) {
        for (int j = 0; j < numTilesCols; j++) {
            //schedule shared rows first
            std::vector<std::vector<int>> Loads(NUM_PES, std::vector<int>(II_DIST, 0));
            CSRMatrix* curr_tile = &tiledMatrices[i][j];
            int num_rows = curr_tile->row_offsets.size() - 1;

            int num_shared_rows = numSharedRows[i][j];
            std::vector<std::vector<int>> sortedSharedRows(num_shared_rows, std::vector<int>(2, 0));
            for(int k = 0; k < num_shared_rows; k++) {
                int row_id = sharedRows[i][j][k]; 
                int row_size = curr_tile->row_offsets[row_id+1] - curr_tile->row_offsets[row_id];
                sortedSharedRows[k][0] = row_id;
                sortedSharedRows[k][1] = row_size;
            }
            sort(sortedSharedRows.begin(), 
                sortedSharedRows.end(),
                [] (const std::vector<int> &a, const std::vector<int> &b)
                {
                    return a[1] > b[1];
            });
            for(int k = 0; k < num_shared_rows; k++) {
                int row_id = sortedSharedRows[k][0]; 
                int row_size = sortedSharedRows[k][1];
                int load_size = ((row_size - 1)/NUM_PES) + 1;

                for (int pe = 0; pe < NUM_PES; pe++) {
                    int min_idx = 0;
                    int min = Loads[pe][min_idx];

                    for (int ii=0; ii<II_DIST; ii++) {
                        if (Loads[pe][ii] < min) {
                            min_idx = ii;
                            min = Loads[pe][ii];
                        }
                    }

                    Loads[pe][min_idx] += load_size;
                }
            }
            //schedule remaining rows
            int num_rem_rows = num_rows - num_shared_rows;
            std::vector<std::vector<int>> sortedRemRows(num_rem_rows, std::vector<int>(2, 0));
            for(int k = 0, l = 0, m = 0; k < num_rows; k++) {
                if (k == sharedRows[i][j][l]) 
                    l++;

                else {
                    int row_size = curr_tile->row_offsets[k+1] - curr_tile->row_offsets[k];
                    sortedRemRows[m][0] = k;
                    sortedRemRows[m][1] = row_size;
                    m++;
                }
            }
            sort(sortedRemRows.begin(), 
                sortedRemRows.end(),
                [] (const std::vector<int> &a, const std::vector<int> &b)
                {
                    return a[1] > b[1];
            });
            for(int k = 0; k < num_rem_rows; k++) {
                int row_id = sortedRemRows[k][0]; 
                int row_size = sortedRemRows[k][1];
                int pe = row_id % NUM_PES;
                int min_idx = 0;
                int min = Loads[pe][min_idx];

                for (int ii=0; ii<II_DIST; ii++) {
                    if (Loads[pe][ii] < min) {
                        min_idx = ii;
                        min = Loads[pe][ii];
                    }
                }

                Loads[pe][min_idx] += row_size;
            }
            int max_load = Loads[0][0];
            for(int p = 0; p < NUM_PES; p++) {
                for(int ii = 0; ii < II_DIST; ii++) {
                if(Loads[p][ii] > max_load) 
                    max_load = Loads[p][ii];
                }
            }

            tileSizes[i][j] = (max_load+ PADDING) * II_DIST;
            
            //find max pe load
        }
    }

    for(int i = 0; i < numTilesRows; i++) 
        for(int j = 0; j < numTilesCols; j++) 
            totalSize += tileSizes[i][j];
        
    return tileSizes;
}

int prepareAmtx1(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, 
    std::vector<std::vector<int>> tileSizes)
{
  
  int totalSize = 0;
  for(int i = 0; i < numTilesRows; i++) {
    for(int j = 0; j < numTilesCols; j++) {
      totalSize += tileSizes[i][j];
    }
  }

  return totalSize;
}

int prepareAmtx2(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols,
    std::vector<std::vector<int>> numSharedRows, std::vector<std::vector<std::vector<int>>> sharedRows, std::vector<std::vector<int>> tileSizes) 
{
    int totalSize = 0;
    for(int i = 0; i < numTilesRows; i++) {
        for(int j = 0; j < numTilesCols; j++) {
            totalSize += tileSizes[i][j];
        }
    }
    return totalSize; 
}

int prepareAmtx3(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols,
    std::vector<std::vector<int>> numSharedRows, std::vector<std::vector<std::vector<int>>> sharedRows, std::vector<std::vector<int>> tileSizes)
{
  int len = 0;
  for (int i = 0; i < numTilesRows; i++) {
      for (int j = 0; j < numTilesCols; j++) {
          len += tileSizes[i][j];
      }
  }
  return len;
}
 
int prepareAmtx(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, 
    const int Depth, const int Window, const int rows, const int cols, const int nnz, const int USE_ROW_SHARE, const int USE_TREE_ADDER) 
{    
    //compute how balanced the matrix is
        float imb0, imb1;
    int run_len3 = 0;
    std::vector<std::vector<int>> tileSizes3(numTilesRows, std::vector<int>(numTilesCols, 0)); 
    std::vector<std::vector<int>> numSharedRows(numTilesRows, std::vector<int>(numTilesCols, 0));
    std::vector<std::vector<std::vector<int>>> sharedRows = balanceWorkload(tiledMatrices, numTilesRows, 
        numTilesCols, Depth, Depth/2, nnz, numSharedRows, tileSizes3, imb0, imb1, run_len3);
    
    int improvement = (int)((imb0 - imb1)*100/(1 + imb0));
   
    int run_len1 = 0;
    std::vector<std::vector<int>> tileSizes1 = computePEloads1(tiledMatrices, numTilesRows, numTilesCols, run_len1);
    printf("Run Length without Row Sharing, and without Tree Adders: %d\n", run_len1);
    printf("Improvement = %d (%f, %f)\n", improvement, imb0, imb1);

    if (improvement < 10) {
        std::cout << "Input Matrix is Balanced, continuing without row sharing and tree adders " << std::endl;
        // USE_DOUBLE_BUFFER = false;
        return prepareAmtx1(tiledMatrices, numTilesRows, numTilesCols, tileSizes1);
    }

    else if (!USE_ROW_SHARE) {
        std::cout << "Row sharing disabled by user, continuing without row sharing and tree adders " << std::endl;
        return prepareAmtx1(tiledMatrices, numTilesRows, numTilesCols, tileSizes1);
    }
    
    //If there is more than 25% improvement with row sharing, continue with row sharing
    std::cout << "Input Matrix is Imbalanced, continuing with row sharing " << std::endl;

    int run_len2 = 0;
    std::vector<std::vector<int>> tileSizes2 = computePEloads2(tiledMatrices, numTilesRows, numTilesCols, numSharedRows, sharedRows, run_len2);
    printf("Run Length with Row Sharing, and without Tree Adders: %d\n", run_len2);

    #ifndef BUILD_TREE_ADDER
        //if hardware doesnt have tree adder 
        std::cout << "Hardware doesn't have Tree Adder, continuing without Tree Adders" << std::endl;
        return prepareAmtx2(tiledMatrices, numTilesRows, numTilesCols, numSharedRows, sharedRows, tileSizes2);
    #endif

    #ifdef BUILD_TREE_ADDER
        //if user disabled tree adders
        if (!USE_TREE_ADDER) {
            std::cout << "Tree Adder disabled by user, continuing without Tree Adders" << std::endl;
            return prepareAmtx2(tiledMatrices, numTilesRows, numTilesCols, numSharedRows, sharedRows, tileSizes2);
        }

        //using both
        printf("Run Length with Row Sharing, and with Tree Adders: %d\n", run_len3);
        return prepareAmtx3(tiledMatrices, numTilesRows, numTilesCols, numSharedRows, sharedRows, tileSizes3);
    #endif
}

// Function to tile a CSR matrix
std::vector<std::vector<CSRMatrix>> tileCSRMatrix(const CSRMatrix& originalMatrix, int numRows, int numCols, int tileRows, int tileCols, int numTilesRows, int numTilesCols) {
    std::vector<std::vector<CSRMatrix>> tiledMatrices;

    std::vector<std::vector<CSRMatrix>> storedtiledMatrix(numTilesRows, std::vector<CSRMatrix>(numTilesCols));
    for (int i = 0; i < numTilesRows; i++) 
        for (int j = 0; j < numTilesCols; j++) 
            for(int ii = 0; ii < tileRows + 1; ii++) 
                storedtiledMatrix[i][j].row_offsets.push_back(0);

    for (int row = 0; row < numRows; row++)
    {
        int tileRow = row / tileRows;
        int tiledRow = row % tileRows;
        for (int j = originalMatrix.row_offsets[row]; j < originalMatrix.row_offsets[row+1]; j++)
        {
            int col = originalMatrix.col_indices[j];
            float val = originalMatrix.values[j];
            int tileCol = col / tileCols;
            int tiledCol = col % tileCols;

            storedtiledMatrix[tileRow][tileCol].col_indices.push_back(tiledCol);
            storedtiledMatrix[tileRow][tileCol].values.push_back(val);
            storedtiledMatrix[tileRow][tileCol].row_offsets[tiledRow+1]++;
        }
    }

    for (int i = 0; i < numTilesRows; i++) 
        for (int j = 0; j < numTilesCols; j++) 
            for(int ii = 1; ii < tileRows + 1; ii++) 
                storedtiledMatrix[i][j].row_offsets[ii] += storedtiledMatrix[i][j].row_offsets[ii-1];

    return storedtiledMatrix;
}

// function from Serpens and functions to read mtx file
int cmp_by_column_row(const void *aa,
                      const void *bb) {
    rcv * a = (rcv *) aa;
    rcv * b = (rcv *) bb;
    
    if (a->c > b->c) return +1;
    if (a->c < b->c) return -1;
    
    if (a->r > b->r) return +1;
    if (a->r < b->r) return -1;
    
    return 0;
}

void sort_by_fn(int nnz_s,
                std::vector<int> & cooRowIndex,
                std::vector<int> & cooColIndex,
                std::vector<float> & cooVal,
                int (* cmp_func)(const void *, const void *)) {
    rcv * rcv_arr = new rcv[nnz_s];
    
    for(int i = 0; i < nnz_s; ++i) {
        rcv_arr[i].r = cooRowIndex[i];
        rcv_arr[i].c = cooColIndex[i];
        rcv_arr[i].v = cooVal[i];
    }
    
    qsort(rcv_arr, nnz_s, sizeof(rcv), cmp_func);
    
    for(int i = 0; i < nnz_s; ++i) {
        cooRowIndex[i] = rcv_arr[i].r;
        cooColIndex[i] = rcv_arr[i].c;
        cooVal[i] = rcv_arr[i].v;
    }
    
    delete [] rcv_arr;
}

void mm_init_read(FILE * f,
                  char * filename,
                  MM_typecode & matcode,
                  int & m,
                  int & n,
                  int & nnz) {

    if (mm_read_banner(f, &matcode) != 0) {
        std::cout << "Could not process Matrix Market banner for " << filename << std::endl;
        exit(1);
    }
    
    int ret_code;
    if ((ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnz)) != 0) {
        std::cout << "Could not read Matrix Market format for " << filename << std::endl;
        exit(1);
    }
}

void load_S_matrix(FILE* f_A,
                   int nnz_mmio,
                   int & nnz,
                   std::vector<int> & cooRowIndex,
                   std::vector<int> & cooColIndex,
                   std::vector<float> & cooVal,
                   MM_typecode & matcode) {
    
    if (mm_is_complex(matcode)) {
        std::cout << "Reading in a complex matrix, not supported yet!" << std::endl;
        exit(1);
    }
    
    if (!mm_is_symmetric(matcode)) {
        std::cout << "It's an NS matrix.\n";
    } else {
        std::cout << "It's an S matrix.\n";
    }
    
    int r_idx, c_idx;
    float value;
    int idx = 0;
    
    for (int i = 0; i < nnz_mmio; ++i) {
        if (mm_is_pattern(matcode)) {
            fscanf(f_A, "%d %d\n", &r_idx, &c_idx);
            value = 1.0;
        }else {
            fscanf(f_A, "%d %d %f\n", &r_idx, &c_idx, &value);
        }
        
        unsigned int * tmpPointer_v = reinterpret_cast<unsigned int*>(&value);;
        unsigned int uint_v = *tmpPointer_v;
        if (uint_v != 0) {
            if (r_idx < 1 || c_idx < 1) { // report error
                std::cout << "idx = " << idx << " [" << r_idx - 1 << ", " << c_idx - 1 << "] = " << value << std::endl;
                exit(1);
            }
            
            cooRowIndex[idx] = r_idx - 1;
            cooColIndex[idx] = c_idx - 1;
            cooVal[idx] = value;
            idx++;
            
            if (mm_is_symmetric(matcode)) {
                if (r_idx != c_idx) {
                    cooRowIndex[idx] = c_idx - 1;
                    cooColIndex[idx] = r_idx - 1;
                    cooVal[idx] = value;
                    idx++;
                }
            }
        }
    }
    nnz = idx;
}

void readMatrixCSC(char* filename, std::vector<float>& values, std::vector<int>& rowIndices, std::vector<int>& colOffsets, int& rows, int& cols, int& nnz) {
    int nnz_mmio;
    MM_typecode matcode;
    FILE * f_A;
    
    if ((f_A = fopen(filename, "r")) == NULL) {
        std::cout << "Could not open " << filename << std::endl;
        exit(1);
    }
    
    mm_init_read(f_A, filename, matcode, rows, cols, nnz_mmio);
    
    if (!mm_is_coordinate(matcode)) {
        std::cout << "The input matrix file " << filename << "is not a coordinate file!" << std::endl;
        exit(1);
    }
    
    int nnz_alloc = (mm_is_symmetric(matcode))? (nnz_mmio * 2): nnz_mmio;
    //std::cout << "Matrix A -- #row: " << rows << " #col: " << cols << std::endl;
    
    std::vector<int> cooRowIndex(nnz_alloc);
    std::vector<int> cooColIndex(nnz_alloc);
    //eleIndex.resize(nnz_alloc);
    values.resize(nnz_alloc);
    
    //std::cout << "Loading input matrix A from " << filename << "\n";
    
    load_S_matrix(f_A, nnz_mmio, nnz, cooRowIndex, cooColIndex, values, matcode);
    
    fclose(f_A);
    
    sort_by_fn(nnz, cooRowIndex, cooColIndex, values, cmp_by_column_row);
    
    // convert to CSC format
    int M_K = cols;
    colOffsets.resize(M_K+1);
    std::vector<int> counter(M_K, 0);
    
    for (int i = 0; i < nnz; i++) {
        counter[cooColIndex[i]]++;
    }
    
    int t = 0;
    for (int i = 0; i < M_K; i++) {
        t += counter[i];
    }
    
    colOffsets[0] = 0;
    for (int i = 1; i <= M_K; i++) {
        colOffsets[i] = colOffsets[i - 1] + counter[i - 1];
    }
    
    rowIndices.resize(nnz);
    
    for (int i = 0; i < nnz; ++i) {
        rowIndices[i] = cooRowIndex[i];
    }
    
    if (mm_is_symmetric(matcode)) {
        //eleIndex.resize(nnz);
        values.resize(nnz);
    }
}

void convertCSCtoCSR(const std::vector<float>& cscValues, const std::vector<int>& cscRowIndices, const std::vector<int>& cscColOffsets,
                     std::vector<float>& csrValues, std::vector<int>& csrColIndices, std::vector<int>& csrRowOffsets, int rows, int cols, int nnz) {
    // allocate memory
    csrValues.resize(nnz);
    csrColIndices.resize(nnz);
    csrRowOffsets.resize(rows + 1);
    std::vector<int> rowCounts(rows, 0);

    for (int i = 0; i < nnz; i++) {
        rowCounts[cscRowIndices[i]]++;
    }

    // convert rowCounts to cumulative sum
    csrRowOffsets[0] = 0;
    for (int i = 0; i < rows; i++) {
      csrRowOffsets[i+1] = csrRowOffsets[i] + rowCounts[i];
    }

    std::vector<int> rowOffset(rows, 0);
    // fill csrValues and csrColIndices
    for (int j = 0; j < cols; j++) {
        for (int i = cscColOffsets[j]; i < cscColOffsets[j + 1]; i++) {
            int row = cscRowIndices[i];
            int index = csrRowOffsets[row] + rowOffset[row];
            csrValues[index] = cscValues[i];
            csrColIndices[index] = j;
            rowOffset[row]++;
        }
    }
}




int main(int argc, char* argv[]) {

  if (argc != 4){
    std::cout << "Invalid Arguments" << std::endl;
    return 0; 
  }

  char* filename = argv[1];
  NUM_CH = atoi(argv[2]);
  NUM_C_CH = atoi(argv[3]);

  NUM_PES = NUM_CH * PES_PER_CH;

  std::cout << "Reading " << filename << std::endl;

  std::vector<float> cscValues;
  std::vector<int> cscRowIndices;
  std::vector<int> cscColOffsets;
  int rows, cols, nnz;
  readMatrixCSC(filename, cscValues, cscRowIndices, cscColOffsets, rows, cols, nnz);


  CSRMatrix cpuAmtx;
  convertCSCtoCSR(cscValues, cscRowIndices, cscColOffsets, cpuAmtx.values, cpuAmtx.col_indices, cpuAmtx.row_offsets, rows, cols, nnz);


  int Window = 8192;
  int Depth = NUM_PES * 3 * 4096;

  int numTilesCols = (cols - 1)/Window + 1;
  int numTilesRows = (rows - 1)/Depth + 1;

 
  std::vector<std::vector<CSRMatrix>> tiledMatrices = tileCSRMatrix(cpuAmtx, rows, cols, Depth, Window, numTilesRows, numTilesCols);

  int comp_len = (uint32_t)prepareAmtx(tiledMatrices, numTilesRows, numTilesCols, Depth, Window, rows, cols,  nnz, 1, 1);

 

  uint32_t numRowsPerPE = (rows - 1) / NUM_PES + 1;
  uint32_t numCols16 = (cols - 1) / 16 + 1;
  uint32_t numRows16 = (rows - 1) / 16 + 1;
  uint32_t numTiles = (numTilesRows * numTilesCols);
  

  // std::vector<aligned_std::vector<uint64_t>> DEBUGmtx(NUM_CH, aligned_std::vector<uint64_t>(PES_PER_CH*run_len));

  int loadX = numCols16 * numTilesRows;
  printf("loadX:%d\n", loadX);
#ifdef HYBRID_DESIGN
  loadX = std::min(comp_len, loadX);
#endif

  int updateY = ((numRows16 - 1) / NUM_C_CH) + 1;

  int initY = std::max((std::min(rows, Depth)/NUM_PES)-512, 0);

  int total = loadX + initY + comp_len + updateY;

  printf("initY:%d\n", initY);
  

  printf("compAx:%d\n", comp_len);  
  printf("updateY:%d\n", updateY);
  printf("Total:%d\n", total);

  // if (USE_DOUBLE_BUFFER)
  //   printf("Density(%lf) is too low, using double bufferring\n", density);

  // else
  //   printf("Density(%lf) is not too low, Not using double bufferring\n", density);


  return 0;

}