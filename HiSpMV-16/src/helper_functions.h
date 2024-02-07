#ifndef HELPER_FUNCTIONS_H
#define HELPER_FUNCTIONS_H

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <vector>
#include <sstream>
#include <tapa.h>

template <typename T>
using aligned_vector = std::vector<T, tapa::aligned_allocator<T>>;

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

void readMatrixCSC(char * filename, std::vector<float>& values, std::vector<int>& rowIndices, std::vector<int>& colOffsets, 
    int& rows, int& cols, int& nnz);

void convertCSCtoCSR(const std::vector<float>& cscValues, const std::vector<int>& cscRowIndices, const std::vector<int>& cscColOffsets,
    std::vector<float>& csrValues, std::vector<int>& csrColIndices, std::vector<int>& csrRowOffsets, int rows, int cols, int nnz);

void printMatrixCSR(std::vector<float> values, std::vector<int> columns, std::vector<int> rowPtr, 
    int numRows, int numCols);

std::vector<std::vector<CSRMatrix>> tileCSRMatrix(const CSRMatrix& originalMatrix, int numRows, int numCols, int tileRows, int tileCols, 
    int numTilesRows, int numTilesCols);

std::vector<aligned_vector<uint64_t>> prepareAmtx(std::vector<std::vector<CSRMatrix>> tiledMatrices, const int numTilesRows, const int numTilesCols, 
    const int Depth, const int Window, const int rows, const int cols, const int nnz, const int USE_ROW_SHARE, const int USE_TREE_ADDER, bool &USE_DOUBLE_BUFFER);


inline uint64_t encode(bool tileEnd, bool rowEnd, bool sharedRow, uint16_t row, int col, uint32_t val)
{
    uint64_t res = 0; // 16bits
    res |= row;
    res <<= 1;
    res |= tileEnd; //will be 47th bit
    res <<= 1;
    res |= rowEnd; // will be 46th bit
    res <<= 1;
    res |= sharedRow; // will be 45th nit
    res <<= 13;
    res |= col & (0x1FFF); // 13 bits col
    res <<= 32;
    res |= val; //32 bits val
    return res;
}

inline void decode(uint64_t a, bool& tileEnd, bool& rowEnd, bool& sharedRow, uint16_t& row16, uint16_t& col, float& val) {
    uint32_t cval = a & 0xFFFFFFFF;
    a >>= 32;
    col = a & 0x1FFF;
    a >>= 13;
    sharedRow = a & 1;
    a >>= 1;
    rowEnd = a & 1;
    a >>= 1;
    tileEnd = a & 1;
    a >>= 1;
    row16 = a & 0xFFFF;
    val = *(float*)&cval;
}

#endif