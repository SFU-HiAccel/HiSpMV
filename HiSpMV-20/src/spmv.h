#ifndef SPMV_H
#define SPMV_H
#include <cstdint>
#include <tapa.h>
#include <ap_int.h>

#define NUM_CH 20
#define NUM_C_CH 2
#define LOG_2_NUM_PES 8
#define NUM_PES 160
#define NUM_PES_HALF 80
#define HYBRID_DESIGN

#define II_DIST 5
#define PADDING 1
#define FIFO_DEPTH 2
#define FIFO_LARGE_DEPTH 8
#define PES_PER_CH 8

#define W 8192
#define W_16 (W/16)

#define MAX_ROWS_PER_PE (2*4096)
#define D (NUM_PES * MAX_ROWS_PER_PE)

using uint64_v = tapa::vec_t<uint64_t, PES_PER_CH>;
using uint64_v2 = tapa::vec_t<uint64_t, 2>;
using float_v2 = tapa::vec_t<float, 2>;
using uint16_v2 = tapa::vec_t<uint16_t, 2>;
using float_v16 = tapa::vec_t<float, 16>;

struct flags_pkt {
    bool sharedRow;
    bool tileEnd;
    bool last;
};

struct Cnoc_pkt {
    bool dummy;
    bool last;
    bool tileEnd;
    bool sharedRow;
    uint16_t row16;
    uint8_t bank;
    float val;
};

struct Cvec_pkt {
    bool dummy;
    bool tileEnd;
    uint16_t row16;
    float val;
};

void SpMV(tapa::mmaps<uint64_v, NUM_CH> A,
          tapa::mmap<float_v16> b,
          tapa::mmaps<float_v16, NUM_C_CH> c_in,
          tapa::mmaps<float_v16, NUM_C_CH> c_out,
          const float alpha, const float beta,
          const uint32_t num_rows_per_pe, const uint32_t num_cols_16,
          const uint16_t num_tiles_r, const uint16_t num_tiles_c,
          const uint32_t num_tiles, const uint32_t len, const uint16_t rp_time, const bool USE_DOUBLE_BUFFER);
#endif