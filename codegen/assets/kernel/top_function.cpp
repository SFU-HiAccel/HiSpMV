void SpMV(tapa::mmaps<uint64_v, NUM_CH> A,
          tapa::mmap<float_v16> b,
          tapa::mmaps<float_v16, NUM_C_CH> c_in,
          tapa::mmaps<float_v16, NUM_C_CH> c_out,
          const float alpha, const float beta,
          const uint32_t num_rows_per_pe, const uint32_t num_cols_16,
          const uint16_t num_tiles_r, const uint16_t num_tiles_c,
          const uint32_t num_tiles, const uint32_t len, const uint16_t rp_time,
          const bool USE_DOUBLE_BUFFER)
{


#ifdef HYBRID_DESIGN
    tapa::streams<uint64_v2, NUM_PES_HALF, FIFO_LARGE_DEPTH> FIFO_A_IN("a_in");
    tapa::streams<uint16_v2, NUM_PES_HALF, FIFO_LARGE_DEPTH> FIFO_C_ROW("c_row");
    tapa::streams<float_v2, NUM_PES_HALF, FIFO_LARGE_DEPTH> FIFO_C_VAL("c_val");
    tapa::streams<flags_pkt, NUM_PES_HALF, FIFO_LARGE_DEPTH> FIFO_C_FLAG("c_flag");
#endif

#ifndef HYBRID_DESIGN
    tapa::streams<uint64_v2, NUM_PES_HALF, FIFO_DEPTH> FIFO_A_IN("a_in");
    tapa::streams<uint16_v2, NUM_PES_HALF, FIFO_DEPTH> FIFO_C_ROW("c_row");
    tapa::streams<float_v2, NUM_PES_HALF, FIFO_DEPTH> FIFO_C_VAL("c_val");
    tapa::streams<flags_pkt, NUM_PES_HALF, FIFO_DEPTH> FIFO_C_FLAG("c_flag");
#endif
    
    tapa::streams<float_v16, NUM_PES_HALF + 1, FIFO_DEPTH> FIFO_B_IN("b_in");
    tapa::buffers<
        float[W], //buffer type
        NUM_PES, //num buffer channels 
        1, //n-sections 
        tapa::array_partition<tapa::cyclic<16>>, // partition info
        tapa::memcore<tapa::bram>> //memory core type
        BUFF_B; 

    tapa::streams<Cnoc_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_SHF("c_shf");
    tapa::streams<Cvec_pkt, NUM_PES, FIFO_DEPTH> FIFO_C_BUF("c_buf");

    tapa::streams<float, NUM_PES, FIFO_DEPTH> FIFO_C_ARB("c_arb");
	tapa::streams<float_v16, NUM_C_CH, FIFO_DEPTH> FIFO_C_AB("c_ab");
    tapa::streams<float_v16, NUM_C_CH, FIFO_DEPTH> FIFO_C_IN("c_in");
    tapa::streams<float_v16, NUM_C_CH, FIFO_DEPTH> FIFO_C_OUT("c_out");

    tapa::task()
        .invoke<tapa::join, NUM_CH>(MM2S_A, A, FIFO_A_IN, len, rp_time)
        .invoke(MM2S_B, b, FIFO_B_IN, num_tiles_r, num_cols_16, rp_time)
        .invoke<tapa::join, NUM_PES_HALF>(LoadB, FIFO_B_IN, FIFO_B_IN, BUFF_B, num_cols_16, num_tiles, USE_DOUBLE_BUFFER)
        .invoke<tapa::join, NUM_PES_HALF>(ComputeAB, FIFO_A_IN, FIFO_C_FLAG, FIFO_C_ROW, FIFO_C_VAL, BUFF_B, num_tiles, USE_DOUBLE_BUFFER)
        .invoke<tapa::join>(DummyRead, FIFO_B_IN, num_cols_16, num_tiles_r, rp_time)
        .invoke<tapa::join, NUM_PES_HALF>(TreeAdder, FIFO_C_ROW, FIFO_C_VAL, FIFO_C_FLAG, FIFO_C_SHF)
        .invoke<tapa::join, NUM_PES>(ResultBuff, FIFO_C_BUF, FIFO_C_ARB, num_rows_per_pe, num_tiles_c, rp_time)
		.invoke(Arbiter_C, FIFO_C_ARB, FIFO_C_AB, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(MM2S_C, c_in, FIFO_C_IN, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(Compute_C, FIFO_C_IN, FIFO_C_AB, FIFO_C_OUT, alpha, beta, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(S2MM_C, FIFO_C_OUT, c_out, num_rows_per_pe, rp_time);
}