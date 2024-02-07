#include "spmv.h"

inline void async_readA(tapa::async_mmap<uint64_v> & A,
                       tapa::ostreams<uint64_v2, PES_PER_CH/2> & fifo_A,
                       const uint32_t A_len,
                       uint32_t& i_req,
                       uint32_t& i_resp) {
    #pragma HLS inline
    uint64_v tmp;
    if ((i_req < A_len) &
        !A.read_addr.full()) {
        A.read_addr.try_write(i_req);
        ++i_req;
    }

    bool full = 0;
    for(int a = 0; a < PES_PER_CH/2; a++)
    #pragma HLS UNROLL
        full |= fifo_A[a].full();

    if (!full & !A.read_data.empty()) {
        A.read_data.try_read(tmp);
        for(int a = 0; a < PES_PER_CH/2; a++) {
        #pragma HLS UNROLL
            uint64_v2 t;
            t[0] = tmp[a*2];
            t[1] = tmp[a*2 + 1];
            fifo_A[a].try_write(t);
        }
        ++i_resp;
    }
}

inline void async_readB(tapa::async_mmap<float_v16> & B,
                       tapa::ostream<float_v16> & fifo_B,
                       const uint32_t B_len,
                       uint32_t & i_req,
                       uint32_t & i_resp) {
    #pragma HLS inline
    if ((i_req < B_len) &
        !B.read_addr.full()) {
        B.read_addr.try_write(i_req);
        ++i_req;
    }

    if (!fifo_B.full() & !B.read_data.empty()) {
        float_v16 tmp;
        B.read_data.try_read(tmp);
        fifo_B.try_write(tmp);
        ++i_resp;
    }
}

inline void async_readC(tapa::async_mmap<float_v16> & C,
                       tapa::ostream<float_v16> & fifo_C,
                       const uint32_t C_len,
                       uint32_t & i_req,
                       uint32_t & i_resp) {
    #pragma HLS inline
    if ((i_req < C_len) &
        !C.read_addr.full()) {
        C.read_addr.try_write(i_req);
        ++i_req;
    }

    if (!fifo_C.full() & !C.read_data.empty()) {
        float_v16 tmp;
        C.read_data.try_read(tmp);
        fifo_C.try_write(tmp);
        ++i_resp;
    }
}

inline void async_writeC(tapa::async_mmap<float_v16>& mem,
    tapa::istream<float_v16>& fifo,
    uint32_t count) {
    #pragma HLS inline

  for(int i_req = 0, i_resp = 0; i_resp < count;) {
    #pragma HLS pipeline II=1
    // issue write requests
    float_v16 tmp;

    if (i_req < count &&
        !fifo.empty() &&
        !mem.write_addr.full() &&
        !mem.write_data.full()) {
    
   
      tmp = fifo.read(nullptr);

      mem.write_addr.try_write(i_req);
      mem.write_data.try_write(tmp);
      ++i_req;
    }

    // receive acks of write success
    if (!mem.write_resp.empty()) {
      i_resp += unsigned(mem.write_resp.read(nullptr)) + 1;
    }
  }
} 

void MM2S_A(tapa::async_mmap<uint64_v>& mmap,
    tapa::ostreams<uint64_v2, PES_PER_CH/2>& streams,
    const uint32_t len, const uint16_t rp_time) {
    for(uint16_t rp = 0; rp < rp_time; rp++) {
        for(uint32_t i_req = 0, i_resp = 0; i_resp < len;) {
        #pragma HLS pipeline II=1
        async_readA(mmap,
                    streams,
                    len,
                    i_req, i_resp); 
        }
    }
    printf("MM2S_A\n");
}

void MM2S_B(tapa::async_mmap<float_v16>& mmap,
    tapa::ostream<float_v16>& stream,
    const uint16_t num_tiles_r, const uint32_t num_cols_16, const uint16_t rp_time) {
    for (int r = 0; r < (num_tiles_r * rp_time); r++) {
        for(uint32_t i_req = 0, i_resp = 0; i_resp < num_cols_16; ) {
        #pragma HLS pipeline II=1
            async_readB(mmap,
                        stream,
                        num_cols_16,
                        i_req, i_resp); 
        }
    }
    printf("MM2S_B\n");
}

void MM2S_C(tapa::async_mmap<float_v16>& mmap,
    tapa::ostream<float_v16>& stream,
    const uint32_t num_rows_per_pe, const uint16_t rp_time) {
    uint32_t len = (num_rows_per_pe * NUM_PES / 16) / NUM_C_CH;
    for(uint16_t rp = 0; rp < rp_time; rp++){
        for(uint32_t i_req = 0, i_resp = 0; i_resp < len; ) {
        #pragma HLS pipeline II=1
            async_readC(mmap,
                        stream,
                        len,
                        i_req, i_resp); 
        }
    }
    printf("MM2S_C\n");
}

void S2MM_C(tapa::istream<float_v16>& stream,
    tapa::async_mmap<float_v16>& mmap, 
    const uint32_t num_rows_per_pe, const uint16_t rp_time) {
    uint32_t len = (num_rows_per_pe * NUM_PES / 16) / NUM_C_CH;
    for(uint16_t rp = 0; rp < rp_time; rp++){
        async_writeC(mmap, stream, len);
    }
    printf("S2MM_C\n");
}

void Arbiter_C(tapa::istreams<float, NUM_PES>& c_ab_in, tapa::ostreams<float_v16, NUM_C_CH>& c_ab_out, const uint32_t num_rows_per_pe, const uint16_t rp_time) {
	float_v16 tmp_in[NUM_C_CH];
	#pragma HLS aggregate variable=tmp_in
	for (uint32_t i = 0; i < (num_rows_per_pe * rp_time) ; i++) {
        for(uint8_t j = 0; j < (NUM_PES >> 4) / NUM_C_CH; j++) {
			for(int  jj=0; jj<16*NUM_C_CH; jj++) 
            #pragma HLS UNROLL 
                c_ab_in[(j<<4)*NUM_C_CH + jj] >> tmp_in[jj>>4][jj&15];
			for(int c = 0; c < NUM_C_CH; c++)
            #pragma HLS UNROLL
                c_ab_out[c] << tmp_in[c];
		}
	}
}

void Compute_C(tapa::istream<float_v16>& c_in, tapa::istream<float_v16>& c_ab, 
    tapa::ostream<float_v16>& c_out, 
    const float alpha, const float beta, const uint32_t num_rows_per_pe,const uint16_t rp_time) {
    float_v16 tmp_in0, tmp_in1, tmp_out;
    #pragma HLS aggregate variable=tmp_in0
    #pragma HLS aggregate variable=tmp_in1
    #pragma HLS aggregate variable=tmp_out
    for (uint32_t i = 0; i < (num_rows_per_pe * rp_time); i++) {
        for(uint8_t j = 0; j < (NUM_PES >> 4) / NUM_C_CH; j++) {
            #pragma HLS PIPELINE II=1
            c_in >> tmp_in0;
			c_ab >> tmp_in1;
            for(int  jj=0; jj<16; jj++) 
            #pragma HLS UNROLL
                tmp_out[jj] = (beta * tmp_in0[jj]) + (alpha * tmp_in1[jj]); 
            c_out << tmp_out;
        }
    }
    printf("Compute C\n");
}


#ifdef HYBRID_DESIGN
void ComputeAB(tapa::istream<uint64_v2>& a_in, tapa::ostream<flags_pkt>& c_flags, 
    tapa::ostream<uint16_v2>& c_row, tapa::ostream<float_v2>& c_val, 
    tapa::ibuffers<float[W], 
        2, //channels
        1, //n-sections 
        tapa::array_partition<tapa::cyclic<16>>, // partition info
        tapa::memcore<tapa::bram>>& local_B, //mem core
        const int last_tile_idx, 
        const bool USE_DOUBLE_BUFFER) 
{
    bool last = 0;
    
    compute:
    for (uint32_t i = 1; !last ;i++) {
        uint64_v2 temp_in;
        uint16_v2 row_out;
        float_v2 val_out;
        flags_pkt flags_out;
        float val_in[2];
        ap_uint<13> col_id[2];

        bool read_new_A;
        float b_line[16];

        if(USE_DOUBLE_BUFFER) {
            auto section = local_B[!(i%2)].acquire();
            auto& buf_ref = section();

            for (bool tileEnd = false, read_new_A = true; !(tileEnd); ) {
            #pragma HLS loop_tripcount min=1 max=100000
            #pragma HLS PIPELINE II=1
                if (read_new_A) {
                    temp_in = a_in.read();
                    for(int p = 0; p < 2; p++) {
                    #pragma HLS UNROLL
                        uint64_t a = temp_in[p];
                        uint32_t val_bits = a & 0xFFFFFFFF;
                        val_in[p] = *(float*)(&val_bits);
                        col_id[p] = (a >> 32) & 0x1FFF;
                        bool rowEnd = (a >> 46) & 1;
                        uint16_t row = (a >> 48) & 0xFFFF;
                        row_out[p] = row | ((uint16_t)rowEnd << 15);
                    }
                    flags_out.sharedRow = (temp_in[0] >> 45) & 1;
                    flags_out.tileEnd   = (temp_in[0] >> 47) & 1;
                    flags_out.last      = (i==last_tile_idx) & flags_out.tileEnd;

                    for (int l = 0; l < 16; l++) 
                    #pragma HLS UNROLL
                        b_line[l] = buf_ref[(col_id[0] >> 4)*16 + l];
                    

                    if ((col_id[0] >> 4) == (col_id[1] >> 4)) {
                        for(int p = 0; p < 2; p++) 
                        #pragma HLS UNROLL
                            val_out[p] = val_in[p] * b_line[col_id[p] & 0xF];  
                        
                        c_val << val_out;
                        c_row << row_out;
                        c_flags << flags_out;
                        read_new_A = true;
                        tileEnd = flags_out.tileEnd;
                        last = flags_out.last;
                    }

                    else {
                        val_out[0] =  val_in[0] * b_line[col_id[0] & 0xF];
                        read_new_A = false;
                    }
                }

                else {
                    for (int l = 0; l < 16; l++) 
                    #pragma HLS UNROLL
                        b_line[l] = buf_ref[(col_id[1] >> 4)*16 + l];
                    
                    val_out[1] = val_in[1] * b_line[col_id[1] & 0xF];
                    c_val << val_out;
                    c_row << row_out;
                    c_flags << flags_out;
                    read_new_A = true;
                    tileEnd = flags_out.tileEnd;
                    last = flags_out.last;
                }
            }
        }

        else{
            auto section0 = local_B[0].acquire();
            auto section1 = local_B[1].acquire();
            auto& buf_ref0 = section0();
            auto& buf_ref1 = section1();

            for (bool tileEnd = false; !(tileEnd); ) {
            #pragma HLS loop_tripcount min=1 max=100000
            #pragma HLS PIPELINE II=1
                temp_in = a_in.read();
                for(int p = 0; p < 2; p++) {
                #pragma HLS UNROLL
                    uint64_t a = temp_in[p];
                    uint32_t val_bits = a & 0xFFFFFFFF;
                    val_in[p] = *(float*)(&val_bits);
                    col_id[p] = (a >> 32) & 0x1FFF;
                    bool rowEnd = (a >> 46) & 1;
                    uint16_t row = (a >> 48) & 0xFFFF;
                    row_out[p] = row | ((uint16_t)rowEnd << 15);
                }
                val_out[0] = val_in[0] * buf_ref0[col_id[0]];
                val_out[1] = val_in[1] * buf_ref1[col_id[1]];
                flags_out.sharedRow = (temp_in[0] >> 45) & 1;
                flags_out.tileEnd   = (temp_in[0] >> 47) & 1;
                flags_out.last      = (i==last_tile_idx) & flags_out.tileEnd;
                c_val << val_out;
                c_row << row_out;
                c_flags << flags_out;
                tileEnd = flags_out.tileEnd;
                last = flags_out.last;
            }
        }
    }
    printf("PEG compute\n");
}

void LoadB(tapa::istream<float_v16>& b_in, tapa::ostream<float_v16>& b_out,
    tapa::obuffers<float[W],
        2, //num-channels 
        1, //n-sections 
        tapa::array_partition<tapa::cyclic<16>>, // partition info
        tapa::memcore<tapa::bram>>& local_B, //mem core
        const int num_cols_16, const int num_tiles,
        const bool USE_DOUBLE_BUFFER)
{
    uint32_t c = 0;
    load_b:
    for (int i = 0; i < num_tiles; i++) {

        if(c == num_cols_16)
                c = 0;

        if (USE_DOUBLE_BUFFER) {
            auto section = local_B[i%2].acquire();
            auto& buf_ref = section();

            for(uint16_t w = 0; (w < W_16) && (c < (num_cols_16)) ; w++, c++) {
            #pragma HLS loop_tripcount min=0 max=512
            #pragma HLS PIPELINE II=1
                float_v16 temp = b_in.read();

                for(uint8_t q = 0; q < 16; q++) 
                #pragma HLS UNROLL
                    buf_ref[w*16 + q] = temp[q];

                b_out << temp;
            }
        }

        else {
            auto section0 = local_B[0].acquire();
            auto section1 = local_B[1].acquire();
            auto& buf_ref0 = section0();
            auto& buf_ref1 = section1();

            for(uint16_t w = 0; (w < W_16) && (c < (num_cols_16)) ; w++, c++) {
            #pragma HLS loop_tripcount min=0 max=512
            #pragma HLS PIPELINE II=1
                float_v16 temp = b_in.read();

                for(uint8_t q = 0; q < 16; q++) {
                #pragma HLS UNROLL
                    buf_ref0[w*16 + q] = temp[q];
                    buf_ref1[w*16 + q] = temp[q];
                }
                    
                b_out << temp;
            }
        }
    }
    printf("PEG load\n");
}
#endif

#ifndef HYBRID_DESIGN
void ComputeAB(tapa::istream<uint64_v2>& a_in, tapa::ostream<flags_pkt>& c_flags, 
    tapa::ostream<uint16_v2>& c_row, tapa::ostream<float_v2>& c_val, 
    tapa::ibuffers<float[W], 
        2, //channels
        1, //n-sections 
        tapa::array_partition<tapa::cyclic<16>>, // partition info
        tapa::memcore<tapa::bram>>& local_B, //mem core
        const int last_tile_idx, 
        const bool USE_DOUBLE_BUFFER) 
{
    bool last = 0;
    
    compute:
    for (uint32_t i = 1; !last ;i++) {
        uint64_v2 temp_in;
        uint16_v2 row_out;
        float_v2 val_out;
        flags_pkt flags_out;
        float val_in[2];
        ap_uint<13> col_id[2];

        auto section0 = local_B[0].acquire();
        auto section1 = local_B[1].acquire();
        auto& buf_ref0 = section0();
        auto& buf_ref1 = section1();

        for (bool tileEnd = false; !(tileEnd); ) {
        #pragma HLS loop_tripcount min=1 max=100000
        #pragma HLS PIPELINE II=1
            temp_in = a_in.read();
            for(int p = 0; p < 2; p++) {
            #pragma HLS UNROLL
                uint64_t a = temp_in[p];
                uint32_t val_bits = a & 0xFFFFFFFF;
                val_in[p] = *(float*)(&val_bits);
                col_id[p] = (a >> 32) & 0x1FFF;
                bool rowEnd = (a >> 46) & 1;
                uint16_t row = (a >> 48) & 0xFFFF;
                row_out[p] = row | ((uint16_t)rowEnd << 15);
            }
            val_out[0] = val_in[0] * buf_ref0[col_id[0]];
            val_out[1] = val_in[1] * buf_ref1[col_id[1]];
            flags_out.sharedRow = (temp_in[0] >> 45) & 1;
            flags_out.tileEnd   = (temp_in[0] >> 47) & 1;
            flags_out.last      = (i==last_tile_idx) & flags_out.tileEnd;
            c_val << val_out;
            c_row << row_out;
            c_flags << flags_out;
            tileEnd = flags_out.tileEnd;
            last = flags_out.last;
        }
    }
    printf("PEG compute\n");
}

void LoadB(tapa::istream<float_v16>& b_in, tapa::ostream<float_v16>& b_out,
    tapa::obuffers<float[W],
        2, //num-channels 
        1, //n-sections 
        tapa::array_partition<tapa::cyclic<16>>, // partition info
        tapa::memcore<tapa::bram>>& local_B, //mem core
        const int num_cols_16, const int num_tiles,
        const bool USE_DOUBLE_BUFFER)
{
    uint32_t c = 0;
    load_b:
    for (int i = 0; i < num_tiles; i++) {
        if(c == num_cols_16)
                c = 0;

        auto section0 = local_B[0].acquire();
        auto section1 = local_B[1].acquire();
        auto& buf_ref0 = section0();
        auto& buf_ref1 = section1();

        for(uint16_t w = 0; (w < W_16) && (c < (num_cols_16)) ; w++, c++) {
        #pragma HLS loop_tripcount min=0 max=512
        #pragma HLS PIPELINE II=1
            float_v16 temp = b_in.read();

            for(uint8_t q = 0; q < 16; q++) {
            #pragma HLS UNROLL
                buf_ref0[w*16 + q] = temp[q];
                buf_ref1[w*16 + q] = temp[q];
            }
                
            b_out << temp;
        }
    }
    printf("PEG load\n");
}
#endif

    
void DummyRead(tapa::istream<float_v16>& b_in,  const uint32_t num_cols_16, const uint16_t num_tiles_r, const uint16_t rp_time) {
    uint64_t run_len = ((uint64_t)num_cols_16 * (num_tiles_r * rp_time));
	// printf("Dummy Read, Run Len: %ld, %d, %d, %d\n", run_len, num_cols_16, num_tiles_r, rp_time);
    for(uint64_t i = 0; i < run_len; i++) {
        #pragma HLS PIPELINE II=1
            float_v16 tmp = b_in.read();
    }
    printf("Dummy Read\n");
}


#ifndef BUILD_TREE_ADDER
void TreeAdder(tapa::istream<uint16_v2>& c_row, tapa::istream<float_v2>& c_val,
        tapa::istream<flags_pkt>& c_flags, tapa::ostreams<Cnoc_pkt, 2>& c_out) { 
    main:
    for (bool last = false;!(last);) {
        compute:
        for (bool tileEnd = false; !(tileEnd);) {
        #pragma HLS PIPELINE II=1
            uint16_v2 row_in = c_row.read();
            float_v2 val_in = c_val.read();
            flags_pkt flags_in = c_flags.read();
            
            uint8_t shared_bank = row_in[0] & ((1U << LOG_2_NUM_PES) - 1);
            uint16_t shared_row16 = row_in[1];

            for(int p = 0; p < 2; p++) {
            #pragma HLS UNROLL
                uint16_t row16 = row_in[p];
                float val = val_in[p];

                uint8_t bank = (flags_in.sharedRow) ? shared_bank : 0;
                uint16_t row = (flags_in.sharedRow) ? shared_row16 : row16;
                bool rowEnd = (row >> 15) & 1;

                Cnoc_pkt curr_out;

                curr_out.dummy = !(rowEnd);
                curr_out.sharedRow = flags_in.sharedRow;
                curr_out.last = flags_in.last;
                curr_out.tileEnd = flags_in.tileEnd;
                curr_out.row16 = (uint16_t)(row & 0x7FFF);
                curr_out.bank = (uint8_t)(bank & ((1U << LOG_2_NUM_PES) - 1));
                curr_out.val = val;

                c_out[p] << curr_out;
            }

            tileEnd = flags_in.tileEnd;
            last = flags_in.last;
        }
    }
    printf("Tree Adder \n");
}
#endif

#ifdef BUILD_TREE_ADDER
void TreeAdder(tapa::istream<uint16_v2>& c_row, tapa::istream<float_v2>& c_val,
        tapa::istream<flags_pkt>& c_flags, tapa::ostreams<Cnoc_pkt, 2>& c_out) { 
    float val_buff_part[2][II_DIST];
    ap_uint<24> row_buff_part[2][II_DIST];
    #pragma HLS bind_storage variable=val_buff_part type=RAM_2P impl=LUTRAM
    #pragma HLS array_partition variable=val_buff_part type=complete

    #pragma HLS bind_storage variable=row_buff_part type=RAM_2P impl=LUTRAM
    #pragma HLS array_partition variable=row_buff_part type=complete

    main:
    for (bool last = false;!(last);) {
        init:
        for (uint8_t p = 0; p < 2; p++) {
        #pragma HLS UNROLL
            for (uint8_t k = 0; k < II_DIST; k++) {
            #pragma HLS UNROLL
                row_buff_part[p][k] = 0;
                val_buff_part[p][k] = 0;
            }
        }

        compute:
        for (bool tileEnd = false; !(tileEnd);) {
        #pragma HLS PIPELINE II=1
            uint16_v2 row_in = c_row.read();
            float_v2 val_in = c_val.read();
            flags_pkt flags_in = c_flags.read();
            
            uint8_t shared_bank = row_in[0] & ((1U << LOG_2_NUM_PES) - 1);
            uint16_t shared_row16 = row_in[1];

            for(int p = 0; p < 2; p++) {
            #pragma HLS UNROLL
                uint16_t row16 = row_in[p];
                float val = val_in[p];

                uint8_t bank = (flags_in.sharedRow) ? shared_bank : 0;
                uint16_t row = (flags_in.sharedRow) ? shared_row16 : row16;
                bool rowEnd = (row >> 15) & 1;
                ap_uint<24> curr_row = ((ap_uint<24>)((row & 0x7FFF) | (flags_in.sharedRow << 15)) << LOG_2_NUM_PES) | (bank & ((1U << LOG_2_NUM_PES) - 1));

                // if (n+p == 0) 
                //     printf("%1d, %1d, %1d, %5d, %.6f\n", flags_in.tileEnd, flags_in.sharedRow, rowEnd, (row & 0x7FFF), val);

                for (uint8_t l = 0; l < II_DIST-1; l++) {
                #pragma HLS UNROLL
                    row_buff_part[p][l] = row_buff_part[p][l+1]; 
                    val_buff_part[p][l] = val_buff_part[p][l+1]; 
                }

                val_buff_part[p][II_DIST-1] = val;
                row_buff_part[p][II_DIST-1] = curr_row;

                float temp[II_DIST];
                for (uint8_t l = 0; l < II_DIST; l++) 
                #pragma HLS UNROLL
                    temp[l] = (row_buff_part[p][l] == curr_row) ? val_buff_part[p][l] : 0;    

                for (uint8_t l = 1; l < II_DIST; l++) 
                #pragma HLS UNROLL
                    temp[0] += temp[l];  

                Cnoc_pkt curr_out;

                curr_out.dummy = !(rowEnd);
                curr_out.sharedRow = flags_in.sharedRow;
                curr_out.last = flags_in.last;
                curr_out.tileEnd = flags_in.tileEnd;
                curr_out.row16 = (uint16_t)((curr_row >> LOG_2_NUM_PES) & 0x7FFF);
                curr_out.bank = (uint8_t)(curr_row & ((1U << LOG_2_NUM_PES) - 1));
                curr_out.val = temp[0];

                c_out[p] << curr_out;
            }

            tileEnd = flags_in.tileEnd;
            last = flags_in.last;
        }
    }
    printf("Tree Adder\n");
}
#endif

void ResultBuff(tapa::istream<Cvec_pkt>& c_in, 
    tapa::ostream<float>& c_out, 
    const uint32_t num_rows_per_pe, const uint16_t num_tiles_c, const uint16_t rp_time) {

    float local_C[MAX_ROWS_PER_PE];
    #pragma HLS bind_storage variable=local_C type=RAM_2P impl=URAM

    float temp_val[8];
	uint16_t temp_row[8];

    #pragma HLS bind_storage variable=temp_val type=RAM_2P impl=LUTRAM
	#pragma HLS bind_storage variable=temp_row type=RAM_2P impl=LUTRAM
    //initialise 
    for(uint16_t rp = 0; rp < rp_time; rp++)
    {
        uint32_t r = 0;
        init0:
        for (uint32_t i = 0; (i < MAX_ROWS_PER_PE) && (i < num_rows_per_pe);) {
        #pragma HLS PIPELINE II=1
		#pragma HLS bind_op variable=i op=add impl=dsp
            local_C[i] = 0;
			i = i + 1;
        }

        ap_uint<3> idx = II_DIST;
        ap_uint<3> t_idx = 0;
        main:
        for(;(r < num_rows_per_pe);) {
            for (uint16_t i = 0; i < num_tiles_c; i++) {
                init:
                for (int j = 0; j < 8; j++) {
                #pragma HLS PIPELINE
                    temp_val[j] = 0;
					temp_row[j] = 0xFFFF;
                }

                acc:
                for (bool tileEnd = false; !(tileEnd); idx++, t_idx++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS DEPENDENCE true type=inter variable=temp_val direction=RAW distance=5
                #pragma HLS DEPENDENCE false type=inter variable=local_C 
                    Cvec_pkt curr_in = c_in.read();
                    if (!curr_in.dummy) {
                        #pragma HLS bind_op variable=temp_val op=fadd latency=4 impl=fabric
                        temp_val[idx] = (curr_in.val) + ((temp_row[t_idx] == curr_in.row16) ? temp_val[t_idx] : local_C[curr_in.row16]);
                        local_C[curr_in.row16] = temp_val[idx];
                        temp_row[idx] =  curr_in.row16;
						// printf("%d, %d, %f, %f, %f\n", temp_row[t_idx], temp_row[idx], temp_val[t_idx], temp_val[idx], curr_in.val );
					}
                    tileEnd = curr_in.tileEnd;
                }
            }
            out:
            for (uint32_t i = 0; (i < MAX_ROWS_PER_PE) && (r < num_rows_per_pe);) {
            #pragma HLS PIPELINE II=1
				c_out << local_C[i];
				local_C[i] = 0;

			#pragma HLS bind_op variable=i op=add impl=dsp
				i = i + 1;
			#pragma HLS bind_op variable=r op=add impl=dsp
				r = r + 1;
            }
        }
    }
    printf("Result Buffer\n");
}

void ADD_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        
        float temp[2];
        bool dummy[2];
        float sum = curr_in0.val + curr_in1.val;

        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy))
        {
            temp[0] = sum;
            temp[1] = 0;

            dummy[0] = false;
            dummy[1] = true;
        }

        else {
            temp[0] = curr_in0.val;
            temp[1] = curr_in1.val;

            dummy[0] = curr_in0.dummy;
            dummy[1] = curr_in1.dummy;
        }

        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0];
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;
        curr_out0.val = temp[0];
        curr_out1.val = temp[1];

        c_out0 << curr_out0;
        c_out1 << curr_out1;

        last = curr_in0.last & curr_in1.last;    
    }
}

void ADD_1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        
        float temp[2];
        bool dummy[2];
        float sum = curr_in0.val + curr_in1.val;

        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy))
        {
            temp[1] = sum;
            temp[0] = 0;

            dummy[1] = false;
            dummy[0] = true;
        }

        else {
            temp[0] = curr_in0.val;
            temp[1] = curr_in1.val;

            dummy[0] = curr_in0.dummy;
            dummy[1] = curr_in1.dummy;
        }

        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0];
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;
        curr_out0.val = temp[0];
        curr_out1.val = temp[1];

        c_out0 << curr_out0;
        c_out1 << curr_out1;
        
        last = curr_in0.last & curr_in1.last;    
    }
}

void ADD_X(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {

    for(bool last = false; !last; ) {
        #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        
        float temp[2];
        bool dummy[2];
        float sum = curr_in0.val + curr_in1.val;

        
        if ((curr_in0.sharedRow) & !(curr_in0.dummy | curr_in1.dummy))
        {
            bool i = ((curr_in0.bank >> (LOG_2_NUM_PES - 1)) & 1);
            temp[i] = sum;
            temp[!i] = 0;

            dummy[i] = false;
            dummy[!i] = true;
        }

        else {
            temp[0] = curr_in0.val;
            temp[1] = curr_in1.val;

            dummy[0] = curr_in0.dummy;
            dummy[1] = curr_in1.dummy;
        }
        Cnoc_pkt curr_out0, curr_out1;

        curr_out0.last = curr_in0.last;
        curr_out1.last = curr_in1.last;
        curr_out0.bank = curr_in0.bank;
        curr_out1.bank = curr_in1.bank;
        curr_out0.dummy = dummy[0];
        curr_out1.dummy = dummy[1];
        curr_out0.tileEnd = curr_in0.tileEnd;
        curr_out1.tileEnd = curr_in1.tileEnd;
        curr_out0.row16 = curr_in0.row16;
        curr_out1.row16 = curr_in1.row16;
        curr_out0.sharedRow = curr_in0.sharedRow;
        curr_out1.sharedRow = curr_in1.sharedRow;
        curr_out0.val = temp[0];
        curr_out1.val = temp[1];

        // printf("ADDX: %d, %d => %d, %d \n", curr_in0.bank, curr_in1.bank, curr_out0.bank, curr_out1.bank);
        c_out0 << curr_out0;
        c_out1 << curr_out1;

        last = curr_in0.last & curr_in1.last;
    }  
}

void SWB0_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    // tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    tapa::ostream<Cvec_pkt>& c_out0, tapa::ostream<Cvec_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cvec_pkt curr_out[2];

        bool i = (curr_in0.bank & 1) && curr_in0.sharedRow;
        curr_out[i].dummy = curr_in0.dummy;
        curr_out[!i].dummy = curr_in1.dummy;
        curr_out[i].tileEnd = curr_in0.tileEnd;
        curr_out[!i].tileEnd = curr_in1.tileEnd;
        curr_out[i].row16 = curr_in0.row16;
        curr_out[!i].row16 = curr_in1.row16;
        curr_out[i].val = curr_in0.val;
        curr_out[!i].val = curr_in1.val;

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

void SWB1_0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cvec_pkt>& c_out0, tapa::ostream<Cvec_pkt>& c_out1) {
    // tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cvec_pkt curr_out[2];


        bool i = (curr_in1.bank & 1) || (!curr_in1.sharedRow);
        curr_out[i].dummy = curr_in1.dummy;
        curr_out[!i].dummy = curr_in0.dummy;
        curr_out[i].tileEnd = curr_in1.tileEnd;
        curr_out[!i].tileEnd = curr_in0.tileEnd;
        curr_out[i].row16 = curr_in1.row16;
        curr_out[!i].row16 = curr_in0.row16;
        curr_out[i].val = curr_in1.val;
        curr_out[!i].val = curr_in0.val;


        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

template<int n>
void SWB0(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];

        bool i = ((curr_in0.bank >> n) & 1) && (curr_in0.sharedRow);
        curr_out[i] = curr_in0;
        curr_out[!i] = curr_in1;

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

template<int n>
void SWB1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];


        bool i = ((curr_in1.bank >> n) & 1) || (!curr_in1.sharedRow);
        curr_out[i] = curr_in1;
        curr_out[!i] = curr_in0;

        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}

void SSW(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
    tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
    for(bool last = false; !last; ) {
    #pragma HLS Pipeline II=1
        Cnoc_pkt curr_in0 = c_in0.read();
        Cnoc_pkt curr_in1 = c_in1.read();
        Cnoc_pkt curr_out[2];

        // if (curr_in0.sharedRow)
        // {
        //     // printf("Sharing \n");
        bool i = curr_in0.sharedRow;

        curr_out[i] = curr_in0;
        curr_out[!i] = curr_in1;


        c_out0 << curr_out[0];
        c_out1 << curr_out[1];

        last = curr_in0.last & curr_in1.last;    
    }
}
void SWB0_1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<1>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_1(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<1>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_2(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<2>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_2(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<2>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_3(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<3>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_3(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<3>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_4(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<4>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_4(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<4>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_5(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<5>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_5(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<5>(c_in0, c_in1, c_out0, c_out1);
}

void SWB0_6(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB0<6>(c_in0, c_in1, c_out0, c_out1);
}

void SWB1_6(tapa::istream<Cnoc_pkt>& c_in0, tapa::istream<Cnoc_pkt>& c_in1,
	tapa::ostream<Cnoc_pkt>& c_out0, tapa::ostream<Cnoc_pkt>& c_out1) {
	SWB1<6>(c_in0, c_in1, c_out0, c_out1);
}

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


	tapa::stream<Cnoc_pkt, 94> s_0_0("s_0_0");
	tapa::stream<Cnoc_pkt, 2> s_1_0("s_1_0");
	tapa::stream<Cnoc_pkt, 2> s_2_0("s_2_0");
	tapa::stream<Cnoc_pkt, 10> s_3_0("s_3_0");
	tapa::stream<Cnoc_pkt, 10> s_4_0("s_4_0");
	tapa::stream<Cnoc_pkt, 2> s_5_0("s_5_0");
	tapa::stream<Cnoc_pkt, 2> s_6_0("s_6_0");
	tapa::stream<Cnoc_pkt, 20> s_7_0("s_7_0");
	tapa::stream<Cnoc_pkt, 20> s_8_0("s_8_0");
	tapa::stream<Cnoc_pkt, 2> s_9_0("s_9_0");
	tapa::stream<Cnoc_pkt, 2> s_10_0("s_10_0");
	tapa::stream<Cnoc_pkt, 10> s_11_0("s_11_0");
	tapa::stream<Cnoc_pkt, 10> s_12_0("s_12_0");
	tapa::stream<Cnoc_pkt, 2> s_13_0("s_13_0");
	tapa::stream<Cnoc_pkt, 2> s_14_0("s_14_0");
	tapa::stream<Cnoc_pkt, 30> s_15_0("s_15_0");
	tapa::stream<Cnoc_pkt, 30> s_16_0("s_16_0");
	tapa::stream<Cnoc_pkt, 2> s_17_0("s_17_0");
	tapa::stream<Cnoc_pkt, 2> s_18_0("s_18_0");
	tapa::stream<Cnoc_pkt, 10> s_19_0("s_19_0");
	tapa::stream<Cnoc_pkt, 10> s_20_0("s_20_0");
	tapa::stream<Cnoc_pkt, 2> s_21_0("s_21_0");
	tapa::stream<Cnoc_pkt, 2> s_22_0("s_22_0");
	tapa::stream<Cnoc_pkt, 20> s_23_0("s_23_0");
	tapa::stream<Cnoc_pkt, 20> s_24_0("s_24_0");
	tapa::stream<Cnoc_pkt, 2> s_25_0("s_25_0");
	tapa::stream<Cnoc_pkt, 2> s_26_0("s_26_0");
	tapa::stream<Cnoc_pkt, 10> s_27_0("s_27_0");
	tapa::stream<Cnoc_pkt, 10> s_28_0("s_28_0");
	tapa::stream<Cnoc_pkt, 2> s_29_0("s_29_0");
	tapa::stream<Cnoc_pkt, 2> s_30_0("s_30_0");
	tapa::stream<Cnoc_pkt, 40> s_31_0("s_31_0");
	tapa::stream<Cnoc_pkt, 40> s_32_0("s_32_0");
	tapa::stream<Cnoc_pkt, 2> s_33_0("s_33_0");
	tapa::stream<Cnoc_pkt, 2> s_34_0("s_34_0");
	tapa::stream<Cnoc_pkt, 10> s_35_0("s_35_0");
	tapa::stream<Cnoc_pkt, 10> s_36_0("s_36_0");
	tapa::stream<Cnoc_pkt, 2> s_37_0("s_37_0");
	tapa::stream<Cnoc_pkt, 2> s_38_0("s_38_0");
	tapa::stream<Cnoc_pkt, 20> s_39_0("s_39_0");
	tapa::stream<Cnoc_pkt, 20> s_40_0("s_40_0");
	tapa::stream<Cnoc_pkt, 2> s_41_0("s_41_0");
	tapa::stream<Cnoc_pkt, 2> s_42_0("s_42_0");
	tapa::stream<Cnoc_pkt, 10> s_43_0("s_43_0");
	tapa::stream<Cnoc_pkt, 10> s_44_0("s_44_0");
	tapa::stream<Cnoc_pkt, 2> s_45_0("s_45_0");
	tapa::stream<Cnoc_pkt, 2> s_46_0("s_46_0");
	tapa::stream<Cnoc_pkt, 30> s_47_0("s_47_0");
	tapa::stream<Cnoc_pkt, 30> s_48_0("s_48_0");
	tapa::stream<Cnoc_pkt, 2> s_49_0("s_49_0");
	tapa::stream<Cnoc_pkt, 2> s_50_0("s_50_0");
	tapa::stream<Cnoc_pkt, 10> s_51_0("s_51_0");
	tapa::stream<Cnoc_pkt, 10> s_52_0("s_52_0");
	tapa::stream<Cnoc_pkt, 2> s_53_0("s_53_0");
	tapa::stream<Cnoc_pkt, 2> s_54_0("s_54_0");
	tapa::stream<Cnoc_pkt, 20> s_55_0("s_55_0");
	tapa::stream<Cnoc_pkt, 20> s_56_0("s_56_0");
	tapa::stream<Cnoc_pkt, 2> s_57_0("s_57_0");
	tapa::stream<Cnoc_pkt, 2> s_58_0("s_58_0");
	tapa::stream<Cnoc_pkt, 10> s_59_0("s_59_0");
	tapa::stream<Cnoc_pkt, 10> s_60_0("s_60_0");
	tapa::stream<Cnoc_pkt, 2> s_61_0("s_61_0");
	tapa::stream<Cnoc_pkt, 2> s_62_0("s_62_0");
	tapa::stream<Cnoc_pkt, 50> s_63_0("s_63_0");
	tapa::stream<Cnoc_pkt, 50> s_64_0("s_64_0");
	tapa::stream<Cnoc_pkt, 2> s_65_0("s_65_0");
	tapa::stream<Cnoc_pkt, 2> s_66_0("s_66_0");
	tapa::stream<Cnoc_pkt, 10> s_67_0("s_67_0");
	tapa::stream<Cnoc_pkt, 10> s_68_0("s_68_0");
	tapa::stream<Cnoc_pkt, 2> s_69_0("s_69_0");
	tapa::stream<Cnoc_pkt, 2> s_70_0("s_70_0");
	tapa::stream<Cnoc_pkt, 20> s_71_0("s_71_0");
	tapa::stream<Cnoc_pkt, 20> s_72_0("s_72_0");
	tapa::stream<Cnoc_pkt, 2> s_73_0("s_73_0");
	tapa::stream<Cnoc_pkt, 2> s_74_0("s_74_0");
	tapa::stream<Cnoc_pkt, 10> s_75_0("s_75_0");
	tapa::stream<Cnoc_pkt, 10> s_76_0("s_76_0");
	tapa::stream<Cnoc_pkt, 2> s_77_0("s_77_0");
	tapa::stream<Cnoc_pkt, 2> s_78_0("s_78_0");
	tapa::stream<Cnoc_pkt, 30> s_79_0("s_79_0");
	tapa::stream<Cnoc_pkt, 30> s_80_0("s_80_0");
	tapa::stream<Cnoc_pkt, 2> s_81_0("s_81_0");
	tapa::stream<Cnoc_pkt, 2> s_82_0("s_82_0");
	tapa::stream<Cnoc_pkt, 10> s_83_0("s_83_0");
	tapa::stream<Cnoc_pkt, 10> s_84_0("s_84_0");
	tapa::stream<Cnoc_pkt, 2> s_85_0("s_85_0");
	tapa::stream<Cnoc_pkt, 2> s_86_0("s_86_0");
	tapa::stream<Cnoc_pkt, 20> s_87_0("s_87_0");
	tapa::stream<Cnoc_pkt, 20> s_88_0("s_88_0");
	tapa::stream<Cnoc_pkt, 2> s_89_0("s_89_0");
	tapa::stream<Cnoc_pkt, 2> s_90_0("s_90_0");
	tapa::stream<Cnoc_pkt, 10> s_91_0("s_91_0");
	tapa::stream<Cnoc_pkt, 10> s_92_0("s_92_0");
	tapa::stream<Cnoc_pkt, 2> s_93_0("s_93_0");
	tapa::stream<Cnoc_pkt, 2> s_94_0("s_94_0");
	tapa::stream<Cnoc_pkt, 40> s_95_0("s_95_0");
	tapa::stream<Cnoc_pkt, 40> s_96_0("s_96_0");
	tapa::stream<Cnoc_pkt, 2> s_97_0("s_97_0");
	tapa::stream<Cnoc_pkt, 2> s_98_0("s_98_0");
	tapa::stream<Cnoc_pkt, 10> s_99_0("s_99_0");
	tapa::stream<Cnoc_pkt, 10> s_100_0("s_100_0");
	tapa::stream<Cnoc_pkt, 2> s_101_0("s_101_0");
	tapa::stream<Cnoc_pkt, 2> s_102_0("s_102_0");
	tapa::stream<Cnoc_pkt, 20> s_103_0("s_103_0");
	tapa::stream<Cnoc_pkt, 20> s_104_0("s_104_0");
	tapa::stream<Cnoc_pkt, 2> s_105_0("s_105_0");
	tapa::stream<Cnoc_pkt, 2> s_106_0("s_106_0");
	tapa::stream<Cnoc_pkt, 10> s_107_0("s_107_0");
	tapa::stream<Cnoc_pkt, 10> s_108_0("s_108_0");
	tapa::stream<Cnoc_pkt, 2> s_109_0("s_109_0");
	tapa::stream<Cnoc_pkt, 2> s_110_0("s_110_0");
	tapa::stream<Cnoc_pkt, 30> s_111_0("s_111_0");
	tapa::stream<Cnoc_pkt, 30> s_112_0("s_112_0");
	tapa::stream<Cnoc_pkt, 2> s_113_0("s_113_0");
	tapa::stream<Cnoc_pkt, 2> s_114_0("s_114_0");
	tapa::stream<Cnoc_pkt, 10> s_115_0("s_115_0");
	tapa::stream<Cnoc_pkt, 10> s_116_0("s_116_0");
	tapa::stream<Cnoc_pkt, 2> s_117_0("s_117_0");
	tapa::stream<Cnoc_pkt, 2> s_118_0("s_118_0");
	tapa::stream<Cnoc_pkt, 20> s_119_0("s_119_0");
	tapa::stream<Cnoc_pkt, 20> s_120_0("s_120_0");
	tapa::stream<Cnoc_pkt, 2> s_121_0("s_121_0");
	tapa::stream<Cnoc_pkt, 2> s_122_0("s_122_0");
	tapa::stream<Cnoc_pkt, 10> s_123_0("s_123_0");
	tapa::stream<Cnoc_pkt, 10> s_124_0("s_124_0");
	tapa::stream<Cnoc_pkt, 2> s_125_0("s_125_0");
	tapa::stream<Cnoc_pkt, 2> s_126_0("s_126_0");
	tapa::stream<Cnoc_pkt, 60> s_127_0("s_127_0");
	tapa::stream<Cnoc_pkt, 60> s_128_0("s_128_0");
	tapa::stream<Cnoc_pkt, 2> s_129_0("s_129_0");
	tapa::stream<Cnoc_pkt, 2> s_130_0("s_130_0");
	tapa::stream<Cnoc_pkt, 10> s_131_0("s_131_0");
	tapa::stream<Cnoc_pkt, 10> s_132_0("s_132_0");
	tapa::stream<Cnoc_pkt, 2> s_133_0("s_133_0");
	tapa::stream<Cnoc_pkt, 2> s_134_0("s_134_0");
	tapa::stream<Cnoc_pkt, 20> s_135_0("s_135_0");
	tapa::stream<Cnoc_pkt, 20> s_136_0("s_136_0");
	tapa::stream<Cnoc_pkt, 2> s_137_0("s_137_0");
	tapa::stream<Cnoc_pkt, 2> s_138_0("s_138_0");
	tapa::stream<Cnoc_pkt, 10> s_139_0("s_139_0");
	tapa::stream<Cnoc_pkt, 10> s_140_0("s_140_0");
	tapa::stream<Cnoc_pkt, 2> s_141_0("s_141_0");
	tapa::stream<Cnoc_pkt, 2> s_142_0("s_142_0");
	tapa::stream<Cnoc_pkt, 30> s_143_0("s_143_0");
	tapa::stream<Cnoc_pkt, 30> s_144_0("s_144_0");
	tapa::stream<Cnoc_pkt, 2> s_145_0("s_145_0");
	tapa::stream<Cnoc_pkt, 2> s_146_0("s_146_0");
	tapa::stream<Cnoc_pkt, 10> s_147_0("s_147_0");
	tapa::stream<Cnoc_pkt, 10> s_148_0("s_148_0");
	tapa::stream<Cnoc_pkt, 2> s_149_0("s_149_0");
	tapa::stream<Cnoc_pkt, 2> s_150_0("s_150_0");
	tapa::stream<Cnoc_pkt, 20> s_151_0("s_151_0");
	tapa::stream<Cnoc_pkt, 20> s_152_0("s_152_0");
	tapa::stream<Cnoc_pkt, 2> s_153_0("s_153_0");
	tapa::stream<Cnoc_pkt, 2> s_154_0("s_154_0");
	tapa::stream<Cnoc_pkt, 10> s_155_0("s_155_0");
	tapa::stream<Cnoc_pkt, 10> s_156_0("s_156_0");
	tapa::stream<Cnoc_pkt, 2> s_157_0("s_157_0");
	tapa::stream<Cnoc_pkt, 2> s_158_0("s_158_0");
	tapa::stream<Cnoc_pkt, 94> s_159_0("s_159_0");
	tapa::stream<Cnoc_pkt, 90> s_1_1("s_1_1");
	tapa::stream<Cnoc_pkt, 8> s_2_1("s_2_1");
	tapa::stream<Cnoc_pkt, 8> s_5_1("s_5_1");
	tapa::stream<Cnoc_pkt, 90> s_6_1("s_6_1");
	tapa::stream<Cnoc_pkt, 90> s_9_1("s_9_1");
	tapa::stream<Cnoc_pkt, 8> s_10_1("s_10_1");
	tapa::stream<Cnoc_pkt, 8> s_13_1("s_13_1");
	tapa::stream<Cnoc_pkt, 90> s_14_1("s_14_1");
	tapa::stream<Cnoc_pkt, 90> s_17_1("s_17_1");
	tapa::stream<Cnoc_pkt, 8> s_18_1("s_18_1");
	tapa::stream<Cnoc_pkt, 8> s_21_1("s_21_1");
	tapa::stream<Cnoc_pkt, 90> s_22_1("s_22_1");
	tapa::stream<Cnoc_pkt, 90> s_25_1("s_25_1");
	tapa::stream<Cnoc_pkt, 8> s_26_1("s_26_1");
	tapa::stream<Cnoc_pkt, 8> s_29_1("s_29_1");
	tapa::stream<Cnoc_pkt, 90> s_30_1("s_30_1");
	tapa::stream<Cnoc_pkt, 90> s_33_1("s_33_1");
	tapa::stream<Cnoc_pkt, 8> s_34_1("s_34_1");
	tapa::stream<Cnoc_pkt, 8> s_37_1("s_37_1");
	tapa::stream<Cnoc_pkt, 90> s_38_1("s_38_1");
	tapa::stream<Cnoc_pkt, 90> s_41_1("s_41_1");
	tapa::stream<Cnoc_pkt, 8> s_42_1("s_42_1");
	tapa::stream<Cnoc_pkt, 8> s_45_1("s_45_1");
	tapa::stream<Cnoc_pkt, 90> s_46_1("s_46_1");
	tapa::stream<Cnoc_pkt, 90> s_49_1("s_49_1");
	tapa::stream<Cnoc_pkt, 8> s_50_1("s_50_1");
	tapa::stream<Cnoc_pkt, 8> s_53_1("s_53_1");
	tapa::stream<Cnoc_pkt, 90> s_54_1("s_54_1");
	tapa::stream<Cnoc_pkt, 90> s_57_1("s_57_1");
	tapa::stream<Cnoc_pkt, 8> s_58_1("s_58_1");
	tapa::stream<Cnoc_pkt, 8> s_61_1("s_61_1");
	tapa::stream<Cnoc_pkt, 90> s_62_1("s_62_1");
	tapa::stream<Cnoc_pkt, 90> s_65_1("s_65_1");
	tapa::stream<Cnoc_pkt, 8> s_66_1("s_66_1");
	tapa::stream<Cnoc_pkt, 8> s_69_1("s_69_1");
	tapa::stream<Cnoc_pkt, 90> s_70_1("s_70_1");
	tapa::stream<Cnoc_pkt, 90> s_73_1("s_73_1");
	tapa::stream<Cnoc_pkt, 8> s_74_1("s_74_1");
	tapa::stream<Cnoc_pkt, 8> s_77_1("s_77_1");
	tapa::stream<Cnoc_pkt, 90> s_78_1("s_78_1");
	tapa::stream<Cnoc_pkt, 90> s_81_1("s_81_1");
	tapa::stream<Cnoc_pkt, 8> s_82_1("s_82_1");
	tapa::stream<Cnoc_pkt, 8> s_85_1("s_85_1");
	tapa::stream<Cnoc_pkt, 90> s_86_1("s_86_1");
	tapa::stream<Cnoc_pkt, 90> s_89_1("s_89_1");
	tapa::stream<Cnoc_pkt, 8> s_90_1("s_90_1");
	tapa::stream<Cnoc_pkt, 8> s_93_1("s_93_1");
	tapa::stream<Cnoc_pkt, 90> s_94_1("s_94_1");
	tapa::stream<Cnoc_pkt, 90> s_97_1("s_97_1");
	tapa::stream<Cnoc_pkt, 8> s_98_1("s_98_1");
	tapa::stream<Cnoc_pkt, 8> s_101_1("s_101_1");
	tapa::stream<Cnoc_pkt, 90> s_102_1("s_102_1");
	tapa::stream<Cnoc_pkt, 90> s_105_1("s_105_1");
	tapa::stream<Cnoc_pkt, 8> s_106_1("s_106_1");
	tapa::stream<Cnoc_pkt, 8> s_109_1("s_109_1");
	tapa::stream<Cnoc_pkt, 90> s_110_1("s_110_1");
	tapa::stream<Cnoc_pkt, 90> s_113_1("s_113_1");
	tapa::stream<Cnoc_pkt, 8> s_114_1("s_114_1");
	tapa::stream<Cnoc_pkt, 8> s_117_1("s_117_1");
	tapa::stream<Cnoc_pkt, 90> s_118_1("s_118_1");
	tapa::stream<Cnoc_pkt, 90> s_121_1("s_121_1");
	tapa::stream<Cnoc_pkt, 8> s_122_1("s_122_1");
	tapa::stream<Cnoc_pkt, 8> s_125_1("s_125_1");
	tapa::stream<Cnoc_pkt, 90> s_126_1("s_126_1");
	tapa::stream<Cnoc_pkt, 90> s_129_1("s_129_1");
	tapa::stream<Cnoc_pkt, 8> s_130_1("s_130_1");
	tapa::stream<Cnoc_pkt, 8> s_133_1("s_133_1");
	tapa::stream<Cnoc_pkt, 90> s_134_1("s_134_1");
	tapa::stream<Cnoc_pkt, 90> s_137_1("s_137_1");
	tapa::stream<Cnoc_pkt, 8> s_138_1("s_138_1");
	tapa::stream<Cnoc_pkt, 8> s_141_1("s_141_1");
	tapa::stream<Cnoc_pkt, 90> s_142_1("s_142_1");
	tapa::stream<Cnoc_pkt, 90> s_145_1("s_145_1");
	tapa::stream<Cnoc_pkt, 8> s_146_1("s_146_1");
	tapa::stream<Cnoc_pkt, 8> s_149_1("s_149_1");
	tapa::stream<Cnoc_pkt, 90> s_150_1("s_150_1");
	tapa::stream<Cnoc_pkt, 90> s_153_1("s_153_1");
	tapa::stream<Cnoc_pkt, 8> s_154_1("s_154_1");
	tapa::stream<Cnoc_pkt, 8> s_157_1("s_157_1");
	tapa::stream<Cnoc_pkt, 90> s_158_1("s_158_1");
	tapa::stream<Cnoc_pkt, 80> s_2_2("s_2_2");
	tapa::stream<Cnoc_pkt, 2> s_3_1("s_3_1");
	tapa::stream<Cnoc_pkt, 2> s_4_1("s_4_1");
	tapa::stream<Cnoc_pkt, 80> s_5_2("s_5_2");
	tapa::stream<Cnoc_pkt, 76> s_3_2("s_3_2");
	tapa::stream<Cnoc_pkt, 8> s_4_2("s_4_2");
	tapa::stream<Cnoc_pkt, 80> s_10_2("s_10_2");
	tapa::stream<Cnoc_pkt, 2> s_11_1("s_11_1");
	tapa::stream<Cnoc_pkt, 2> s_12_1("s_12_1");
	tapa::stream<Cnoc_pkt, 80> s_13_2("s_13_2");
	tapa::stream<Cnoc_pkt, 8> s_11_2("s_11_2");
	tapa::stream<Cnoc_pkt, 76> s_12_2("s_12_2");
	tapa::stream<Cnoc_pkt, 80> s_18_2("s_18_2");
	tapa::stream<Cnoc_pkt, 2> s_19_1("s_19_1");
	tapa::stream<Cnoc_pkt, 2> s_20_1("s_20_1");
	tapa::stream<Cnoc_pkt, 80> s_21_2("s_21_2");
	tapa::stream<Cnoc_pkt, 76> s_19_2("s_19_2");
	tapa::stream<Cnoc_pkt, 8> s_20_2("s_20_2");
	tapa::stream<Cnoc_pkt, 80> s_26_2("s_26_2");
	tapa::stream<Cnoc_pkt, 2> s_27_1("s_27_1");
	tapa::stream<Cnoc_pkt, 2> s_28_1("s_28_1");
	tapa::stream<Cnoc_pkt, 80> s_29_2("s_29_2");
	tapa::stream<Cnoc_pkt, 8> s_27_2("s_27_2");
	tapa::stream<Cnoc_pkt, 76> s_28_2("s_28_2");
	tapa::stream<Cnoc_pkt, 80> s_34_2("s_34_2");
	tapa::stream<Cnoc_pkt, 2> s_35_1("s_35_1");
	tapa::stream<Cnoc_pkt, 2> s_36_1("s_36_1");
	tapa::stream<Cnoc_pkt, 80> s_37_2("s_37_2");
	tapa::stream<Cnoc_pkt, 76> s_35_2("s_35_2");
	tapa::stream<Cnoc_pkt, 8> s_36_2("s_36_2");
	tapa::stream<Cnoc_pkt, 80> s_42_2("s_42_2");
	tapa::stream<Cnoc_pkt, 2> s_43_1("s_43_1");
	tapa::stream<Cnoc_pkt, 2> s_44_1("s_44_1");
	tapa::stream<Cnoc_pkt, 80> s_45_2("s_45_2");
	tapa::stream<Cnoc_pkt, 8> s_43_2("s_43_2");
	tapa::stream<Cnoc_pkt, 76> s_44_2("s_44_2");
	tapa::stream<Cnoc_pkt, 80> s_50_2("s_50_2");
	tapa::stream<Cnoc_pkt, 2> s_51_1("s_51_1");
	tapa::stream<Cnoc_pkt, 2> s_52_1("s_52_1");
	tapa::stream<Cnoc_pkt, 80> s_53_2("s_53_2");
	tapa::stream<Cnoc_pkt, 76> s_51_2("s_51_2");
	tapa::stream<Cnoc_pkt, 8> s_52_2("s_52_2");
	tapa::stream<Cnoc_pkt, 80> s_58_2("s_58_2");
	tapa::stream<Cnoc_pkt, 2> s_59_1("s_59_1");
	tapa::stream<Cnoc_pkt, 2> s_60_1("s_60_1");
	tapa::stream<Cnoc_pkt, 80> s_61_2("s_61_2");
	tapa::stream<Cnoc_pkt, 8> s_59_2("s_59_2");
	tapa::stream<Cnoc_pkt, 76> s_60_2("s_60_2");
	tapa::stream<Cnoc_pkt, 80> s_66_2("s_66_2");
	tapa::stream<Cnoc_pkt, 2> s_67_1("s_67_1");
	tapa::stream<Cnoc_pkt, 2> s_68_1("s_68_1");
	tapa::stream<Cnoc_pkt, 80> s_69_2("s_69_2");
	tapa::stream<Cnoc_pkt, 76> s_67_2("s_67_2");
	tapa::stream<Cnoc_pkt, 8> s_68_2("s_68_2");
	tapa::stream<Cnoc_pkt, 80> s_74_2("s_74_2");
	tapa::stream<Cnoc_pkt, 2> s_75_1("s_75_1");
	tapa::stream<Cnoc_pkt, 2> s_76_1("s_76_1");
	tapa::stream<Cnoc_pkt, 80> s_77_2("s_77_2");
	tapa::stream<Cnoc_pkt, 8> s_75_2("s_75_2");
	tapa::stream<Cnoc_pkt, 76> s_76_2("s_76_2");
	tapa::stream<Cnoc_pkt, 80> s_82_2("s_82_2");
	tapa::stream<Cnoc_pkt, 2> s_83_1("s_83_1");
	tapa::stream<Cnoc_pkt, 2> s_84_1("s_84_1");
	tapa::stream<Cnoc_pkt, 80> s_85_2("s_85_2");
	tapa::stream<Cnoc_pkt, 76> s_83_2("s_83_2");
	tapa::stream<Cnoc_pkt, 8> s_84_2("s_84_2");
	tapa::stream<Cnoc_pkt, 80> s_90_2("s_90_2");
	tapa::stream<Cnoc_pkt, 2> s_91_1("s_91_1");
	tapa::stream<Cnoc_pkt, 2> s_92_1("s_92_1");
	tapa::stream<Cnoc_pkt, 80> s_93_2("s_93_2");
	tapa::stream<Cnoc_pkt, 8> s_91_2("s_91_2");
	tapa::stream<Cnoc_pkt, 76> s_92_2("s_92_2");
	tapa::stream<Cnoc_pkt, 80> s_98_2("s_98_2");
	tapa::stream<Cnoc_pkt, 2> s_99_1("s_99_1");
	tapa::stream<Cnoc_pkt, 2> s_100_1("s_100_1");
	tapa::stream<Cnoc_pkt, 80> s_101_2("s_101_2");
	tapa::stream<Cnoc_pkt, 76> s_99_2("s_99_2");
	tapa::stream<Cnoc_pkt, 8> s_100_2("s_100_2");
	tapa::stream<Cnoc_pkt, 80> s_106_2("s_106_2");
	tapa::stream<Cnoc_pkt, 2> s_107_1("s_107_1");
	tapa::stream<Cnoc_pkt, 2> s_108_1("s_108_1");
	tapa::stream<Cnoc_pkt, 80> s_109_2("s_109_2");
	tapa::stream<Cnoc_pkt, 8> s_107_2("s_107_2");
	tapa::stream<Cnoc_pkt, 76> s_108_2("s_108_2");
	tapa::stream<Cnoc_pkt, 80> s_114_2("s_114_2");
	tapa::stream<Cnoc_pkt, 2> s_115_1("s_115_1");
	tapa::stream<Cnoc_pkt, 2> s_116_1("s_116_1");
	tapa::stream<Cnoc_pkt, 80> s_117_2("s_117_2");
	tapa::stream<Cnoc_pkt, 76> s_115_2("s_115_2");
	tapa::stream<Cnoc_pkt, 8> s_116_2("s_116_2");
	tapa::stream<Cnoc_pkt, 80> s_122_2("s_122_2");
	tapa::stream<Cnoc_pkt, 2> s_123_1("s_123_1");
	tapa::stream<Cnoc_pkt, 2> s_124_1("s_124_1");
	tapa::stream<Cnoc_pkt, 80> s_125_2("s_125_2");
	tapa::stream<Cnoc_pkt, 8> s_123_2("s_123_2");
	tapa::stream<Cnoc_pkt, 76> s_124_2("s_124_2");
	tapa::stream<Cnoc_pkt, 80> s_130_2("s_130_2");
	tapa::stream<Cnoc_pkt, 2> s_131_1("s_131_1");
	tapa::stream<Cnoc_pkt, 2> s_132_1("s_132_1");
	tapa::stream<Cnoc_pkt, 80> s_133_2("s_133_2");
	tapa::stream<Cnoc_pkt, 76> s_131_2("s_131_2");
	tapa::stream<Cnoc_pkt, 8> s_132_2("s_132_2");
	tapa::stream<Cnoc_pkt, 80> s_138_2("s_138_2");
	tapa::stream<Cnoc_pkt, 2> s_139_1("s_139_1");
	tapa::stream<Cnoc_pkt, 2> s_140_1("s_140_1");
	tapa::stream<Cnoc_pkt, 80> s_141_2("s_141_2");
	tapa::stream<Cnoc_pkt, 8> s_139_2("s_139_2");
	tapa::stream<Cnoc_pkt, 76> s_140_2("s_140_2");
	tapa::stream<Cnoc_pkt, 80> s_146_2("s_146_2");
	tapa::stream<Cnoc_pkt, 2> s_147_1("s_147_1");
	tapa::stream<Cnoc_pkt, 2> s_148_1("s_148_1");
	tapa::stream<Cnoc_pkt, 80> s_149_2("s_149_2");
	tapa::stream<Cnoc_pkt, 76> s_147_2("s_147_2");
	tapa::stream<Cnoc_pkt, 8> s_148_2("s_148_2");
	tapa::stream<Cnoc_pkt, 80> s_154_2("s_154_2");
	tapa::stream<Cnoc_pkt, 2> s_155_1("s_155_1");
	tapa::stream<Cnoc_pkt, 2> s_156_1("s_156_1");
	tapa::stream<Cnoc_pkt, 80> s_157_2("s_157_2");
	tapa::stream<Cnoc_pkt, 8> s_155_2("s_155_2");
	tapa::stream<Cnoc_pkt, 76> s_156_2("s_156_2");
	tapa::stream<Cnoc_pkt, 66> s_4_3("s_4_3");
	tapa::stream<Cnoc_pkt, 2> s_7_1("s_7_1");
	tapa::stream<Cnoc_pkt, 2> s_8_1("s_8_1");
	tapa::stream<Cnoc_pkt, 66> s_11_3("s_11_3");
	tapa::stream<Cnoc_pkt, 62> s_7_2("s_7_2");
	tapa::stream<Cnoc_pkt, 8> s_8_2("s_8_2");
	tapa::stream<Cnoc_pkt, 66> s_20_3("s_20_3");
	tapa::stream<Cnoc_pkt, 2> s_23_1("s_23_1");
	tapa::stream<Cnoc_pkt, 2> s_24_1("s_24_1");
	tapa::stream<Cnoc_pkt, 66> s_27_3("s_27_3");
	tapa::stream<Cnoc_pkt, 8> s_23_2("s_23_2");
	tapa::stream<Cnoc_pkt, 62> s_24_2("s_24_2");
	tapa::stream<Cnoc_pkt, 66> s_36_3("s_36_3");
	tapa::stream<Cnoc_pkt, 2> s_39_1("s_39_1");
	tapa::stream<Cnoc_pkt, 2> s_40_1("s_40_1");
	tapa::stream<Cnoc_pkt, 66> s_43_3("s_43_3");
	tapa::stream<Cnoc_pkt, 62> s_39_2("s_39_2");
	tapa::stream<Cnoc_pkt, 8> s_40_2("s_40_2");
	tapa::stream<Cnoc_pkt, 66> s_52_3("s_52_3");
	tapa::stream<Cnoc_pkt, 2> s_55_1("s_55_1");
	tapa::stream<Cnoc_pkt, 2> s_56_1("s_56_1");
	tapa::stream<Cnoc_pkt, 66> s_59_3("s_59_3");
	tapa::stream<Cnoc_pkt, 8> s_55_2("s_55_2");
	tapa::stream<Cnoc_pkt, 62> s_56_2("s_56_2");
	tapa::stream<Cnoc_pkt, 66> s_68_3("s_68_3");
	tapa::stream<Cnoc_pkt, 2> s_71_1("s_71_1");
	tapa::stream<Cnoc_pkt, 2> s_72_1("s_72_1");
	tapa::stream<Cnoc_pkt, 66> s_75_3("s_75_3");
	tapa::stream<Cnoc_pkt, 62> s_71_2("s_71_2");
	tapa::stream<Cnoc_pkt, 8> s_72_2("s_72_2");
	tapa::stream<Cnoc_pkt, 66> s_84_3("s_84_3");
	tapa::stream<Cnoc_pkt, 2> s_87_1("s_87_1");
	tapa::stream<Cnoc_pkt, 2> s_88_1("s_88_1");
	tapa::stream<Cnoc_pkt, 66> s_91_3("s_91_3");
	tapa::stream<Cnoc_pkt, 8> s_87_2("s_87_2");
	tapa::stream<Cnoc_pkt, 62> s_88_2("s_88_2");
	tapa::stream<Cnoc_pkt, 66> s_100_3("s_100_3");
	tapa::stream<Cnoc_pkt, 2> s_103_1("s_103_1");
	tapa::stream<Cnoc_pkt, 2> s_104_1("s_104_1");
	tapa::stream<Cnoc_pkt, 66> s_107_3("s_107_3");
	tapa::stream<Cnoc_pkt, 62> s_103_2("s_103_2");
	tapa::stream<Cnoc_pkt, 8> s_104_2("s_104_2");
	tapa::stream<Cnoc_pkt, 66> s_116_3("s_116_3");
	tapa::stream<Cnoc_pkt, 2> s_119_1("s_119_1");
	tapa::stream<Cnoc_pkt, 2> s_120_1("s_120_1");
	tapa::stream<Cnoc_pkt, 66> s_123_3("s_123_3");
	tapa::stream<Cnoc_pkt, 8> s_119_2("s_119_2");
	tapa::stream<Cnoc_pkt, 62> s_120_2("s_120_2");
	tapa::stream<Cnoc_pkt, 66> s_132_3("s_132_3");
	tapa::stream<Cnoc_pkt, 2> s_135_1("s_135_1");
	tapa::stream<Cnoc_pkt, 2> s_136_1("s_136_1");
	tapa::stream<Cnoc_pkt, 66> s_139_3("s_139_3");
	tapa::stream<Cnoc_pkt, 62> s_135_2("s_135_2");
	tapa::stream<Cnoc_pkt, 8> s_136_2("s_136_2");
	tapa::stream<Cnoc_pkt, 66> s_148_3("s_148_3");
	tapa::stream<Cnoc_pkt, 2> s_151_1("s_151_1");
	tapa::stream<Cnoc_pkt, 2> s_152_1("s_152_1");
	tapa::stream<Cnoc_pkt, 66> s_155_3("s_155_3");
	tapa::stream<Cnoc_pkt, 8> s_151_2("s_151_2");
	tapa::stream<Cnoc_pkt, 62> s_152_2("s_152_2");
	tapa::stream<Cnoc_pkt, 52> s_8_3("s_8_3");
	tapa::stream<Cnoc_pkt, 2> s_15_1("s_15_1");
	tapa::stream<Cnoc_pkt, 2> s_16_1("s_16_1");
	tapa::stream<Cnoc_pkt, 52> s_23_3("s_23_3");
	tapa::stream<Cnoc_pkt, 48> s_15_2("s_15_2");
	tapa::stream<Cnoc_pkt, 8> s_16_2("s_16_2");
	tapa::stream<Cnoc_pkt, 52> s_40_3("s_40_3");
	tapa::stream<Cnoc_pkt, 2> s_47_1("s_47_1");
	tapa::stream<Cnoc_pkt, 2> s_48_1("s_48_1");
	tapa::stream<Cnoc_pkt, 52> s_55_3("s_55_3");
	tapa::stream<Cnoc_pkt, 8> s_47_2("s_47_2");
	tapa::stream<Cnoc_pkt, 48> s_48_2("s_48_2");
	tapa::stream<Cnoc_pkt, 52> s_72_3("s_72_3");
	tapa::stream<Cnoc_pkt, 2> s_79_1("s_79_1");
	tapa::stream<Cnoc_pkt, 2> s_80_1("s_80_1");
	tapa::stream<Cnoc_pkt, 52> s_87_3("s_87_3");
	tapa::stream<Cnoc_pkt, 48> s_79_2("s_79_2");
	tapa::stream<Cnoc_pkt, 8> s_80_2("s_80_2");
	tapa::stream<Cnoc_pkt, 52> s_104_3("s_104_3");
	tapa::stream<Cnoc_pkt, 2> s_111_1("s_111_1");
	tapa::stream<Cnoc_pkt, 2> s_112_1("s_112_1");
	tapa::stream<Cnoc_pkt, 52> s_119_3("s_119_3");
	tapa::stream<Cnoc_pkt, 8> s_111_2("s_111_2");
	tapa::stream<Cnoc_pkt, 48> s_112_2("s_112_2");
	tapa::stream<Cnoc_pkt, 52> s_136_3("s_136_3");
	tapa::stream<Cnoc_pkt, 2> s_143_1("s_143_1");
	tapa::stream<Cnoc_pkt, 2> s_144_1("s_144_1");
	tapa::stream<Cnoc_pkt, 52> s_151_3("s_151_3");
	tapa::stream<Cnoc_pkt, 48> s_143_2("s_143_2");
	tapa::stream<Cnoc_pkt, 28> s_144_2("s_144_2");
	tapa::stream<Cnoc_pkt, 38> s_16_3("s_16_3");
	tapa::stream<Cnoc_pkt, 2> s_31_1("s_31_1");
	tapa::stream<Cnoc_pkt, 2> s_32_1("s_32_1");
	tapa::stream<Cnoc_pkt, 38> s_47_3("s_47_3");
	tapa::stream<Cnoc_pkt, 34> s_31_2("s_31_2");
	tapa::stream<Cnoc_pkt, 8> s_32_2("s_32_2");
	tapa::stream<Cnoc_pkt, 38> s_80_3("s_80_3");
	tapa::stream<Cnoc_pkt, 2> s_95_1("s_95_1");
	tapa::stream<Cnoc_pkt, 2> s_96_1("s_96_1");
	tapa::stream<Cnoc_pkt, 38> s_111_3("s_111_3");
	tapa::stream<Cnoc_pkt, 8> s_95_2("s_95_2");
	tapa::stream<Cnoc_pkt, 34> s_96_2("s_96_2");
	tapa::stream<Cnoc_pkt, 24> s_32_3("s_32_3");
	tapa::stream<Cnoc_pkt, 2> s_63_1("s_63_1");
	tapa::stream<Cnoc_pkt, 2> s_64_1("s_64_1");
	tapa::stream<Cnoc_pkt, 24> s_95_3("s_95_3");
	tapa::stream<Cnoc_pkt, 20> s_63_2("s_63_2");
	tapa::stream<Cnoc_pkt, 8> s_64_2("s_64_2");
	tapa::stream<Cnoc_pkt, 10> s_64_3("s_64_3");
	tapa::stream<Cnoc_pkt, 2> s_127_1("s_127_1");
	tapa::stream<Cnoc_pkt, 2> s_128_1("s_128_1");
	tapa::stream<Cnoc_pkt, 10> s_144_3("s_144_3");
	tapa::stream<Cnoc_pkt, 8> s_127_2("s_127_2");
	tapa::stream<Cnoc_pkt, 8> s_128_2("s_128_2");
	tapa::stream<Cnoc_pkt, 2> s_64_4("s_64_4");
	tapa::stream<Cnoc_pkt, 24> s_127_3("s_127_3");
	tapa::stream<Cnoc_pkt, 24> s_128_3("s_128_3");
	tapa::stream<Cnoc_pkt, 10> s_144_4("s_144_4");
	tapa::stream<Cnoc_pkt, 2> s_63_3("s_63_3");
	tapa::stream<Cnoc_pkt, 2> s_64_5("s_64_5");
	tapa::stream<Cnoc_pkt, 2> s_32_4("s_32_4");
	tapa::stream<Cnoc_pkt, 20> s_63_4("s_63_4");
	tapa::stream<Cnoc_pkt, 20> s_64_6("s_64_6");
	tapa::stream<Cnoc_pkt, 2> s_95_4("s_95_4");
	tapa::stream<Cnoc_pkt, 2> s_31_3("s_31_3");
	tapa::stream<Cnoc_pkt, 2> s_32_5("s_32_5");
	tapa::stream<Cnoc_pkt, 2> s_16_4("s_16_4");
	tapa::stream<Cnoc_pkt, 16> s_31_4("s_31_4");
	tapa::stream<Cnoc_pkt, 16> s_32_6("s_32_6");
	tapa::stream<Cnoc_pkt, 2> s_47_4("s_47_4");
	tapa::stream<Cnoc_pkt, 2> s_95_5("s_95_5");
	tapa::stream<Cnoc_pkt, 2> s_96_3("s_96_3");
	tapa::stream<Cnoc_pkt, 2> s_80_4("s_80_4");
	tapa::stream<Cnoc_pkt, 16> s_95_6("s_95_6");
	tapa::stream<Cnoc_pkt, 16> s_96_4("s_96_4");
	tapa::stream<Cnoc_pkt, 2> s_111_4("s_111_4");
	tapa::stream<Cnoc_pkt, 2> s_15_3("s_15_3");
	tapa::stream<Cnoc_pkt, 2> s_16_5("s_16_5");
	tapa::stream<Cnoc_pkt, 2> s_8_4("s_8_4");
	tapa::stream<Cnoc_pkt, 12> s_15_4("s_15_4");
	tapa::stream<Cnoc_pkt, 12> s_16_6("s_16_6");
	tapa::stream<Cnoc_pkt, 2> s_23_4("s_23_4");
	tapa::stream<Cnoc_pkt, 2> s_47_5("s_47_5");
	tapa::stream<Cnoc_pkt, 2> s_48_3("s_48_3");
	tapa::stream<Cnoc_pkt, 2> s_40_4("s_40_4");
	tapa::stream<Cnoc_pkt, 12> s_47_6("s_47_6");
	tapa::stream<Cnoc_pkt, 12> s_48_4("s_48_4");
	tapa::stream<Cnoc_pkt, 2> s_55_4("s_55_4");
	tapa::stream<Cnoc_pkt, 2> s_79_3("s_79_3");
	tapa::stream<Cnoc_pkt, 2> s_80_5("s_80_5");
	tapa::stream<Cnoc_pkt, 2> s_72_4("s_72_4");
	tapa::stream<Cnoc_pkt, 12> s_79_4("s_79_4");
	tapa::stream<Cnoc_pkt, 12> s_80_6("s_80_6");
	tapa::stream<Cnoc_pkt, 2> s_87_4("s_87_4");
	tapa::stream<Cnoc_pkt, 2> s_111_5("s_111_5");
	tapa::stream<Cnoc_pkt, 2> s_112_3("s_112_3");
	tapa::stream<Cnoc_pkt, 2> s_104_4("s_104_4");
	tapa::stream<Cnoc_pkt, 12> s_111_6("s_111_6");
	tapa::stream<Cnoc_pkt, 12> s_112_4("s_112_4");
	tapa::stream<Cnoc_pkt, 2> s_119_4("s_119_4");
	tapa::stream<Cnoc_pkt, 2> s_143_3("s_143_3");
	tapa::stream<Cnoc_pkt, 2> s_144_5("s_144_5");
	tapa::stream<Cnoc_pkt, 2> s_136_4("s_136_4");
	tapa::stream<Cnoc_pkt, 12> s_143_4("s_143_4");
	tapa::stream<Cnoc_pkt, 12> s_144_6("s_144_6");
	tapa::stream<Cnoc_pkt, 2> s_151_4("s_151_4");
	tapa::stream<Cnoc_pkt, 2> s_7_3("s_7_3");
	tapa::stream<Cnoc_pkt, 2> s_8_5("s_8_5");
	tapa::stream<Cnoc_pkt, 2> s_4_4("s_4_4");
	tapa::stream<Cnoc_pkt, 8> s_7_4("s_7_4");
	tapa::stream<Cnoc_pkt, 8> s_8_6("s_8_6");
	tapa::stream<Cnoc_pkt, 2> s_11_4("s_11_4");
	tapa::stream<Cnoc_pkt, 2> s_23_5("s_23_5");
	tapa::stream<Cnoc_pkt, 2> s_24_3("s_24_3");
	tapa::stream<Cnoc_pkt, 2> s_20_4("s_20_4");
	tapa::stream<Cnoc_pkt, 8> s_23_6("s_23_6");
	tapa::stream<Cnoc_pkt, 8> s_24_4("s_24_4");
	tapa::stream<Cnoc_pkt, 2> s_27_4("s_27_4");
	tapa::stream<Cnoc_pkt, 2> s_39_3("s_39_3");
	tapa::stream<Cnoc_pkt, 2> s_40_5("s_40_5");
	tapa::stream<Cnoc_pkt, 2> s_36_4("s_36_4");
	tapa::stream<Cnoc_pkt, 8> s_39_4("s_39_4");
	tapa::stream<Cnoc_pkt, 8> s_40_6("s_40_6");
	tapa::stream<Cnoc_pkt, 2> s_43_4("s_43_4");
	tapa::stream<Cnoc_pkt, 2> s_55_5("s_55_5");
	tapa::stream<Cnoc_pkt, 2> s_56_3("s_56_3");
	tapa::stream<Cnoc_pkt, 2> s_52_4("s_52_4");
	tapa::stream<Cnoc_pkt, 8> s_55_6("s_55_6");
	tapa::stream<Cnoc_pkt, 8> s_56_4("s_56_4");
	tapa::stream<Cnoc_pkt, 2> s_59_4("s_59_4");
	tapa::stream<Cnoc_pkt, 2> s_71_3("s_71_3");
	tapa::stream<Cnoc_pkt, 2> s_72_5("s_72_5");
	tapa::stream<Cnoc_pkt, 2> s_68_4("s_68_4");
	tapa::stream<Cnoc_pkt, 8> s_71_4("s_71_4");
	tapa::stream<Cnoc_pkt, 8> s_72_6("s_72_6");
	tapa::stream<Cnoc_pkt, 2> s_75_4("s_75_4");
	tapa::stream<Cnoc_pkt, 2> s_87_5("s_87_5");
	tapa::stream<Cnoc_pkt, 2> s_88_3("s_88_3");
	tapa::stream<Cnoc_pkt, 2> s_84_4("s_84_4");
	tapa::stream<Cnoc_pkt, 8> s_87_6("s_87_6");
	tapa::stream<Cnoc_pkt, 8> s_88_4("s_88_4");
	tapa::stream<Cnoc_pkt, 2> s_91_4("s_91_4");
	tapa::stream<Cnoc_pkt, 2> s_103_3("s_103_3");
	tapa::stream<Cnoc_pkt, 2> s_104_5("s_104_5");
	tapa::stream<Cnoc_pkt, 2> s_100_4("s_100_4");
	tapa::stream<Cnoc_pkt, 8> s_103_4("s_103_4");
	tapa::stream<Cnoc_pkt, 8> s_104_6("s_104_6");
	tapa::stream<Cnoc_pkt, 2> s_107_4("s_107_4");
	tapa::stream<Cnoc_pkt, 2> s_119_5("s_119_5");
	tapa::stream<Cnoc_pkt, 2> s_120_3("s_120_3");
	tapa::stream<Cnoc_pkt, 2> s_116_4("s_116_4");
	tapa::stream<Cnoc_pkt, 8> s_119_6("s_119_6");
	tapa::stream<Cnoc_pkt, 8> s_120_4("s_120_4");
	tapa::stream<Cnoc_pkt, 2> s_123_4("s_123_4");
	tapa::stream<Cnoc_pkt, 2> s_135_3("s_135_3");
	tapa::stream<Cnoc_pkt, 2> s_136_5("s_136_5");
	tapa::stream<Cnoc_pkt, 2> s_132_4("s_132_4");
	tapa::stream<Cnoc_pkt, 8> s_135_4("s_135_4");
	tapa::stream<Cnoc_pkt, 8> s_136_6("s_136_6");
	tapa::stream<Cnoc_pkt, 2> s_139_4("s_139_4");
	tapa::stream<Cnoc_pkt, 2> s_151_5("s_151_5");
	tapa::stream<Cnoc_pkt, 2> s_152_3("s_152_3");
	tapa::stream<Cnoc_pkt, 2> s_148_4("s_148_4");
	tapa::stream<Cnoc_pkt, 8> s_151_6("s_151_6");
	tapa::stream<Cnoc_pkt, 8> s_152_4("s_152_4");
	tapa::stream<Cnoc_pkt, 2> s_155_4("s_155_4");
	tapa::stream<Cnoc_pkt, 2> s_3_3("s_3_3");
	tapa::stream<Cnoc_pkt, 2> s_4_5("s_4_5");
	tapa::stream<Cnoc_pkt, 2> s_2_3("s_2_3");
	tapa::stream<Cnoc_pkt, 4> s_3_4("s_3_4");
	tapa::stream<Cnoc_pkt, 4> s_4_6("s_4_6");
	tapa::stream<Cnoc_pkt, 2> s_5_3("s_5_3");
	tapa::stream<Cnoc_pkt, 2> s_11_5("s_11_5");
	tapa::stream<Cnoc_pkt, 2> s_12_3("s_12_3");
	tapa::stream<Cnoc_pkt, 2> s_10_3("s_10_3");
	tapa::stream<Cnoc_pkt, 4> s_11_6("s_11_6");
	tapa::stream<Cnoc_pkt, 4> s_12_4("s_12_4");
	tapa::stream<Cnoc_pkt, 2> s_13_3("s_13_3");
	tapa::stream<Cnoc_pkt, 2> s_19_3("s_19_3");
	tapa::stream<Cnoc_pkt, 2> s_20_5("s_20_5");
	tapa::stream<Cnoc_pkt, 2> s_18_3("s_18_3");
	tapa::stream<Cnoc_pkt, 4> s_19_4("s_19_4");
	tapa::stream<Cnoc_pkt, 4> s_20_6("s_20_6");
	tapa::stream<Cnoc_pkt, 2> s_21_3("s_21_3");
	tapa::stream<Cnoc_pkt, 2> s_27_5("s_27_5");
	tapa::stream<Cnoc_pkt, 2> s_28_3("s_28_3");
	tapa::stream<Cnoc_pkt, 2> s_26_3("s_26_3");
	tapa::stream<Cnoc_pkt, 4> s_27_6("s_27_6");
	tapa::stream<Cnoc_pkt, 4> s_28_4("s_28_4");
	tapa::stream<Cnoc_pkt, 2> s_29_3("s_29_3");
	tapa::stream<Cnoc_pkt, 2> s_35_3("s_35_3");
	tapa::stream<Cnoc_pkt, 2> s_36_5("s_36_5");
	tapa::stream<Cnoc_pkt, 2> s_34_3("s_34_3");
	tapa::stream<Cnoc_pkt, 4> s_35_4("s_35_4");
	tapa::stream<Cnoc_pkt, 4> s_36_6("s_36_6");
	tapa::stream<Cnoc_pkt, 2> s_37_3("s_37_3");
	tapa::stream<Cnoc_pkt, 2> s_43_5("s_43_5");
	tapa::stream<Cnoc_pkt, 2> s_44_3("s_44_3");
	tapa::stream<Cnoc_pkt, 2> s_42_3("s_42_3");
	tapa::stream<Cnoc_pkt, 4> s_43_6("s_43_6");
	tapa::stream<Cnoc_pkt, 4> s_44_4("s_44_4");
	tapa::stream<Cnoc_pkt, 2> s_45_3("s_45_3");
	tapa::stream<Cnoc_pkt, 2> s_51_3("s_51_3");
	tapa::stream<Cnoc_pkt, 2> s_52_5("s_52_5");
	tapa::stream<Cnoc_pkt, 2> s_50_3("s_50_3");
	tapa::stream<Cnoc_pkt, 4> s_51_4("s_51_4");
	tapa::stream<Cnoc_pkt, 4> s_52_6("s_52_6");
	tapa::stream<Cnoc_pkt, 2> s_53_3("s_53_3");
	tapa::stream<Cnoc_pkt, 2> s_59_5("s_59_5");
	tapa::stream<Cnoc_pkt, 2> s_60_3("s_60_3");
	tapa::stream<Cnoc_pkt, 2> s_58_3("s_58_3");
	tapa::stream<Cnoc_pkt, 4> s_59_6("s_59_6");
	tapa::stream<Cnoc_pkt, 4> s_60_4("s_60_4");
	tapa::stream<Cnoc_pkt, 2> s_61_3("s_61_3");
	tapa::stream<Cnoc_pkt, 2> s_67_3("s_67_3");
	tapa::stream<Cnoc_pkt, 2> s_68_5("s_68_5");
	tapa::stream<Cnoc_pkt, 2> s_66_3("s_66_3");
	tapa::stream<Cnoc_pkt, 4> s_67_4("s_67_4");
	tapa::stream<Cnoc_pkt, 4> s_68_6("s_68_6");
	tapa::stream<Cnoc_pkt, 2> s_69_3("s_69_3");
	tapa::stream<Cnoc_pkt, 2> s_75_5("s_75_5");
	tapa::stream<Cnoc_pkt, 2> s_76_3("s_76_3");
	tapa::stream<Cnoc_pkt, 2> s_74_3("s_74_3");
	tapa::stream<Cnoc_pkt, 4> s_75_6("s_75_6");
	tapa::stream<Cnoc_pkt, 4> s_76_4("s_76_4");
	tapa::stream<Cnoc_pkt, 2> s_77_3("s_77_3");
	tapa::stream<Cnoc_pkt, 2> s_83_3("s_83_3");
	tapa::stream<Cnoc_pkt, 2> s_84_5("s_84_5");
	tapa::stream<Cnoc_pkt, 2> s_82_3("s_82_3");
	tapa::stream<Cnoc_pkt, 4> s_83_4("s_83_4");
	tapa::stream<Cnoc_pkt, 4> s_84_6("s_84_6");
	tapa::stream<Cnoc_pkt, 2> s_85_3("s_85_3");
	tapa::stream<Cnoc_pkt, 2> s_91_5("s_91_5");
	tapa::stream<Cnoc_pkt, 2> s_92_3("s_92_3");
	tapa::stream<Cnoc_pkt, 2> s_90_3("s_90_3");
	tapa::stream<Cnoc_pkt, 4> s_91_6("s_91_6");
	tapa::stream<Cnoc_pkt, 4> s_92_4("s_92_4");
	tapa::stream<Cnoc_pkt, 2> s_93_3("s_93_3");
	tapa::stream<Cnoc_pkt, 2> s_99_3("s_99_3");
	tapa::stream<Cnoc_pkt, 2> s_100_5("s_100_5");
	tapa::stream<Cnoc_pkt, 2> s_98_3("s_98_3");
	tapa::stream<Cnoc_pkt, 4> s_99_4("s_99_4");
	tapa::stream<Cnoc_pkt, 4> s_100_6("s_100_6");
	tapa::stream<Cnoc_pkt, 2> s_101_3("s_101_3");
	tapa::stream<Cnoc_pkt, 2> s_107_5("s_107_5");
	tapa::stream<Cnoc_pkt, 2> s_108_3("s_108_3");
	tapa::stream<Cnoc_pkt, 2> s_106_3("s_106_3");
	tapa::stream<Cnoc_pkt, 4> s_107_6("s_107_6");
	tapa::stream<Cnoc_pkt, 4> s_108_4("s_108_4");
	tapa::stream<Cnoc_pkt, 2> s_109_3("s_109_3");
	tapa::stream<Cnoc_pkt, 2> s_115_3("s_115_3");
	tapa::stream<Cnoc_pkt, 2> s_116_5("s_116_5");
	tapa::stream<Cnoc_pkt, 2> s_114_3("s_114_3");
	tapa::stream<Cnoc_pkt, 4> s_115_4("s_115_4");
	tapa::stream<Cnoc_pkt, 4> s_116_6("s_116_6");
	tapa::stream<Cnoc_pkt, 2> s_117_3("s_117_3");
	tapa::stream<Cnoc_pkt, 2> s_123_5("s_123_5");
	tapa::stream<Cnoc_pkt, 2> s_124_3("s_124_3");
	tapa::stream<Cnoc_pkt, 2> s_122_3("s_122_3");
	tapa::stream<Cnoc_pkt, 4> s_123_6("s_123_6");
	tapa::stream<Cnoc_pkt, 4> s_124_4("s_124_4");
	tapa::stream<Cnoc_pkt, 2> s_125_3("s_125_3");
	tapa::stream<Cnoc_pkt, 2> s_131_3("s_131_3");
	tapa::stream<Cnoc_pkt, 2> s_132_5("s_132_5");
	tapa::stream<Cnoc_pkt, 2> s_130_3("s_130_3");
	tapa::stream<Cnoc_pkt, 4> s_131_4("s_131_4");
	tapa::stream<Cnoc_pkt, 4> s_132_6("s_132_6");
	tapa::stream<Cnoc_pkt, 2> s_133_3("s_133_3");
	tapa::stream<Cnoc_pkt, 2> s_139_5("s_139_5");
	tapa::stream<Cnoc_pkt, 2> s_140_3("s_140_3");
	tapa::stream<Cnoc_pkt, 2> s_138_3("s_138_3");
	tapa::stream<Cnoc_pkt, 4> s_139_6("s_139_6");
	tapa::stream<Cnoc_pkt, 4> s_140_4("s_140_4");
	tapa::stream<Cnoc_pkt, 2> s_141_3("s_141_3");
	tapa::stream<Cnoc_pkt, 2> s_147_3("s_147_3");
	tapa::stream<Cnoc_pkt, 2> s_148_5("s_148_5");
	tapa::stream<Cnoc_pkt, 2> s_146_3("s_146_3");
	tapa::stream<Cnoc_pkt, 4> s_147_4("s_147_4");
	tapa::stream<Cnoc_pkt, 4> s_148_6("s_148_6");
	tapa::stream<Cnoc_pkt, 2> s_149_3("s_149_3");
	tapa::stream<Cnoc_pkt, 2> s_155_5("s_155_5");
	tapa::stream<Cnoc_pkt, 2> s_156_3("s_156_3");
	tapa::stream<Cnoc_pkt, 2> s_154_3("s_154_3");
	tapa::stream<Cnoc_pkt, 4> s_155_6("s_155_6");
	tapa::stream<Cnoc_pkt, 4> s_156_4("s_156_4");
	tapa::stream<Cnoc_pkt, 2> s_157_3("s_157_3");
	tapa::stream<Cnoc_pkt, 2> s_1_2("s_1_2");
	tapa::stream<Cnoc_pkt, 2> s_2_4("s_2_4");
	tapa::stream<Cnoc_pkt, 2> s_5_4("s_5_4");
	tapa::stream<Cnoc_pkt, 2> s_6_2("s_6_2");
	tapa::stream<Cnoc_pkt, 2> s_9_2("s_9_2");
	tapa::stream<Cnoc_pkt, 2> s_10_4("s_10_4");
	tapa::stream<Cnoc_pkt, 2> s_13_4("s_13_4");
	tapa::stream<Cnoc_pkt, 2> s_14_2("s_14_2");
	tapa::stream<Cnoc_pkt, 2> s_17_2("s_17_2");
	tapa::stream<Cnoc_pkt, 2> s_18_4("s_18_4");
	tapa::stream<Cnoc_pkt, 2> s_21_4("s_21_4");
	tapa::stream<Cnoc_pkt, 2> s_22_2("s_22_2");
	tapa::stream<Cnoc_pkt, 2> s_25_2("s_25_2");
	tapa::stream<Cnoc_pkt, 2> s_26_4("s_26_4");
	tapa::stream<Cnoc_pkt, 2> s_29_4("s_29_4");
	tapa::stream<Cnoc_pkt, 2> s_30_2("s_30_2");
	tapa::stream<Cnoc_pkt, 2> s_33_2("s_33_2");
	tapa::stream<Cnoc_pkt, 2> s_34_4("s_34_4");
	tapa::stream<Cnoc_pkt, 2> s_37_4("s_37_4");
	tapa::stream<Cnoc_pkt, 2> s_38_2("s_38_2");
	tapa::stream<Cnoc_pkt, 2> s_41_2("s_41_2");
	tapa::stream<Cnoc_pkt, 2> s_42_4("s_42_4");
	tapa::stream<Cnoc_pkt, 2> s_45_4("s_45_4");
	tapa::stream<Cnoc_pkt, 2> s_46_2("s_46_2");
	tapa::stream<Cnoc_pkt, 2> s_49_2("s_49_2");
	tapa::stream<Cnoc_pkt, 2> s_50_4("s_50_4");
	tapa::stream<Cnoc_pkt, 2> s_53_4("s_53_4");
	tapa::stream<Cnoc_pkt, 2> s_54_2("s_54_2");
	tapa::stream<Cnoc_pkt, 2> s_57_2("s_57_2");
	tapa::stream<Cnoc_pkt, 2> s_58_4("s_58_4");
	tapa::stream<Cnoc_pkt, 2> s_61_4("s_61_4");
	tapa::stream<Cnoc_pkt, 2> s_62_2("s_62_2");
	tapa::stream<Cnoc_pkt, 2> s_65_2("s_65_2");
	tapa::stream<Cnoc_pkt, 2> s_66_4("s_66_4");
	tapa::stream<Cnoc_pkt, 2> s_69_4("s_69_4");
	tapa::stream<Cnoc_pkt, 2> s_70_2("s_70_2");
	tapa::stream<Cnoc_pkt, 2> s_73_2("s_73_2");
	tapa::stream<Cnoc_pkt, 2> s_74_4("s_74_4");
	tapa::stream<Cnoc_pkt, 2> s_77_4("s_77_4");
	tapa::stream<Cnoc_pkt, 2> s_78_2("s_78_2");
	tapa::stream<Cnoc_pkt, 2> s_81_2("s_81_2");
	tapa::stream<Cnoc_pkt, 2> s_82_4("s_82_4");
	tapa::stream<Cnoc_pkt, 2> s_85_4("s_85_4");
	tapa::stream<Cnoc_pkt, 2> s_86_2("s_86_2");
	tapa::stream<Cnoc_pkt, 2> s_89_2("s_89_2");
	tapa::stream<Cnoc_pkt, 2> s_90_4("s_90_4");
	tapa::stream<Cnoc_pkt, 2> s_93_4("s_93_4");
	tapa::stream<Cnoc_pkt, 2> s_94_2("s_94_2");
	tapa::stream<Cnoc_pkt, 2> s_97_2("s_97_2");
	tapa::stream<Cnoc_pkt, 2> s_98_4("s_98_4");
	tapa::stream<Cnoc_pkt, 2> s_101_4("s_101_4");
	tapa::stream<Cnoc_pkt, 2> s_102_2("s_102_2");
	tapa::stream<Cnoc_pkt, 2> s_105_2("s_105_2");
	tapa::stream<Cnoc_pkt, 2> s_106_4("s_106_4");
	tapa::stream<Cnoc_pkt, 2> s_109_4("s_109_4");
	tapa::stream<Cnoc_pkt, 2> s_110_2("s_110_2");
	tapa::stream<Cnoc_pkt, 2> s_113_2("s_113_2");
	tapa::stream<Cnoc_pkt, 2> s_114_4("s_114_4");
	tapa::stream<Cnoc_pkt, 2> s_117_4("s_117_4");
	tapa::stream<Cnoc_pkt, 2> s_118_2("s_118_2");
	tapa::stream<Cnoc_pkt, 2> s_121_2("s_121_2");
	tapa::stream<Cnoc_pkt, 2> s_122_4("s_122_4");
	tapa::stream<Cnoc_pkt, 2> s_125_4("s_125_4");
	tapa::stream<Cnoc_pkt, 2> s_126_2("s_126_2");
	tapa::stream<Cnoc_pkt, 2> s_129_2("s_129_2");
	tapa::stream<Cnoc_pkt, 2> s_130_4("s_130_4");
	tapa::stream<Cnoc_pkt, 2> s_133_4("s_133_4");
	tapa::stream<Cnoc_pkt, 2> s_134_2("s_134_2");
	tapa::stream<Cnoc_pkt, 2> s_137_2("s_137_2");
	tapa::stream<Cnoc_pkt, 2> s_138_4("s_138_4");
	tapa::stream<Cnoc_pkt, 2> s_141_4("s_141_4");
	tapa::stream<Cnoc_pkt, 2> s_142_2("s_142_2");
	tapa::stream<Cnoc_pkt, 2> s_145_2("s_145_2");
	tapa::stream<Cnoc_pkt, 2> s_146_4("s_146_4");
	tapa::stream<Cnoc_pkt, 2> s_149_4("s_149_4");
	tapa::stream<Cnoc_pkt, 2> s_150_2("s_150_2");
	tapa::stream<Cnoc_pkt, 2> s_153_2("s_153_2");
	tapa::stream<Cnoc_pkt, 2> s_154_4("s_154_4");
	tapa::stream<Cnoc_pkt, 2> s_157_4("s_157_4");
	tapa::stream<Cnoc_pkt, 2> s_158_2("s_158_2");

    tapa::task()
        .invoke<tapa::join, NUM_CH>(MM2S_A, A, FIFO_A_IN, len, rp_time)
        .invoke(MM2S_B, b, FIFO_B_IN, num_tiles_r, num_cols_16, rp_time)
        .invoke<tapa::join, NUM_PES_HALF>(LoadB, FIFO_B_IN, FIFO_B_IN, BUFF_B, num_cols_16, num_tiles, USE_DOUBLE_BUFFER)
        .invoke<tapa::join, NUM_PES_HALF>(ComputeAB, FIFO_A_IN, FIFO_C_FLAG, FIFO_C_ROW, FIFO_C_VAL, BUFF_B, num_tiles, USE_DOUBLE_BUFFER)
        .invoke<tapa::join>(DummyRead, FIFO_B_IN, num_cols_16, num_tiles_r, rp_time)
        .invoke<tapa::join, NUM_PES_HALF>(TreeAdder, FIFO_C_ROW, FIFO_C_VAL, FIFO_C_FLAG, FIFO_C_SHF)
		.invoke(ADD_1, FIFO_C_SHF[0], FIFO_C_SHF[1], s_0_0, s_1_0)/*0*/
		.invoke(ADD_0, FIFO_C_SHF[2], FIFO_C_SHF[3], s_2_0, s_3_0)/*1*/
		.invoke(ADD_1, FIFO_C_SHF[4], FIFO_C_SHF[5], s_4_0, s_5_0)/*2*/
		.invoke(ADD_0, FIFO_C_SHF[6], FIFO_C_SHF[7], s_6_0, s_7_0)/*3*/
		.invoke(ADD_1, FIFO_C_SHF[8], FIFO_C_SHF[9], s_8_0, s_9_0)/*4*/
		.invoke(ADD_0, FIFO_C_SHF[10], FIFO_C_SHF[11], s_10_0, s_11_0)/*5*/
		.invoke(ADD_1, FIFO_C_SHF[12], FIFO_C_SHF[13], s_12_0, s_13_0)/*6*/
		.invoke(ADD_0, FIFO_C_SHF[14], FIFO_C_SHF[15], s_14_0, s_15_0)/*7*/
		.invoke(ADD_1, FIFO_C_SHF[16], FIFO_C_SHF[17], s_16_0, s_17_0)/*8*/
		.invoke(ADD_0, FIFO_C_SHF[18], FIFO_C_SHF[19], s_18_0, s_19_0)/*9*/
		.invoke(ADD_1, FIFO_C_SHF[20], FIFO_C_SHF[21], s_20_0, s_21_0)/*10*/
		.invoke(ADD_0, FIFO_C_SHF[22], FIFO_C_SHF[23], s_22_0, s_23_0)/*11*/
		.invoke(ADD_1, FIFO_C_SHF[24], FIFO_C_SHF[25], s_24_0, s_25_0)/*12*/
		.invoke(ADD_0, FIFO_C_SHF[26], FIFO_C_SHF[27], s_26_0, s_27_0)/*13*/
		.invoke(ADD_1, FIFO_C_SHF[28], FIFO_C_SHF[29], s_28_0, s_29_0)/*14*/
		.invoke(ADD_0, FIFO_C_SHF[30], FIFO_C_SHF[31], s_30_0, s_31_0)/*15*/
		.invoke(ADD_1, FIFO_C_SHF[32], FIFO_C_SHF[33], s_32_0, s_33_0)/*16*/
		.invoke(ADD_0, FIFO_C_SHF[34], FIFO_C_SHF[35], s_34_0, s_35_0)/*17*/
		.invoke(ADD_1, FIFO_C_SHF[36], FIFO_C_SHF[37], s_36_0, s_37_0)/*18*/
		.invoke(ADD_0, FIFO_C_SHF[38], FIFO_C_SHF[39], s_38_0, s_39_0)/*19*/
		.invoke(ADD_1, FIFO_C_SHF[40], FIFO_C_SHF[41], s_40_0, s_41_0)/*20*/
		.invoke(ADD_0, FIFO_C_SHF[42], FIFO_C_SHF[43], s_42_0, s_43_0)/*21*/
		.invoke(ADD_1, FIFO_C_SHF[44], FIFO_C_SHF[45], s_44_0, s_45_0)/*22*/
		.invoke(ADD_0, FIFO_C_SHF[46], FIFO_C_SHF[47], s_46_0, s_47_0)/*23*/
		.invoke(ADD_1, FIFO_C_SHF[48], FIFO_C_SHF[49], s_48_0, s_49_0)/*24*/
		.invoke(ADD_0, FIFO_C_SHF[50], FIFO_C_SHF[51], s_50_0, s_51_0)/*25*/
		.invoke(ADD_1, FIFO_C_SHF[52], FIFO_C_SHF[53], s_52_0, s_53_0)/*26*/
		.invoke(ADD_0, FIFO_C_SHF[54], FIFO_C_SHF[55], s_54_0, s_55_0)/*27*/
		.invoke(ADD_1, FIFO_C_SHF[56], FIFO_C_SHF[57], s_56_0, s_57_0)/*28*/
		.invoke(ADD_0, FIFO_C_SHF[58], FIFO_C_SHF[59], s_58_0, s_59_0)/*29*/
		.invoke(ADD_1, FIFO_C_SHF[60], FIFO_C_SHF[61], s_60_0, s_61_0)/*30*/
		.invoke(ADD_0, FIFO_C_SHF[62], FIFO_C_SHF[63], s_62_0, s_63_0)/*31*/
		.invoke(ADD_1, FIFO_C_SHF[64], FIFO_C_SHF[65], s_64_0, s_65_0)/*32*/
		.invoke(ADD_0, FIFO_C_SHF[66], FIFO_C_SHF[67], s_66_0, s_67_0)/*33*/
		.invoke(ADD_1, FIFO_C_SHF[68], FIFO_C_SHF[69], s_68_0, s_69_0)/*34*/
		.invoke(ADD_0, FIFO_C_SHF[70], FIFO_C_SHF[71], s_70_0, s_71_0)/*35*/
		.invoke(ADD_1, FIFO_C_SHF[72], FIFO_C_SHF[73], s_72_0, s_73_0)/*36*/
		.invoke(ADD_0, FIFO_C_SHF[74], FIFO_C_SHF[75], s_74_0, s_75_0)/*37*/
		.invoke(ADD_1, FIFO_C_SHF[76], FIFO_C_SHF[77], s_76_0, s_77_0)/*38*/
		.invoke(ADD_0, FIFO_C_SHF[78], FIFO_C_SHF[79], s_78_0, s_79_0)/*39*/
		.invoke(ADD_1, FIFO_C_SHF[80], FIFO_C_SHF[81], s_80_0, s_81_0)/*40*/
		.invoke(ADD_0, FIFO_C_SHF[82], FIFO_C_SHF[83], s_82_0, s_83_0)/*41*/
		.invoke(ADD_1, FIFO_C_SHF[84], FIFO_C_SHF[85], s_84_0, s_85_0)/*42*/
		.invoke(ADD_0, FIFO_C_SHF[86], FIFO_C_SHF[87], s_86_0, s_87_0)/*43*/
		.invoke(ADD_1, FIFO_C_SHF[88], FIFO_C_SHF[89], s_88_0, s_89_0)/*44*/
		.invoke(ADD_0, FIFO_C_SHF[90], FIFO_C_SHF[91], s_90_0, s_91_0)/*45*/
		.invoke(ADD_1, FIFO_C_SHF[92], FIFO_C_SHF[93], s_92_0, s_93_0)/*46*/
		.invoke(ADD_0, FIFO_C_SHF[94], FIFO_C_SHF[95], s_94_0, s_95_0)/*47*/
		.invoke(ADD_1, FIFO_C_SHF[96], FIFO_C_SHF[97], s_96_0, s_97_0)/*48*/
		.invoke(ADD_0, FIFO_C_SHF[98], FIFO_C_SHF[99], s_98_0, s_99_0)/*49*/
		.invoke(ADD_1, FIFO_C_SHF[100], FIFO_C_SHF[101], s_100_0, s_101_0)/*50*/
		.invoke(ADD_0, FIFO_C_SHF[102], FIFO_C_SHF[103], s_102_0, s_103_0)/*51*/
		.invoke(ADD_1, FIFO_C_SHF[104], FIFO_C_SHF[105], s_104_0, s_105_0)/*52*/
		.invoke(ADD_0, FIFO_C_SHF[106], FIFO_C_SHF[107], s_106_0, s_107_0)/*53*/
		.invoke(ADD_1, FIFO_C_SHF[108], FIFO_C_SHF[109], s_108_0, s_109_0)/*54*/
		.invoke(ADD_0, FIFO_C_SHF[110], FIFO_C_SHF[111], s_110_0, s_111_0)/*55*/
		.invoke(ADD_1, FIFO_C_SHF[112], FIFO_C_SHF[113], s_112_0, s_113_0)/*56*/
		.invoke(ADD_0, FIFO_C_SHF[114], FIFO_C_SHF[115], s_114_0, s_115_0)/*57*/
		.invoke(ADD_1, FIFO_C_SHF[116], FIFO_C_SHF[117], s_116_0, s_117_0)/*58*/
		.invoke(ADD_0, FIFO_C_SHF[118], FIFO_C_SHF[119], s_118_0, s_119_0)/*59*/
		.invoke(ADD_1, FIFO_C_SHF[120], FIFO_C_SHF[121], s_120_0, s_121_0)/*60*/
		.invoke(ADD_0, FIFO_C_SHF[122], FIFO_C_SHF[123], s_122_0, s_123_0)/*61*/
		.invoke(ADD_1, FIFO_C_SHF[124], FIFO_C_SHF[125], s_124_0, s_125_0)/*62*/
		.invoke(ADD_0, FIFO_C_SHF[126], FIFO_C_SHF[127], s_126_0, s_127_0)/*63*/
		.invoke(ADD_1, FIFO_C_SHF[128], FIFO_C_SHF[129], s_128_0, s_129_0)/*64*/
		.invoke(ADD_0, FIFO_C_SHF[130], FIFO_C_SHF[131], s_130_0, s_131_0)/*65*/
		.invoke(ADD_1, FIFO_C_SHF[132], FIFO_C_SHF[133], s_132_0, s_133_0)/*66*/
		.invoke(ADD_0, FIFO_C_SHF[134], FIFO_C_SHF[135], s_134_0, s_135_0)/*67*/
		.invoke(ADD_1, FIFO_C_SHF[136], FIFO_C_SHF[137], s_136_0, s_137_0)/*68*/
		.invoke(ADD_0, FIFO_C_SHF[138], FIFO_C_SHF[139], s_138_0, s_139_0)/*69*/
		.invoke(ADD_1, FIFO_C_SHF[140], FIFO_C_SHF[141], s_140_0, s_141_0)/*70*/
		.invoke(ADD_0, FIFO_C_SHF[142], FIFO_C_SHF[143], s_142_0, s_143_0)/*71*/
		.invoke(ADD_1, FIFO_C_SHF[144], FIFO_C_SHF[145], s_144_0, s_145_0)/*72*/
		.invoke(ADD_0, FIFO_C_SHF[146], FIFO_C_SHF[147], s_146_0, s_147_0)/*73*/
		.invoke(ADD_1, FIFO_C_SHF[148], FIFO_C_SHF[149], s_148_0, s_149_0)/*74*/
		.invoke(ADD_0, FIFO_C_SHF[150], FIFO_C_SHF[151], s_150_0, s_151_0)/*75*/
		.invoke(ADD_1, FIFO_C_SHF[152], FIFO_C_SHF[153], s_152_0, s_153_0)/*76*/
		.invoke(ADD_0, FIFO_C_SHF[154], FIFO_C_SHF[155], s_154_0, s_155_0)/*77*/
		.invoke(ADD_1, FIFO_C_SHF[156], FIFO_C_SHF[157], s_156_0, s_157_0)/*78*/
		.invoke(ADD_0, FIFO_C_SHF[158], FIFO_C_SHF[159], s_158_0, s_159_0)/*79*/
		.invoke(ADD_1, s_1_0, s_2_0, s_1_1, s_2_1)/*80*/
		.invoke(ADD_0, s_5_0, s_6_0, s_5_1, s_6_1)/*81*/
		.invoke(ADD_1, s_9_0, s_10_0, s_9_1, s_10_1)/*82*/
		.invoke(ADD_0, s_13_0, s_14_0, s_13_1, s_14_1)/*83*/
		.invoke(ADD_1, s_17_0, s_18_0, s_17_1, s_18_1)/*84*/
		.invoke(ADD_0, s_21_0, s_22_0, s_21_1, s_22_1)/*85*/
		.invoke(ADD_1, s_25_0, s_26_0, s_25_1, s_26_1)/*86*/
		.invoke(ADD_0, s_29_0, s_30_0, s_29_1, s_30_1)/*87*/
		.invoke(ADD_1, s_33_0, s_34_0, s_33_1, s_34_1)/*88*/
		.invoke(ADD_0, s_37_0, s_38_0, s_37_1, s_38_1)/*89*/
		.invoke(ADD_1, s_41_0, s_42_0, s_41_1, s_42_1)/*90*/
		.invoke(ADD_0, s_45_0, s_46_0, s_45_1, s_46_1)/*91*/
		.invoke(ADD_1, s_49_0, s_50_0, s_49_1, s_50_1)/*92*/
		.invoke(ADD_0, s_53_0, s_54_0, s_53_1, s_54_1)/*93*/
		.invoke(ADD_1, s_57_0, s_58_0, s_57_1, s_58_1)/*94*/
		.invoke(ADD_0, s_61_0, s_62_0, s_61_1, s_62_1)/*95*/
		.invoke(ADD_1, s_65_0, s_66_0, s_65_1, s_66_1)/*96*/
		.invoke(ADD_0, s_69_0, s_70_0, s_69_1, s_70_1)/*97*/
		.invoke(ADD_1, s_73_0, s_74_0, s_73_1, s_74_1)/*98*/
		.invoke(ADD_0, s_77_0, s_78_0, s_77_1, s_78_1)/*99*/
		.invoke(ADD_1, s_81_0, s_82_0, s_81_1, s_82_1)/*100*/
		.invoke(ADD_0, s_85_0, s_86_0, s_85_1, s_86_1)/*101*/
		.invoke(ADD_1, s_89_0, s_90_0, s_89_1, s_90_1)/*102*/
		.invoke(ADD_0, s_93_0, s_94_0, s_93_1, s_94_1)/*103*/
		.invoke(ADD_1, s_97_0, s_98_0, s_97_1, s_98_1)/*104*/
		.invoke(ADD_0, s_101_0, s_102_0, s_101_1, s_102_1)/*105*/
		.invoke(ADD_1, s_105_0, s_106_0, s_105_1, s_106_1)/*106*/
		.invoke(ADD_0, s_109_0, s_110_0, s_109_1, s_110_1)/*107*/
		.invoke(ADD_1, s_113_0, s_114_0, s_113_1, s_114_1)/*108*/
		.invoke(ADD_0, s_117_0, s_118_0, s_117_1, s_118_1)/*109*/
		.invoke(ADD_1, s_121_0, s_122_0, s_121_1, s_122_1)/*110*/
		.invoke(ADD_0, s_125_0, s_126_0, s_125_1, s_126_1)/*111*/
		.invoke(ADD_1, s_129_0, s_130_0, s_129_1, s_130_1)/*112*/
		.invoke(ADD_0, s_133_0, s_134_0, s_133_1, s_134_1)/*113*/
		.invoke(ADD_1, s_137_0, s_138_0, s_137_1, s_138_1)/*114*/
		.invoke(ADD_0, s_141_0, s_142_0, s_141_1, s_142_1)/*115*/
		.invoke(ADD_1, s_145_0, s_146_0, s_145_1, s_146_1)/*116*/
		.invoke(ADD_0, s_149_0, s_150_0, s_149_1, s_150_1)/*117*/
		.invoke(ADD_1, s_153_0, s_154_0, s_153_1, s_154_1)/*118*/
		.invoke(ADD_0, s_157_0, s_158_0, s_157_1, s_158_1)/*119*/
		.invoke(SSW, s_2_1, s_3_0, s_2_2, s_3_1)/*120*/
		.invoke(SSW, s_4_0, s_5_1, s_4_1, s_5_2)/*121*/
		.invoke(ADD_1, s_3_1, s_4_1, s_3_2, s_4_2)/*122*/
		.invoke(SSW, s_10_1, s_11_0, s_10_2, s_11_1)/*123*/
		.invoke(SSW, s_12_0, s_13_1, s_12_1, s_13_2)/*124*/
		.invoke(ADD_0, s_11_1, s_12_1, s_11_2, s_12_2)/*125*/
		.invoke(SSW, s_18_1, s_19_0, s_18_2, s_19_1)/*126*/
		.invoke(SSW, s_20_0, s_21_1, s_20_1, s_21_2)/*127*/
		.invoke(ADD_1, s_19_1, s_20_1, s_19_2, s_20_2)/*128*/
		.invoke(SSW, s_26_1, s_27_0, s_26_2, s_27_1)/*129*/
		.invoke(SSW, s_28_0, s_29_1, s_28_1, s_29_2)/*130*/
		.invoke(ADD_0, s_27_1, s_28_1, s_27_2, s_28_2)/*131*/
		.invoke(SSW, s_34_1, s_35_0, s_34_2, s_35_1)/*132*/
		.invoke(SSW, s_36_0, s_37_1, s_36_1, s_37_2)/*133*/
		.invoke(ADD_1, s_35_1, s_36_1, s_35_2, s_36_2)/*134*/
		.invoke(SSW, s_42_1, s_43_0, s_42_2, s_43_1)/*135*/
		.invoke(SSW, s_44_0, s_45_1, s_44_1, s_45_2)/*136*/
		.invoke(ADD_0, s_43_1, s_44_1, s_43_2, s_44_2)/*137*/
		.invoke(SSW, s_50_1, s_51_0, s_50_2, s_51_1)/*138*/
		.invoke(SSW, s_52_0, s_53_1, s_52_1, s_53_2)/*139*/
		.invoke(ADD_1, s_51_1, s_52_1, s_51_2, s_52_2)/*140*/
		.invoke(SSW, s_58_1, s_59_0, s_58_2, s_59_1)/*141*/
		.invoke(SSW, s_60_0, s_61_1, s_60_1, s_61_2)/*142*/
		.invoke(ADD_0, s_59_1, s_60_1, s_59_2, s_60_2)/*143*/
		.invoke(SSW, s_66_1, s_67_0, s_66_2, s_67_1)/*144*/
		.invoke(SSW, s_68_0, s_69_1, s_68_1, s_69_2)/*145*/
		.invoke(ADD_1, s_67_1, s_68_1, s_67_2, s_68_2)/*146*/
		.invoke(SSW, s_74_1, s_75_0, s_74_2, s_75_1)/*147*/
		.invoke(SSW, s_76_0, s_77_1, s_76_1, s_77_2)/*148*/
		.invoke(ADD_0, s_75_1, s_76_1, s_75_2, s_76_2)/*149*/
		.invoke(SSW, s_82_1, s_83_0, s_82_2, s_83_1)/*150*/
		.invoke(SSW, s_84_0, s_85_1, s_84_1, s_85_2)/*151*/
		.invoke(ADD_1, s_83_1, s_84_1, s_83_2, s_84_2)/*152*/
		.invoke(SSW, s_90_1, s_91_0, s_90_2, s_91_1)/*153*/
		.invoke(SSW, s_92_0, s_93_1, s_92_1, s_93_2)/*154*/
		.invoke(ADD_0, s_91_1, s_92_1, s_91_2, s_92_2)/*155*/
		.invoke(SSW, s_98_1, s_99_0, s_98_2, s_99_1)/*156*/
		.invoke(SSW, s_100_0, s_101_1, s_100_1, s_101_2)/*157*/
		.invoke(ADD_1, s_99_1, s_100_1, s_99_2, s_100_2)/*158*/
		.invoke(SSW, s_106_1, s_107_0, s_106_2, s_107_1)/*159*/
		.invoke(SSW, s_108_0, s_109_1, s_108_1, s_109_2)/*160*/
		.invoke(ADD_0, s_107_1, s_108_1, s_107_2, s_108_2)/*161*/
		.invoke(SSW, s_114_1, s_115_0, s_114_2, s_115_1)/*162*/
		.invoke(SSW, s_116_0, s_117_1, s_116_1, s_117_2)/*163*/
		.invoke(ADD_1, s_115_1, s_116_1, s_115_2, s_116_2)/*164*/
		.invoke(SSW, s_122_1, s_123_0, s_122_2, s_123_1)/*165*/
		.invoke(SSW, s_124_0, s_125_1, s_124_1, s_125_2)/*166*/
		.invoke(ADD_0, s_123_1, s_124_1, s_123_2, s_124_2)/*167*/
		.invoke(SSW, s_130_1, s_131_0, s_130_2, s_131_1)/*168*/
		.invoke(SSW, s_132_0, s_133_1, s_132_1, s_133_2)/*169*/
		.invoke(ADD_1, s_131_1, s_132_1, s_131_2, s_132_2)/*170*/
		.invoke(SSW, s_138_1, s_139_0, s_138_2, s_139_1)/*171*/
		.invoke(SSW, s_140_0, s_141_1, s_140_1, s_141_2)/*172*/
		.invoke(ADD_0, s_139_1, s_140_1, s_139_2, s_140_2)/*173*/
		.invoke(SSW, s_146_1, s_147_0, s_146_2, s_147_1)/*174*/
		.invoke(SSW, s_148_0, s_149_1, s_148_1, s_149_2)/*175*/
		.invoke(ADD_1, s_147_1, s_148_1, s_147_2, s_148_2)/*176*/
		.invoke(SSW, s_154_1, s_155_0, s_154_2, s_155_1)/*177*/
		.invoke(SSW, s_156_0, s_157_1, s_156_1, s_157_2)/*178*/
		.invoke(ADD_0, s_155_1, s_156_1, s_155_2, s_156_2)/*179*/
		.invoke(SSW, s_4_2, s_7_0, s_4_3, s_7_1)/*180*/
		.invoke(SSW, s_8_0, s_11_2, s_8_1, s_11_3)/*181*/
		.invoke(ADD_1, s_7_1, s_8_1, s_7_2, s_8_2)/*182*/
		.invoke(SSW, s_20_2, s_23_0, s_20_3, s_23_1)/*183*/
		.invoke(SSW, s_24_0, s_27_2, s_24_1, s_27_3)/*184*/
		.invoke(ADD_0, s_23_1, s_24_1, s_23_2, s_24_2)/*185*/
		.invoke(SSW, s_36_2, s_39_0, s_36_3, s_39_1)/*186*/
		.invoke(SSW, s_40_0, s_43_2, s_40_1, s_43_3)/*187*/
		.invoke(ADD_1, s_39_1, s_40_1, s_39_2, s_40_2)/*188*/
		.invoke(SSW, s_52_2, s_55_0, s_52_3, s_55_1)/*189*/
		.invoke(SSW, s_56_0, s_59_2, s_56_1, s_59_3)/*190*/
		.invoke(ADD_0, s_55_1, s_56_1, s_55_2, s_56_2)/*191*/
		.invoke(SSW, s_68_2, s_71_0, s_68_3, s_71_1)/*192*/
		.invoke(SSW, s_72_0, s_75_2, s_72_1, s_75_3)/*193*/
		.invoke(ADD_1, s_71_1, s_72_1, s_71_2, s_72_2)/*194*/
		.invoke(SSW, s_84_2, s_87_0, s_84_3, s_87_1)/*195*/
		.invoke(SSW, s_88_0, s_91_2, s_88_1, s_91_3)/*196*/
		.invoke(ADD_0, s_87_1, s_88_1, s_87_2, s_88_2)/*197*/
		.invoke(SSW, s_100_2, s_103_0, s_100_3, s_103_1)/*198*/
		.invoke(SSW, s_104_0, s_107_2, s_104_1, s_107_3)/*199*/
		.invoke(ADD_1, s_103_1, s_104_1, s_103_2, s_104_2)/*200*/
		.invoke(SSW, s_116_2, s_119_0, s_116_3, s_119_1)/*201*/
		.invoke(SSW, s_120_0, s_123_2, s_120_1, s_123_3)/*202*/
		.invoke(ADD_0, s_119_1, s_120_1, s_119_2, s_120_2)/*203*/
		.invoke(SSW, s_132_2, s_135_0, s_132_3, s_135_1)/*204*/
		.invoke(SSW, s_136_0, s_139_2, s_136_1, s_139_3)/*205*/
		.invoke(ADD_1, s_135_1, s_136_1, s_135_2, s_136_2)/*206*/
		.invoke(SSW, s_148_2, s_151_0, s_148_3, s_151_1)/*207*/
		.invoke(SSW, s_152_0, s_155_2, s_152_1, s_155_3)/*208*/
		.invoke(ADD_0, s_151_1, s_152_1, s_151_2, s_152_2)/*209*/
		.invoke(SSW, s_8_2, s_15_0, s_8_3, s_15_1)/*210*/
		.invoke(SSW, s_16_0, s_23_2, s_16_1, s_23_3)/*211*/
		.invoke(ADD_1, s_15_1, s_16_1, s_15_2, s_16_2)/*212*/
		.invoke(SSW, s_40_2, s_47_0, s_40_3, s_47_1)/*213*/
		.invoke(SSW, s_48_0, s_55_2, s_48_1, s_55_3)/*214*/
		.invoke(ADD_0, s_47_1, s_48_1, s_47_2, s_48_2)/*215*/
		.invoke(SSW, s_72_2, s_79_0, s_72_3, s_79_1)/*216*/
		.invoke(SSW, s_80_0, s_87_2, s_80_1, s_87_3)/*217*/
		.invoke(ADD_1, s_79_1, s_80_1, s_79_2, s_80_2)/*218*/
		.invoke(SSW, s_104_2, s_111_0, s_104_3, s_111_1)/*219*/
		.invoke(SSW, s_112_0, s_119_2, s_112_1, s_119_3)/*220*/
		.invoke(ADD_0, s_111_1, s_112_1, s_111_2, s_112_2)/*221*/
		.invoke(SSW, s_136_2, s_143_0, s_136_3, s_143_1)/*222*/
		.invoke(SSW, s_144_0, s_151_2, s_144_1, s_151_3)/*223*/
		.invoke(ADD_1, s_143_1, s_144_1, s_143_2, s_144_2)/*224*/
		.invoke(SSW, s_16_2, s_31_0, s_16_3, s_31_1)/*225*/
		.invoke(SSW, s_32_0, s_47_2, s_32_1, s_47_3)/*226*/
		.invoke(ADD_1, s_31_1, s_32_1, s_31_2, s_32_2)/*227*/
		.invoke(SSW, s_80_2, s_95_0, s_80_3, s_95_1)/*228*/
		.invoke(SSW, s_96_0, s_111_2, s_96_1, s_111_3)/*229*/
		.invoke(ADD_0, s_95_1, s_96_1, s_95_2, s_96_2)/*230*/
		.invoke(SSW, s_32_2, s_63_0, s_32_3, s_63_1)/*231*/
		.invoke(SSW, s_64_0, s_95_2, s_64_1, s_95_3)/*232*/
		.invoke(ADD_1, s_63_1, s_64_1, s_63_2, s_64_2)/*233*/
		.invoke(SSW, s_64_2, s_127_0, s_64_3, s_127_1)/*234*/
		.invoke(SSW, s_128_0, s_144_2, s_128_1, s_144_3)/*235*/
		.invoke(ADD_X, s_127_1, s_128_1, s_127_2, s_128_2)/*236*/
		.invoke(SSW, s_64_3, s_127_2, s_64_4, s_127_3)/*237*/
		.invoke(SSW, s_128_2, s_144_3, s_128_3, s_144_4)/*238*/
		.invoke(SWB1_6, s_63_2, s_64_4, s_63_3, s_64_5)/*239*/
		.invoke(SSW, s_32_3, s_63_3, s_32_4, s_63_4)/*240*/
		.invoke(SSW, s_64_5, s_95_3, s_64_6, s_95_4)/*241*/
		.invoke(SWB1_5, s_31_2, s_32_4, s_31_3, s_32_5)/*242*/
		.invoke(SSW, s_16_3, s_31_3, s_16_4, s_31_4)/*243*/
		.invoke(SSW, s_32_5, s_47_3, s_32_6, s_47_4)/*244*/
		.invoke(SWB0_5, s_95_4, s_96_2, s_95_5, s_96_3)/*245*/
		.invoke(SSW, s_80_3, s_95_5, s_80_4, s_95_6)/*246*/
		.invoke(SSW, s_96_3, s_111_3, s_96_4, s_111_4)/*247*/
		.invoke(SWB1_4, s_15_2, s_16_4, s_15_3, s_16_5)/*248*/
		.invoke(SSW, s_8_3, s_15_3, s_8_4, s_15_4)/*249*/
		.invoke(SSW, s_16_5, s_23_3, s_16_6, s_23_4)/*250*/
		.invoke(SWB0_4, s_47_4, s_48_2, s_47_5, s_48_3)/*251*/
		.invoke(SSW, s_40_3, s_47_5, s_40_4, s_47_6)/*252*/
		.invoke(SSW, s_48_3, s_55_3, s_48_4, s_55_4)/*253*/
		.invoke(SWB1_4, s_79_2, s_80_4, s_79_3, s_80_5)/*254*/
		.invoke(SSW, s_72_3, s_79_3, s_72_4, s_79_4)/*255*/
		.invoke(SSW, s_80_5, s_87_3, s_80_6, s_87_4)/*256*/
		.invoke(SWB0_4, s_111_4, s_112_2, s_111_5, s_112_3)/*257*/
		.invoke(SSW, s_104_3, s_111_5, s_104_4, s_111_6)/*258*/
		.invoke(SSW, s_112_3, s_119_3, s_112_4, s_119_4)/*259*/
		.invoke(SWB1_4, s_143_2, s_144_4, s_143_3, s_144_5)/*260*/
		.invoke(SSW, s_136_3, s_143_3, s_136_4, s_143_4)/*261*/
		.invoke(SSW, s_144_5, s_151_3, s_144_6, s_151_4)/*262*/
		.invoke(SWB1_3, s_7_2, s_8_4, s_7_3, s_8_5)/*263*/
		.invoke(SSW, s_4_3, s_7_3, s_4_4, s_7_4)/*264*/
		.invoke(SSW, s_8_5, s_11_3, s_8_6, s_11_4)/*265*/
		.invoke(SWB0_3, s_23_4, s_24_2, s_23_5, s_24_3)/*266*/
		.invoke(SSW, s_20_3, s_23_5, s_20_4, s_23_6)/*267*/
		.invoke(SSW, s_24_3, s_27_3, s_24_4, s_27_4)/*268*/
		.invoke(SWB1_3, s_39_2, s_40_4, s_39_3, s_40_5)/*269*/
		.invoke(SSW, s_36_3, s_39_3, s_36_4, s_39_4)/*270*/
		.invoke(SSW, s_40_5, s_43_3, s_40_6, s_43_4)/*271*/
		.invoke(SWB0_3, s_55_4, s_56_2, s_55_5, s_56_3)/*272*/
		.invoke(SSW, s_52_3, s_55_5, s_52_4, s_55_6)/*273*/
		.invoke(SSW, s_56_3, s_59_3, s_56_4, s_59_4)/*274*/
		.invoke(SWB1_3, s_71_2, s_72_4, s_71_3, s_72_5)/*275*/
		.invoke(SSW, s_68_3, s_71_3, s_68_4, s_71_4)/*276*/
		.invoke(SSW, s_72_5, s_75_3, s_72_6, s_75_4)/*277*/
		.invoke(SWB0_3, s_87_4, s_88_2, s_87_5, s_88_3)/*278*/
		.invoke(SSW, s_84_3, s_87_5, s_84_4, s_87_6)/*279*/
		.invoke(SSW, s_88_3, s_91_3, s_88_4, s_91_4)/*280*/
		.invoke(SWB1_3, s_103_2, s_104_4, s_103_3, s_104_5)/*281*/
		.invoke(SSW, s_100_3, s_103_3, s_100_4, s_103_4)/*282*/
		.invoke(SSW, s_104_5, s_107_3, s_104_6, s_107_4)/*283*/
		.invoke(SWB0_3, s_119_4, s_120_2, s_119_5, s_120_3)/*284*/
		.invoke(SSW, s_116_3, s_119_5, s_116_4, s_119_6)/*285*/
		.invoke(SSW, s_120_3, s_123_3, s_120_4, s_123_4)/*286*/
		.invoke(SWB1_3, s_135_2, s_136_4, s_135_3, s_136_5)/*287*/
		.invoke(SSW, s_132_3, s_135_3, s_132_4, s_135_4)/*288*/
		.invoke(SSW, s_136_5, s_139_3, s_136_6, s_139_4)/*289*/
		.invoke(SWB0_3, s_151_4, s_152_2, s_151_5, s_152_3)/*290*/
		.invoke(SSW, s_148_3, s_151_5, s_148_4, s_151_6)/*291*/
		.invoke(SSW, s_152_3, s_155_3, s_152_4, s_155_4)/*292*/
		.invoke(SWB1_2, s_3_2, s_4_4, s_3_3, s_4_5)/*293*/
		.invoke(SSW, s_2_2, s_3_3, s_2_3, s_3_4)/*294*/
		.invoke(SSW, s_4_5, s_5_2, s_4_6, s_5_3)/*295*/
		.invoke(SWB0_2, s_11_4, s_12_2, s_11_5, s_12_3)/*296*/
		.invoke(SSW, s_10_2, s_11_5, s_10_3, s_11_6)/*297*/
		.invoke(SSW, s_12_3, s_13_2, s_12_4, s_13_3)/*298*/
		.invoke(SWB1_2, s_19_2, s_20_4, s_19_3, s_20_5)/*299*/
		.invoke(SSW, s_18_2, s_19_3, s_18_3, s_19_4)/*300*/
		.invoke(SSW, s_20_5, s_21_2, s_20_6, s_21_3)/*301*/
		.invoke(SWB0_2, s_27_4, s_28_2, s_27_5, s_28_3)/*302*/
		.invoke(SSW, s_26_2, s_27_5, s_26_3, s_27_6)/*303*/
		.invoke(SSW, s_28_3, s_29_2, s_28_4, s_29_3)/*304*/
		.invoke(SWB1_2, s_35_2, s_36_4, s_35_3, s_36_5)/*305*/
		.invoke(SSW, s_34_2, s_35_3, s_34_3, s_35_4)/*306*/
		.invoke(SSW, s_36_5, s_37_2, s_36_6, s_37_3)/*307*/
		.invoke(SWB0_2, s_43_4, s_44_2, s_43_5, s_44_3)/*308*/
		.invoke(SSW, s_42_2, s_43_5, s_42_3, s_43_6)/*309*/
		.invoke(SSW, s_44_3, s_45_2, s_44_4, s_45_3)/*310*/
		.invoke(SWB1_2, s_51_2, s_52_4, s_51_3, s_52_5)/*311*/
		.invoke(SSW, s_50_2, s_51_3, s_50_3, s_51_4)/*312*/
		.invoke(SSW, s_52_5, s_53_2, s_52_6, s_53_3)/*313*/
		.invoke(SWB0_2, s_59_4, s_60_2, s_59_5, s_60_3)/*314*/
		.invoke(SSW, s_58_2, s_59_5, s_58_3, s_59_6)/*315*/
		.invoke(SSW, s_60_3, s_61_2, s_60_4, s_61_3)/*316*/
		.invoke(SWB1_2, s_67_2, s_68_4, s_67_3, s_68_5)/*317*/
		.invoke(SSW, s_66_2, s_67_3, s_66_3, s_67_4)/*318*/
		.invoke(SSW, s_68_5, s_69_2, s_68_6, s_69_3)/*319*/
		.invoke(SWB0_2, s_75_4, s_76_2, s_75_5, s_76_3)/*320*/
		.invoke(SSW, s_74_2, s_75_5, s_74_3, s_75_6)/*321*/
		.invoke(SSW, s_76_3, s_77_2, s_76_4, s_77_3)/*322*/
		.invoke(SWB1_2, s_83_2, s_84_4, s_83_3, s_84_5)/*323*/
		.invoke(SSW, s_82_2, s_83_3, s_82_3, s_83_4)/*324*/
		.invoke(SSW, s_84_5, s_85_2, s_84_6, s_85_3)/*325*/
		.invoke(SWB0_2, s_91_4, s_92_2, s_91_5, s_92_3)/*326*/
		.invoke(SSW, s_90_2, s_91_5, s_90_3, s_91_6)/*327*/
		.invoke(SSW, s_92_3, s_93_2, s_92_4, s_93_3)/*328*/
		.invoke(SWB1_2, s_99_2, s_100_4, s_99_3, s_100_5)/*329*/
		.invoke(SSW, s_98_2, s_99_3, s_98_3, s_99_4)/*330*/
		.invoke(SSW, s_100_5, s_101_2, s_100_6, s_101_3)/*331*/
		.invoke(SWB0_2, s_107_4, s_108_2, s_107_5, s_108_3)/*332*/
		.invoke(SSW, s_106_2, s_107_5, s_106_3, s_107_6)/*333*/
		.invoke(SSW, s_108_3, s_109_2, s_108_4, s_109_3)/*334*/
		.invoke(SWB1_2, s_115_2, s_116_4, s_115_3, s_116_5)/*335*/
		.invoke(SSW, s_114_2, s_115_3, s_114_3, s_115_4)/*336*/
		.invoke(SSW, s_116_5, s_117_2, s_116_6, s_117_3)/*337*/
		.invoke(SWB0_2, s_123_4, s_124_2, s_123_5, s_124_3)/*338*/
		.invoke(SSW, s_122_2, s_123_5, s_122_3, s_123_6)/*339*/
		.invoke(SSW, s_124_3, s_125_2, s_124_4, s_125_3)/*340*/
		.invoke(SWB1_2, s_131_2, s_132_4, s_131_3, s_132_5)/*341*/
		.invoke(SSW, s_130_2, s_131_3, s_130_3, s_131_4)/*342*/
		.invoke(SSW, s_132_5, s_133_2, s_132_6, s_133_3)/*343*/
		.invoke(SWB0_2, s_139_4, s_140_2, s_139_5, s_140_3)/*344*/
		.invoke(SSW, s_138_2, s_139_5, s_138_3, s_139_6)/*345*/
		.invoke(SSW, s_140_3, s_141_2, s_140_4, s_141_3)/*346*/
		.invoke(SWB1_2, s_147_2, s_148_4, s_147_3, s_148_5)/*347*/
		.invoke(SSW, s_146_2, s_147_3, s_146_3, s_147_4)/*348*/
		.invoke(SSW, s_148_5, s_149_2, s_148_6, s_149_3)/*349*/
		.invoke(SWB0_2, s_155_4, s_156_2, s_155_5, s_156_3)/*350*/
		.invoke(SSW, s_154_2, s_155_5, s_154_3, s_155_6)/*351*/
		.invoke(SSW, s_156_3, s_157_2, s_156_4, s_157_3)/*352*/
		.invoke(SWB1_1, s_1_1, s_2_3, s_1_2, s_2_4)/*353*/
		.invoke(SWB0_1, s_5_3, s_6_1, s_5_4, s_6_2)/*354*/
		.invoke(SWB1_1, s_9_1, s_10_3, s_9_2, s_10_4)/*355*/
		.invoke(SWB0_1, s_13_3, s_14_1, s_13_4, s_14_2)/*356*/
		.invoke(SWB1_1, s_17_1, s_18_3, s_17_2, s_18_4)/*357*/
		.invoke(SWB0_1, s_21_3, s_22_1, s_21_4, s_22_2)/*358*/
		.invoke(SWB1_1, s_25_1, s_26_3, s_25_2, s_26_4)/*359*/
		.invoke(SWB0_1, s_29_3, s_30_1, s_29_4, s_30_2)/*360*/
		.invoke(SWB1_1, s_33_1, s_34_3, s_33_2, s_34_4)/*361*/
		.invoke(SWB0_1, s_37_3, s_38_1, s_37_4, s_38_2)/*362*/
		.invoke(SWB1_1, s_41_1, s_42_3, s_41_2, s_42_4)/*363*/
		.invoke(SWB0_1, s_45_3, s_46_1, s_45_4, s_46_2)/*364*/
		.invoke(SWB1_1, s_49_1, s_50_3, s_49_2, s_50_4)/*365*/
		.invoke(SWB0_1, s_53_3, s_54_1, s_53_4, s_54_2)/*366*/
		.invoke(SWB1_1, s_57_1, s_58_3, s_57_2, s_58_4)/*367*/
		.invoke(SWB0_1, s_61_3, s_62_1, s_61_4, s_62_2)/*368*/
		.invoke(SWB1_1, s_65_1, s_66_3, s_65_2, s_66_4)/*369*/
		.invoke(SWB0_1, s_69_3, s_70_1, s_69_4, s_70_2)/*370*/
		.invoke(SWB1_1, s_73_1, s_74_3, s_73_2, s_74_4)/*371*/
		.invoke(SWB0_1, s_77_3, s_78_1, s_77_4, s_78_2)/*372*/
		.invoke(SWB1_1, s_81_1, s_82_3, s_81_2, s_82_4)/*373*/
		.invoke(SWB0_1, s_85_3, s_86_1, s_85_4, s_86_2)/*374*/
		.invoke(SWB1_1, s_89_1, s_90_3, s_89_2, s_90_4)/*375*/
		.invoke(SWB0_1, s_93_3, s_94_1, s_93_4, s_94_2)/*376*/
		.invoke(SWB1_1, s_97_1, s_98_3, s_97_2, s_98_4)/*377*/
		.invoke(SWB0_1, s_101_3, s_102_1, s_101_4, s_102_2)/*378*/
		.invoke(SWB1_1, s_105_1, s_106_3, s_105_2, s_106_4)/*379*/
		.invoke(SWB0_1, s_109_3, s_110_1, s_109_4, s_110_2)/*380*/
		.invoke(SWB1_1, s_113_1, s_114_3, s_113_2, s_114_4)/*381*/
		.invoke(SWB0_1, s_117_3, s_118_1, s_117_4, s_118_2)/*382*/
		.invoke(SWB1_1, s_121_1, s_122_3, s_121_2, s_122_4)/*383*/
		.invoke(SWB0_1, s_125_3, s_126_1, s_125_4, s_126_2)/*384*/
		.invoke(SWB1_1, s_129_1, s_130_3, s_129_2, s_130_4)/*385*/
		.invoke(SWB0_1, s_133_3, s_134_1, s_133_4, s_134_2)/*386*/
		.invoke(SWB1_1, s_137_1, s_138_3, s_137_2, s_138_4)/*387*/
		.invoke(SWB0_1, s_141_3, s_142_1, s_141_4, s_142_2)/*388*/
		.invoke(SWB1_1, s_145_1, s_146_3, s_145_2, s_146_4)/*389*/
		.invoke(SWB0_1, s_149_3, s_150_1, s_149_4, s_150_2)/*390*/
		.invoke(SWB1_1, s_153_1, s_154_3, s_153_2, s_154_4)/*391*/
		.invoke(SWB0_1, s_157_3, s_158_1, s_157_4, s_158_2)/*392*/
		.invoke(SWB1_0, s_0_0, s_1_2, FIFO_C_BUF[0], FIFO_C_BUF[1])/*393*/
		.invoke(SWB0_0, s_2_4, s_3_4, FIFO_C_BUF[2], FIFO_C_BUF[3])/*394*/
		.invoke(SWB1_0, s_4_6, s_5_4, FIFO_C_BUF[4], FIFO_C_BUF[5])/*395*/
		.invoke(SWB0_0, s_6_2, s_7_4, FIFO_C_BUF[6], FIFO_C_BUF[7])/*396*/
		.invoke(SWB1_0, s_8_6, s_9_2, FIFO_C_BUF[8], FIFO_C_BUF[9])/*397*/
		.invoke(SWB0_0, s_10_4, s_11_6, FIFO_C_BUF[10], FIFO_C_BUF[11])/*398*/
		.invoke(SWB1_0, s_12_4, s_13_4, FIFO_C_BUF[12], FIFO_C_BUF[13])/*399*/
		.invoke(SWB0_0, s_14_2, s_15_4, FIFO_C_BUF[14], FIFO_C_BUF[15])/*400*/
		.invoke(SWB1_0, s_16_6, s_17_2, FIFO_C_BUF[16], FIFO_C_BUF[17])/*401*/
		.invoke(SWB0_0, s_18_4, s_19_4, FIFO_C_BUF[18], FIFO_C_BUF[19])/*402*/
		.invoke(SWB1_0, s_20_6, s_21_4, FIFO_C_BUF[20], FIFO_C_BUF[21])/*403*/
		.invoke(SWB0_0, s_22_2, s_23_6, FIFO_C_BUF[22], FIFO_C_BUF[23])/*404*/
		.invoke(SWB1_0, s_24_4, s_25_2, FIFO_C_BUF[24], FIFO_C_BUF[25])/*405*/
		.invoke(SWB0_0, s_26_4, s_27_6, FIFO_C_BUF[26], FIFO_C_BUF[27])/*406*/
		.invoke(SWB1_0, s_28_4, s_29_4, FIFO_C_BUF[28], FIFO_C_BUF[29])/*407*/
		.invoke(SWB0_0, s_30_2, s_31_4, FIFO_C_BUF[30], FIFO_C_BUF[31])/*408*/
		.invoke(SWB1_0, s_32_6, s_33_2, FIFO_C_BUF[32], FIFO_C_BUF[33])/*409*/
		.invoke(SWB0_0, s_34_4, s_35_4, FIFO_C_BUF[34], FIFO_C_BUF[35])/*410*/
		.invoke(SWB1_0, s_36_6, s_37_4, FIFO_C_BUF[36], FIFO_C_BUF[37])/*411*/
		.invoke(SWB0_0, s_38_2, s_39_4, FIFO_C_BUF[38], FIFO_C_BUF[39])/*412*/
		.invoke(SWB1_0, s_40_6, s_41_2, FIFO_C_BUF[40], FIFO_C_BUF[41])/*413*/
		.invoke(SWB0_0, s_42_4, s_43_6, FIFO_C_BUF[42], FIFO_C_BUF[43])/*414*/
		.invoke(SWB1_0, s_44_4, s_45_4, FIFO_C_BUF[44], FIFO_C_BUF[45])/*415*/
		.invoke(SWB0_0, s_46_2, s_47_6, FIFO_C_BUF[46], FIFO_C_BUF[47])/*416*/
		.invoke(SWB1_0, s_48_4, s_49_2, FIFO_C_BUF[48], FIFO_C_BUF[49])/*417*/
		.invoke(SWB0_0, s_50_4, s_51_4, FIFO_C_BUF[50], FIFO_C_BUF[51])/*418*/
		.invoke(SWB1_0, s_52_6, s_53_4, FIFO_C_BUF[52], FIFO_C_BUF[53])/*419*/
		.invoke(SWB0_0, s_54_2, s_55_6, FIFO_C_BUF[54], FIFO_C_BUF[55])/*420*/
		.invoke(SWB1_0, s_56_4, s_57_2, FIFO_C_BUF[56], FIFO_C_BUF[57])/*421*/
		.invoke(SWB0_0, s_58_4, s_59_6, FIFO_C_BUF[58], FIFO_C_BUF[59])/*422*/
		.invoke(SWB1_0, s_60_4, s_61_4, FIFO_C_BUF[60], FIFO_C_BUF[61])/*423*/
		.invoke(SWB0_0, s_62_2, s_63_4, FIFO_C_BUF[62], FIFO_C_BUF[63])/*424*/
		.invoke(SWB1_0, s_64_6, s_65_2, FIFO_C_BUF[64], FIFO_C_BUF[65])/*425*/
		.invoke(SWB0_0, s_66_4, s_67_4, FIFO_C_BUF[66], FIFO_C_BUF[67])/*426*/
		.invoke(SWB1_0, s_68_6, s_69_4, FIFO_C_BUF[68], FIFO_C_BUF[69])/*427*/
		.invoke(SWB0_0, s_70_2, s_71_4, FIFO_C_BUF[70], FIFO_C_BUF[71])/*428*/
		.invoke(SWB1_0, s_72_6, s_73_2, FIFO_C_BUF[72], FIFO_C_BUF[73])/*429*/
		.invoke(SWB0_0, s_74_4, s_75_6, FIFO_C_BUF[74], FIFO_C_BUF[75])/*430*/
		.invoke(SWB1_0, s_76_4, s_77_4, FIFO_C_BUF[76], FIFO_C_BUF[77])/*431*/
		.invoke(SWB0_0, s_78_2, s_79_4, FIFO_C_BUF[78], FIFO_C_BUF[79])/*432*/
		.invoke(SWB1_0, s_80_6, s_81_2, FIFO_C_BUF[80], FIFO_C_BUF[81])/*433*/
		.invoke(SWB0_0, s_82_4, s_83_4, FIFO_C_BUF[82], FIFO_C_BUF[83])/*434*/
		.invoke(SWB1_0, s_84_6, s_85_4, FIFO_C_BUF[84], FIFO_C_BUF[85])/*435*/
		.invoke(SWB0_0, s_86_2, s_87_6, FIFO_C_BUF[86], FIFO_C_BUF[87])/*436*/
		.invoke(SWB1_0, s_88_4, s_89_2, FIFO_C_BUF[88], FIFO_C_BUF[89])/*437*/
		.invoke(SWB0_0, s_90_4, s_91_6, FIFO_C_BUF[90], FIFO_C_BUF[91])/*438*/
		.invoke(SWB1_0, s_92_4, s_93_4, FIFO_C_BUF[92], FIFO_C_BUF[93])/*439*/
		.invoke(SWB0_0, s_94_2, s_95_6, FIFO_C_BUF[94], FIFO_C_BUF[95])/*440*/
		.invoke(SWB1_0, s_96_4, s_97_2, FIFO_C_BUF[96], FIFO_C_BUF[97])/*441*/
		.invoke(SWB0_0, s_98_4, s_99_4, FIFO_C_BUF[98], FIFO_C_BUF[99])/*442*/
		.invoke(SWB1_0, s_100_6, s_101_4, FIFO_C_BUF[100], FIFO_C_BUF[101])/*443*/
		.invoke(SWB0_0, s_102_2, s_103_4, FIFO_C_BUF[102], FIFO_C_BUF[103])/*444*/
		.invoke(SWB1_0, s_104_6, s_105_2, FIFO_C_BUF[104], FIFO_C_BUF[105])/*445*/
		.invoke(SWB0_0, s_106_4, s_107_6, FIFO_C_BUF[106], FIFO_C_BUF[107])/*446*/
		.invoke(SWB1_0, s_108_4, s_109_4, FIFO_C_BUF[108], FIFO_C_BUF[109])/*447*/
		.invoke(SWB0_0, s_110_2, s_111_6, FIFO_C_BUF[110], FIFO_C_BUF[111])/*448*/
		.invoke(SWB1_0, s_112_4, s_113_2, FIFO_C_BUF[112], FIFO_C_BUF[113])/*449*/
		.invoke(SWB0_0, s_114_4, s_115_4, FIFO_C_BUF[114], FIFO_C_BUF[115])/*450*/
		.invoke(SWB1_0, s_116_6, s_117_4, FIFO_C_BUF[116], FIFO_C_BUF[117])/*451*/
		.invoke(SWB0_0, s_118_2, s_119_6, FIFO_C_BUF[118], FIFO_C_BUF[119])/*452*/
		.invoke(SWB1_0, s_120_4, s_121_2, FIFO_C_BUF[120], FIFO_C_BUF[121])/*453*/
		.invoke(SWB0_0, s_122_4, s_123_6, FIFO_C_BUF[122], FIFO_C_BUF[123])/*454*/
		.invoke(SWB1_0, s_124_4, s_125_4, FIFO_C_BUF[124], FIFO_C_BUF[125])/*455*/
		.invoke(SWB0_0, s_126_2, s_127_3, FIFO_C_BUF[126], FIFO_C_BUF[127])/*456*/
		.invoke(SWB1_0, s_128_3, s_129_2, FIFO_C_BUF[128], FIFO_C_BUF[129])/*457*/
		.invoke(SWB0_0, s_130_4, s_131_4, FIFO_C_BUF[130], FIFO_C_BUF[131])/*458*/
		.invoke(SWB1_0, s_132_6, s_133_4, FIFO_C_BUF[132], FIFO_C_BUF[133])/*459*/
		.invoke(SWB0_0, s_134_2, s_135_4, FIFO_C_BUF[134], FIFO_C_BUF[135])/*460*/
		.invoke(SWB1_0, s_136_6, s_137_2, FIFO_C_BUF[136], FIFO_C_BUF[137])/*461*/
		.invoke(SWB0_0, s_138_4, s_139_6, FIFO_C_BUF[138], FIFO_C_BUF[139])/*462*/
		.invoke(SWB1_0, s_140_4, s_141_4, FIFO_C_BUF[140], FIFO_C_BUF[141])/*463*/
		.invoke(SWB0_0, s_142_2, s_143_4, FIFO_C_BUF[142], FIFO_C_BUF[143])/*464*/
		.invoke(SWB1_0, s_144_6, s_145_2, FIFO_C_BUF[144], FIFO_C_BUF[145])/*465*/
		.invoke(SWB0_0, s_146_4, s_147_4, FIFO_C_BUF[146], FIFO_C_BUF[147])/*466*/
		.invoke(SWB1_0, s_148_6, s_149_4, FIFO_C_BUF[148], FIFO_C_BUF[149])/*467*/
		.invoke(SWB0_0, s_150_2, s_151_6, FIFO_C_BUF[150], FIFO_C_BUF[151])/*468*/
		.invoke(SWB1_0, s_152_4, s_153_2, FIFO_C_BUF[152], FIFO_C_BUF[153])/*469*/
		.invoke(SWB0_0, s_154_4, s_155_6, FIFO_C_BUF[154], FIFO_C_BUF[155])/*470*/
		.invoke(SWB1_0, s_156_4, s_157_4, FIFO_C_BUF[156], FIFO_C_BUF[157])/*471*/
		.invoke(SWB0_0, s_158_2, s_159_0, FIFO_C_BUF[158], FIFO_C_BUF[159])/*472*/
        .invoke<tapa::join, NUM_PES>(ResultBuff, FIFO_C_BUF, FIFO_C_ARB, num_rows_per_pe, num_tiles_c, rp_time)
		.invoke(Arbiter_C, FIFO_C_ARB, FIFO_C_AB, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(MM2S_C, c_in, FIFO_C_IN, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(Compute_C, FIFO_C_IN, FIFO_C_AB, FIFO_C_OUT, alpha, beta, num_rows_per_pe, rp_time)
        .invoke<tapa::join, NUM_C_CH>(S2MM_C, FIFO_C_OUT, c_out, num_rows_per_pe, rp_time);
}