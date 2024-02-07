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