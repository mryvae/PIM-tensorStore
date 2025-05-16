#ifndef _MSG_BLOCK_H
#define _MSG_BLOCK_H

#include "msg_tensor.h"

enum pim_op
{
    PIM_OP_GEMV_Q4_Q8 = 0,
    PIM_OP_GEMV_Q4_Q8_128,
    PIM_OP_TENSOR_ADD_FOR_TEST,
    PIM_OP_TENSOR_GET_FOR_TEST,

    PIM_OP_COUNT,
};

typedef ALIGN8 struct
{
    uint32_t op;
    pim_tensor_des src0; // weight
    pim_tensor_des src1; // input
    pim_tensor_des src2; // table_f32_f16
} msg_block_header;

typedef struct
{
    msg_block_header header;
    void *extra;
    uint32_t extra_size;
} msg_block_des;

void msg_block_builder_op_tensor_add_for_test(msg_block_des *msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1, int32_t num);
void msg_block_builder_op_tensor_get_for_test(msg_block_des *msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1);
void msg_block_builder_op_gemv_q4_q8(msg_block_des *msg, remote_ptr w, int32_t w_ne0, int32_t w_ne1, int32_t in_q_ne0, 
                                        void *in_q_data, uint32_t in_q_size, remote_ptr table_f32_f16);
void msg_block_builder_op_gemv_q4_q8_128(msg_block_des *msg, remote_ptr w, int32_t w_ne0, int32_t w_ne1, int32_t in_q_ne0,
                                         void *in_q_data, uint32_t in_q_size, remote_ptr table_f32_f16);
void msg_block_free(msg_block_des *msg);

#endif