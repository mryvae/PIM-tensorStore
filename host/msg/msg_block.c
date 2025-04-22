#include "msg_block.h"
#include <assert.h>
#include <stdio.h>

void msg_block_builder_op_tensor_add_for_test(msg_block_des *msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1, int32_t num)
{
    msg->header.op = PIM_OP_TENSOR_ADD_FOR_TEST;
    msg->header.src0.type = 0; // int32_t
    msg->header.src0.ne[0] = ne0;
    msg->header.src0.ne[1] = ne1;
    msg->header.src0.ptr = target_tensor;

    msg->extra = malloc(sizeof(int));

    memcpy(msg->extra, &num, sizeof(num));
    msg->extra_size = sizeof(num);
}

void msg_block_builder_op_tensor_get_for_test(msg_block_des *msg, remote_ptr target_tensor, int32_t ne0, int32_t ne1)
{
    msg->header.op = PIM_OP_TENSOR_GET_FOR_TEST;
    msg->header.src0.type = 0; // int32_t
    msg->header.src0.ne[0] = ne0;
    msg->header.src0.ne[1] = ne1;
    msg->header.src0.ptr = target_tensor;

    msg->extra = NULL;
    msg->extra_size = 0;
}

void msg_block_builder_op_gemv_q4_q8(msg_block_des *msg, remote_ptr w, int32_t w_ne0, int32_t w_ne1, int32_t in_q_ne0, 
                                        void *in_q_data, uint32_t in_q_size, remote_ptr table_f32_f16){
    assert(w_ne0 == in_q_ne0);
    
    msg->header.op = PIM_OP_GEMV_Q4_Q8;
    msg->header.src0.type = PIM_TYPE_Q4_0; 
    msg->header.src0.ne[0] = w_ne0;
    msg->header.src0.ne[1] = w_ne1;
    msg->header.src0.ptr = w;

    msg->header.src1.type = PIM_TYPE_Q8_0;
    msg->header.src1.ne[0] = in_q_ne0;
    msg->header.src1.ne[1] = 1;
    msg->header.src1.ptr.dpu_id = ALL_DPU;
    msg->header.src1.ptr.dpu_addr = DPU_EXTRA_PTR;
    
    msg->header.src2.ptr = table_f32_f16;
    // printf("%d\n",msg->header.src2.ptr.dpu_addr);

    msg->extra = in_q_data;
    msg->extra_size = in_q_size;
}

void msg_block_free(msg_block_des *msg)
{
    if (msg && msg->extra)
    {
        free(msg->extra);
    }
}