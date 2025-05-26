#include "gemv_q4_q8.h"
#include <assert.h>
#include <stdio.h>

#define OP_GEMV_DEBUG_PRINT 0
#ifndef SEGMENT_PER_ROW
#define SEGMENT_PER_ROW 4
#endif

static __mram_ptr pim_half_int8 *table_half_int8 = NULL;
static pim_tensor_des *src0 = NULL; // weight
static pim_tensor_des *src1 = NULL; // input
static pim_tensor_des *src2 = NULL; // table_half_int8

static __mram_ptr pim_block_q4_0 *weight_data_addr = NULL;
static pim_block_q8_0 *input_cache_addr = NULL;
static float *sumf = NULL;
static int64_t *row_ms = NULL;
static uint16_t *row_ks = NULL;

static pim_half_int8 lookup_fp16_to_half_int8(uint16_t f)
{
    uint16_t s;
    memcpy(&s, &f, sizeof(uint16_t));
    uint16_t alignedOffset;
    __dma_aligned pim_half_int8 temp[8];

    alignedOffset = s & 0xfff8;
    mram_read((__mram_ptr void const *)(table_half_int8 + alignedOffset), temp, sizeof(pim_half_int8) * 8);
    return temp[s & 0x7];
}

#define FP16_TO_HALF_INT8(x) lookup_fp16_to_half_int8(x)

void gemv_q4_q8_int_only_prepare(msg_block_header *header_ptr)
{
    src0 = &header_ptr->src0;
    assert(src0->ptr.dpu_id == ALL_DPU);
    weight_data_addr = (__mram_ptr pim_block_q4_0 *)(DPU_MRAM_HEAP_POINTER + src0->ptr.dpu_addr);
    src1 = &header_ptr->src1;
    input_cache_addr = (pim_block_q8_0 *)((char *)header_ptr + sizeof(msg_block_header));
    src2 = &header_ptr->src2;
    table_half_int8 = (__mram_ptr pim_half_int8 *)(DPU_MRAM_HEAP_POINTER + src2->ptr.dpu_addr);

#if OP_GEMV_DEBUG_PRINT
    printf("src0.dpu_addr: %d, ne0: %d, ne1: %d\n", src0->ptr.dpu_addr,
           src0->ne[0], src0->ne[1]);
    printf("src1.dpu_addr: %d, ne0: %d, ne1: %d\n", src1->ptr.dpu_addr,
           src1->ne[0], src1->ne[1]);
    printf("src2.dpu_addr: %d\n", src2->ptr.dpu_addr);
#endif

    assert(src0->ne[0] == src1->ne[0]);
    assert(src1->ne[1] == 1 && "Only support vector as input.");
    assert(src0->type == PIM_TYPE_Q4_0 && "Only support Q4_0 weight.");
    assert(src1->type == PIM_TYPE_Q8_0 && "Only support Q8_0 input.");

    sumf = (float *)mem_alloc(sizeof(float) * src0->ne[1]);
    row_ms = (int64_t *)mem_alloc(sizeof(int64_t) * src0->ne[1]);
    row_ks = (uint16_t *)mem_alloc(sizeof(uint16_t) * src0->ne[1]);
    memset(sumf, 0, sizeof(float) * src0->ne[1]);
    memset(row_ms, 0, sizeof(int64_t) * src0->ne[1]);
    memset(row_ks, 0, sizeof(uint16_t) * src0->ne[1]);
#if OP_GEMV_DEBUG_PRINT
    printf("sumf0: %f, sumf1: %f, sumf2: %f\n", sumf[0], sumf[1], sumf[2]);
#endif
}

void gemv_q4_q8_int_only_tasklets_run_stage1()
{
    unsigned int tasklet_id = me();

    uint32_t segments_num = src0->ne[1] * SEGMENT_PER_ROW;
    int32_t segment_start = BLOCK_LOW(tasklet_id, NR_TASKLETS, segments_num);
    int32_t segment_end = BLOCK_HIGH(tasklet_id, NR_TASKLETS, segments_num);

#if OP_GEMV_DEBUG_PRINT
    if (tasklet_id == 0)
    {
        printf("segments_num: %d, segment_start: %d, segment_end: %d\n", segments_num,
               segment_start, segment_end);
    }
#endif

    if (segment_start > segment_end)
    {
        return;
    }
    // assert(segment_start <= segment_end && "There are not enough segments to allocate to the tasklets");

    int qk = QK8_0;
    uint32_t nb = src1->ne[0] / QK8_0;
#if OP_GEMV_DEBUG_PRINT
    if (tasklet_id == 0)
    {
        printf("input0: %d %d, input_last: %d %d\n", input_cache_addr[0].qs[0],
               input_cache_addr[0].qs[1], input_cache_addr[nb - 1].qs[0], input_cache_addr[nb - 1].qs[1]);
    }
#endif
    assert(nb % 2 == 0);
    uint32_t segment_nb_size_max = nb / SEGMENT_PER_ROW + 1 + 1;
    char *pweight_cache_align8 = (char *)mem_alloc(align8(sizeof(pim_block_q4_0) * segment_nb_size_max) + 8);
    pim_block_q4_0 *pweight_cache = NULL;

    for (int k = segment_start; k <= segment_end; ++k)
    {
        int32_t row_idx = k / SEGMENT_PER_ROW;
        int32_t segment_idx = k % SEGMENT_PER_ROW;
        int32_t segment_nb_start = BLOCK_LOW(segment_idx, SEGMENT_PER_ROW, nb) / 2 * 2;
        int32_t segment_nb_size = BLOCK_LOW(segment_idx + 1, SEGMENT_PER_ROW, nb) / 2 * 2 - segment_nb_start;
        if (segment_nb_size <= 0)
        {
            continue;
        }
        __mram_ptr pim_block_q4_0 *pweight = weight_data_addr + row_idx * nb + segment_nb_start;
        uintptr_t pweight_uint32 = (uintptr_t)pweight;
        assert(pweight_uint32 % 4 == 0);
        __mram_ptr char *pweight_align8 = (__mram_ptr char *)pweight - pweight_uint32 % 8;
        mram2wram(pweight_align8, pweight_cache_align8, align8(sizeof(pim_block_q4_0) * segment_nb_size) + 8);
        pweight_cache = (pim_block_q4_0 *)(pweight_cache_align8 + pweight_uint32 % 8);

        pim_block_q8_0 *pinput_cache = input_cache_addr + segment_nb_start;
#if OP_GEMV_DEBUG_PRINT
        printf("pweight: %p, pweight_uint32: %d, pweight_align8: %p, pweight_cache_align8: %p, pweight_cache: %p, pinput_cache: %p\n",
               pweight, pweight_uint32, pweight_align8, pweight_cache_align8, pweight_cache, pinput_cache);
#endif
        float sum = 0;
        for (int i = 0; i < segment_nb_size; i++)
        {
            int32_t sumi = 0;
            for (int j = 0; j < qk / 2; ++j)
            {
                const int8_t v0 = (pweight_cache[i].qs[j] & 0x0F) - 8;
                const int8_t v1 = (pweight_cache[i].qs[j] >> 4) - 8;

                sumi += mul_table_int4_int8[v0 + 8][pinput_cache[i].qs[j] - INT8_MIN] +
                        mul_table_int4_int8[v1 + 8][pinput_cache[i].qs[j + qk / 2] - INT8_MIN];
            }

            pim_half_int8 half1 = FP16_TO_HALF_INT8(pweight_cache[i].d);
            pim_half_int8 half2 = FP16_TO_HALF_INT8(pinput_cache[i].d);
            int32_t block_m = sumi * half1.m * half2.m;
            uint16_t block_k = (uint16_t)half1.k + half2.k;

            buckets_mutex_lock(row_idx);
            uint16_t mk = block_k;
            int64_t block_m_int64 = (int64_t)block_m;
            if (mk < row_ks[row_idx])
            {
                mk = row_ks[row_idx];
            }
            row_ms[row_idx] = (row_ms[row_idx] << (mk - row_ks[row_idx])) + (block_m_int64 << (mk - block_k));
            row_ks[row_idx] = mk;
            buckets_mutex_unlock(row_idx);
        }
    }
}

void gemv_q4_q8_int_only_tasklets_run_stage2()
{
    unsigned int tasklet_id = me();
    for (int i = tasklet_id; i < src0->ne[1]; i += NR_TASKLETS)
    {
        sumf[i] = (float)row_ms[i] / ((int64_t)1 << row_ks[i]);
    }
}

void gemv_q4_q8_int_only_merge()
{
#if OP_GEMV_DEBUG_PRINT
    printf("sumf0: %f, sumf1: %f, sumf2: %f\n", sumf[0], sumf[1], sumf[2]);
#endif
    wram2mram((__mram_ptr void *)RESULT_BUFFER_ADDR, sumf, sizeof(float) * src0->ne[1]);
}