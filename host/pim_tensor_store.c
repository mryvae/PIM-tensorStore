#include "pim_tensor_store.h"

int64_t usec()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (((int64_t)tv.tv_sec) * 1000000) + tv.tv_usec;
}

void dpu_set_launch(struct dpu_set_t dpu_set, bool async)
{
    if(async) {
        DPU_ASSERT(dpu_launch(dpu_set, DPU_ASYNCHRONOUS));
    } else {
        DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    }
}

void dpu_set_wait(struct dpu_set_t dpu_set)
{
    DPU_ASSERT(dpu_sync(dpu_set));
}

void dpu_set_log_read(struct dpu_set_t dpu_set)
{
    struct dpu_set_t dpu;
    DPU_FOREACH(dpu_set, dpu)
    {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
}

void dpu_set_log_read_single_dpu(struct dpu_set_t dpu_set)
{
    struct dpu_set_t dpu;
    int dumped = 0;
    DPU_FOREACH(dpu_set, dpu)
    {
        if(!dumped) {
            DPU_ASSERT(dpu_log_read(dpu, stdout));
            dumped = 1;
        }
    }
}