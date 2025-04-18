#ifndef _PIM_API_H
#define _PIM_API_H

#include "./msg/msg_comm.h"
#include "./mm/pim_direct_comm.h"

void dpu_set_launch(struct dpu_set_t dpu_set)
{
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
}

void dpu_set_log_read(struct dpu_set_t dpu_set)
{
    struct dpu_set_t dpu;
    DPU_FOREACH(dpu_set, dpu)
    {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
    }
}

#endif