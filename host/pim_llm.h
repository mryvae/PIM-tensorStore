#ifndef _PIM_LLM_H
#define _PIM_LLM_H

#include "./msg/msg_comm.h"
#include "./mm/pim_direct_comm.h"
#include <stdint.h>
#include <sys/time.h>

int64_t usec();
void dpu_set_launch(struct dpu_set_t dpu_set);
void dpu_set_launch_async(struct dpu_set_t dpu_set);
void dpu_set_wait(struct dpu_set_t dpu_set);
void dpu_set_log_read(struct dpu_set_t dpu_set);
void dpu_set_log_read_single_dpu(struct dpu_set_t dpu_set);

#endif