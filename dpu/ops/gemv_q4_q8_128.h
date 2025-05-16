#ifndef _GEMV_Q4_Q8_128_H
#define _GEMV_Q4_Q8_128_H

#include <mram.h>
#include <defs.h>
#include "msg_block.h"
#include "msg_tensor.h"
#include "mram_mm.h"
#include "g_vars.h"
#include "g_mutex.h"

void gemv_q4_q8_128_prepare(msg_block_header *header_ptr);
void gemv_q4_q8_128_tasklets_run();
void gemv_q4_q8_128_merge();

#endif