#ifndef _GEMV_Q4_Q8_H
#define _GEMV_Q4_Q8_H

#include <mram.h>
#include <defs.h>
#include "../../host/msg/msg_block.h"
#include "../../host/msg/msg_tensor.h"
#include "../sto/mram_mm.h"
#include "../sto/g_vars.h"
#include "../util/g_mutex.h"

void gemv_q4_q8_prepare(msg_block_header *header_ptr);
void gemv_q4_q8_tasklets_run();
void gemv_q4_q8_merge();

#endif