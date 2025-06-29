#ifndef _TENSOR_GET_H
#define _TENSOR_GET_H

#include <mram.h>
#include "msg_block.h"
#include "mram_mm.h"

void tensor_get_prepare(msg_block_header *header_ptr);
void tensor_get_tasklets_run();
void tensor_get_merge();

#endif