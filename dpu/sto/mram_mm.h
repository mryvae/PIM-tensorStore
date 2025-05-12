#ifndef _MRAM_MM_H
#define _MRAM_MM_H

#include <mram.h>
#include <alloc.h>
#include "pim_mm.h"

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

#define MESSAGE_BUFFER_ADDR (DPU_MRAM_HEAP_POINTER)
#define RESULT_BUFFER_ADDR (MESSAGE_BUFFER_ADDR + MESSAGE_BUFFER_SIZE)
#define FREE_STORAGE_ADDR (RESULT_BUFFER_ADDR + RESULT_BUFFER_SIZE)

int mram2wram(__mram_ptr void *pmram,void *pwram,uint32_t size);
int wram2mram(__mram_ptr void *pmram,void *pwram,uint32_t size);

#endif