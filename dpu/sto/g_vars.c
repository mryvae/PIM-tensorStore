#include "g_vars.h"

__host msg_buffer_header g_msg_header;
__host char *g_msg_cache_ptr = NULL;
__host uint32_t *g_blocks_offset_ptr = NULL;

// __host int16_t mul_table_int4_int8[1 << 4][1 << 8];
// __mram float table_f32_f16[1 << 16];