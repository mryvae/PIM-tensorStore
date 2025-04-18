#ifndef _MSG_TENSOR_H
#define _MSG_TENSOR_H

#include "../mm/pim_mm.h"
#include <string.h>
#include <stdlib.h>

enum pim_type
{
    PIM_TYPE_Q4_0 = 2,
    PIM_TYPE_Q8_0 = 8
};

typedef ALIGN8 struct
{
    int32_t type;
    int32_t ne[2];
    remote_ptr ptr;
} pim_tensor_des;

#ifndef QK8_0
#define QK8_0 32
#endif

typedef uint16_t pim_half;

typedef struct
{
    pim_half d;       // delta
    int8_t qs[QK8_0]; // quants
} pim_block_q8_0;

#ifndef QK4_0
#define QK4_0 32
#endif

typedef struct
{
    pim_half d;            // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} pim_block_q4_0;

#endif