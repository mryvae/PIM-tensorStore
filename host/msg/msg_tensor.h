#ifndef _MSG_TENSOR_H
#define _MSG_TENSOR_H

#include "../mm/pim_mm.h"
#include <string.h>
#include <stdlib.h>

enum pim_type
{
    PIM_TYPE_Q4_0 = 2,
    PIM_TYPE_Q4_0_128 = 3,
    PIM_TYPE_Q8_0 = 8,
    PIM_TYPE_Q8_0_128 = 9
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

#ifndef QK8_0_128
#define QK8_0_128 128
#endif

typedef uint16_t pim_half;

// float d = m / (2^k)
typedef struct
{
    int8_t m;
    uint8_t k;
} pim_half_int8;

typedef struct
{
    pim_half d;       // delta
    int8_t qs[QK8_0]; // quants
} pim_block_q8_0;

typedef struct
{
    pim_half d;           // delta
    int8_t qs[QK8_0_128]; // quants
} pim_block_q8_0_128;

#ifndef QK4_0
#define QK4_0 32
#endif

#ifndef QK4_0_128
#define QK4_0_128 128
#endif

typedef struct
{
    pim_half d;            // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} pim_block_q4_0;

typedef struct
{
    pim_half d;                // delta
    uint8_t qs[QK4_0_128 / 2]; // nibbles / quants
} pim_block_q4_0_128;

typedef struct
{
    int64_t ne[2];
    int64_t nb[2];
    pim_block_q4_0_128 *data;
} pim_tensor_q4_0_128;

pim_tensor_q4_0_128* pim_new_tensor_q4_0_128(int64_t ne0, int64_t ne1);
void pim_free_tensor_q4_0_128(pim_tensor_q4_0_128 *tensor);

#endif