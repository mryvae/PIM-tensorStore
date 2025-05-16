#include "msg_tensor.h"
#include <assert.h>

pim_tensor_q4_0_128 *pim_new_tensor_q4_0_128(int64_t ne0, int64_t ne1)
{
    static const int qk = QK4_0_128;
    assert(ne0 % qk == 0);
    assert(ne1 > 0);

    pim_tensor_q4_0_128 *tensor = (pim_tensor_q4_0_128 *)malloc(sizeof(pim_tensor_q4_0_128));
    tensor->ne[0] = ne0;
    tensor->ne[1] = ne1;
    tensor->nb[0] = sizeof(pim_block_q4_0_128);
    tensor->nb[1] = sizeof(pim_block_q4_0_128) * ne0 / qk;
    tensor->data = (pim_block_q4_0_128 *)malloc(sizeof(pim_block_q4_0_128) * ne0 / qk * ne1);
    return tensor;
}

void pim_free_tensor_q4_0_128(pim_tensor_q4_0_128 *tensor)
{
    if (!tensor)
    {
        return;
    }
    free(tensor->data);
    free(tensor);
}