#ifndef _MSG_COMM_H
#define _MSG_COMM_H

#include <dpu.h>
#include "msg_buffer.h"

void msg_buffer_send(msg_buffer *builder, struct dpu_set_t dpu_set, bool async);
void msg_buffer_recv(struct dpu_set_t dpu_set, void **dst, uint32_t size_per_dpu, bool async);

#endif