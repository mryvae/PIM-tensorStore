#include "util.h"

uint32_t align8(uint32_t s)
{
    if ((s & 0x7) == 0)
        return s;
    return ((s >> 3) + 1) << 3;
}

uint32_t align2(uint32_t s)
{
    if ((s & 0x1) != 0)
        return ((s >> 1) + 1) << 1;
    return s;
}
