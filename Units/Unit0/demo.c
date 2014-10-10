
#include "limits.h"

long limit() {
    return LONG_MAX;
}

long overflow() {
    long x = LONG_MAX;
    return x+1;
}