#include <stdio.h>

int main()
{
    int i = 2;
    int j = 3;
    int *p;
    int *q;
    *p = i;
    q = &j;
    printf("p  = %p\n", p);
    printf("*p = %d\n", *p);
    printf("&p = %p\n", &p);
    printf("q  = %p\n", q);
    printf("*q = %d\n", *q);
    printf("&q = %p\n", &q);
}