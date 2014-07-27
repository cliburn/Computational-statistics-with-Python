#include <stdio.h>

void change_arg(int *p) {
    *p *= 2;
}
int main()
{
    int x = 5;
    change_arg(&x);
    printf("%d\n", x);
}