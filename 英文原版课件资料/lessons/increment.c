#include <stdio.h>
#include <stdlib.h>

int main()
{
    int x = 3, y;
    y = x++; // x is incremented and y takes the value of x before incrementation
    printf("x = %d, y = %d\n", x, y); 
    y = ++x; // x is incremented and y takes the value of x after incrementation
    printf("x = %d, y = %d\n", x, y); 
}