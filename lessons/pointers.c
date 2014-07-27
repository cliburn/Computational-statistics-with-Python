#include <stdio.h>
#include <stdlib.h>

int main()
{
    int *ps = malloc(5 * sizeof(int));
    for (int i =0; i < 5; i++) {
        ps[i] = i + 10;
    }

    printf("%d, %d\n", *ps, ps[0]); // remmeber that *ptr is just a regular variable outside of a declaration, in this case, an int
    printf("%d, %d\n", *(ps+2), ps[2]); 
    printf("%d, %d\n", *(ps+4), *(&ps[4])); // * and & are inverses
}