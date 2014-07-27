#include <stdio.h>
#include <stdlib.h>

int main()
{
    // example 1
    typedef char* string;
    char *s[] = {"mary ", "had ", "a ", "little ", "lamb", NULL};
    for (char **sp = s; *sp != NULL; sp++) {
        printf("%s", *sp);
    }
    printf("\n");

    // example 2
    char *src = "abcde";
    char *dest = malloc(5); // char is always 1 byte by C99 definition
    
    char *p = src + 4;
    char *q = dest;
    while ((*q++ = *p--)); // put the string in src into dest in reverse order

    for (int i = 0; i < 5; i++) {
        printf("i = %d, src[i] = %c, dest[i] = %c\n", i, src[i], dest[i]);
    }
}