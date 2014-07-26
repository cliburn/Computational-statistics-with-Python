#include <stdio.h>

int main(int argc, char* argv[])
{
    printf("Number of arguments = %d\n", argc);
    for (int i=0; i<argc; i++) {
        printf("%s %s", argv[i], i < (argc-1)? ", " : "\n");
    }
}