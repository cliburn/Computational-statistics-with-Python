/* fib.c */
int fib(int n) {
  if (n < 3) {
    return 1;
  } else {
    return fib(n-1) + fib(n-2);
  }
}
