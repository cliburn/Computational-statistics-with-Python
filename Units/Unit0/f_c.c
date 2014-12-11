
long f_c(long N) {
    long s = 0;
    for (int i=0; i<N; i++) {
        s += i*i;
    }
    return s;
}