%module apop
%{
extern double ftest(double n1, double n2, double n3, double n4);
%}
%include ftest.c