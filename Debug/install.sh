
# Make sure we have cythongsl
wget ftp://ftp.gnu.org/gnu/gsl/gsl-latest.tar.gz
tar -xzf gsl-latest.tar.gz
cd gsl-1.16
# The next three lines seem to have an issue
pwd
./configure --prefix=/usr/local
make
make install