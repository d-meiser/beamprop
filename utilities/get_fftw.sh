#!/bin/sh
THIRD_PARTY_DIR=`pwd`/third_party
FFTW_DIR=${THIRD_PARTY_DIR}/fftw
FFTW_VERSION=fftw-3.3.8
FFTW_TARBALL=${FFTW_VERSION}.tar.gz
FFTW_TARBALL_URL=http://www.fftw.org/$FFTW_TARBALL

if [ ! -e $FFTW_DIR/$FFTW_TARBALL ]; then
	mkdir -p $FFTW_DIR
	curl -o $FFTW_DIR/$FFTW_TARBALL $FFTW_TARBALL_URL
fi

if [ ! -e $FFTW_DIR/${FFTW_VERSION}/configure ]; then
	cd $FFTW_DIR
	tar xf $FFTW_TARBALL
	cd -
fi

if [ ! -e $THIRD_PARTY_DIR/lib/libfftw3.a ]; then
	cd $FFTW_DIR/$FFTW_VERSION
	CFLAGS='-ffast-math -O3' \
		./configure \
		--prefix=${THIRD_PARTY_DIR} \
		--enable-avx
	make -j2
	make check
	make install
	cd -
fi
