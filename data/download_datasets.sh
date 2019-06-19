#!/usr/bin/env bash

DIST_DIR=./mnist
mkdir -p ${DIST_DIR}

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O ${DIST_DIR}/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O ${DIST_DIR}/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O ${DIST_DIR}/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O ${DIST_DIR}/t10k-labels-idx1-ubyte.gz

DIST_DIR=./fashion
mkdir -p ${DIST_DIR}

wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -O ${DIST_DIR}/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -O ${DIST_DIR}/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -O ${DIST_DIR}/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -O ${DIST_DIR}/t10k-labels-idx1-ubyte.gz
