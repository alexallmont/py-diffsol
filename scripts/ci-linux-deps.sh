#!/bin/bash

yum install -y llvm-devel
llvm-config --version
llvm-config --prefix
