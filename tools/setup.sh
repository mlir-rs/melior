#!/bin/sh

set -e

llvm_version=18

brew install llvm@$llvm_version z3

echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
echo LIBRARY_PATH=$(brew --prefix)/lib:$LIBRARY_PATH >>$GITHUB_ENV
