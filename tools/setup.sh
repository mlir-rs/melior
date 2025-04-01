#!/bin/sh

set -e

llvm_version=20

brew install llvm@$llvm_version

echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
