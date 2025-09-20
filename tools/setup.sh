#!/bin/sh

set -e

[ -n "$CI" ]

llvm_version=20

if [ $(uname) = Darwin ]; then
  brew reinstall zstd
  brew install llvm@$llvm_version

  echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
else
  curl -fsSL https://apt.llvm.org/llvm.sh | bash -s $llvm_version
fi
