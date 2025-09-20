#!/bin/sh

set -e

[ -n "$CI" ]

llvm_version=20

if [ $(uname) = Linux ]; then
  curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash -s
fi

brew install llvm@$llvm_version pkgconf zstd

echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
