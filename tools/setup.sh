#!/bin/sh

set -e

[ -n "$CI" ]

llvm_version=20

if [ $(uname) = Linux ]; then
  curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash -s
fi

if [ $(uname) = Darwin ]; then
  brew uninstall --ignore-dependencies zstd
  brew install -s zstd
fi

brew install llvm@$llvm_version

echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
