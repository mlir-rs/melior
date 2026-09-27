#!/bin/sh

set -e

[ -n "$CI" ]

llvm_version=23

case $RUNNER_OS in
Linux | macOS)
  if [ $(uname) = Linux ]; then
    curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh | bash -s
  fi

  brew install llvm@$llvm_version

  echo PATH=$(brew --prefix)/opt/llvm@$llvm_version/bin:$PATH >>$GITHUB_ENV
  ;;
Windows)
  llvm_prefix=$RUNNER_TEMP/llvm

  # The zlib and libxml2 development packages provide import libraries listed by `llvm-config`.
  $CONDA/Scripts/conda.exe create -y \
    -c conda-forge \
    -p $llvm_prefix \
    --override-channels \
    mlir=$llvm_version zlib libxml2-devel

  # `llvm-config` lists the zstd import library by its DLL name.
  # https://github.com/llvm/llvm-project/issues/134025
  (
    cd $llvm_prefix/Library/lib
    cp zstd.lib zstd.dll.lib
  )

  echo $llvm_prefix/Library/bin >>$GITHUB_PATH
  ;;
esac
