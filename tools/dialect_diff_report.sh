#!/bin/sh

set -e

[ $# -eq 2 ]

diff_file=$1
run_url=$2

if [ -s $diff_file ]; then
  echo 'Generated dialect code differs from `main`.'
  echo '```diff'
  # Keep the report under the size limit of pull request comments.
  awk '(size += length + 1) <= 60000' $diff_file
  echo '```'
  echo "See [the job log]($run_url) for the full diff."
else
  echo 'Generated dialect code is identical to `main`.'
fi
