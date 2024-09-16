#!/bin/bash

# Check if argument N is provided
if [ -z "$1" ]; then
  echo "Usage: $0 N"
  exit 1
fi

# Assign the argument N
N=$1

for ((i=1; i<=N; i++))
do
  echo "Running experiment $i/$N"
  python scripts/sacred_run.py with interim_plot=False
done

echo "All experiments completed."
