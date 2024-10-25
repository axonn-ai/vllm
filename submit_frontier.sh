#!/bin/bash

#for bs in 1 4 8 16 32; do
for bs in 1 4 8 16 32; do
	for pl in 128 256 512 1024 2048 4096; do
		set -x
		sbatch benchmarking_frontier.sh $bs $pl
		set +x
	done
done
