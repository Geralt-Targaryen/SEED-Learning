#!/bin/bash

for k in "linear" "rbf" "poly" "sigmoid"; do
	for C in '1e-5' '1e-4' '1e-3' '1e-2' '1e-1' '1' '10' '100'; do
		python baseline.py --C $C --kernel $k --save
	done
done
