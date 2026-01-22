#!/bin/bash
fcc -Nclang -O3 -march=armv8.2-a+sve -o bench_hand_scheduled bench_hand_scheduled.c -lm
./bench_hand_scheduled
