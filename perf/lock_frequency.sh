#!/usr/bin/bash
sudo cpupower frequency-set -d 3.8GHz
sudo xpu-smi -d 0 -t 0 --frequencyrange 2400,2400
sudo xpu-smi -d 1 -t 0 --frequencyrange 2400,2400
sudo xpu-smi -d 2 -t 0 --frequencyrange 2400,2400
sudo xpu-smi -d 3 -t 0 --frequencyrange 2400,2400