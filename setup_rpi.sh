#!/bin/bash

echo "Optimizing Raspberry Pi 4 for face detection..."



echo "Setting CPU to performance mode..."
sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 2>/dev/null || echo "CPU governor not adjustable"



echo "Optimizing thread priorities..."
ulimit -r 50 2>/dev/null || echo "Real-time priority not available"

echo "Checking camera module..."
vcgencmd get_camera

echo "Available memory:"
free -h

echo "Current temperature:"
vcgencmd measure_temp

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing optimized packages..."
pip install --upgrade pip

pip install opencv-python-headless==4.8.1.78

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.3.202

pip install requests numpy

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

