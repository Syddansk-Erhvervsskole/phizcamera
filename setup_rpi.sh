#!/bin/bash

echo "Optimizing Raspberry Pi 4 for face detection..."



# CPU Governor to performance mode
echo "Setting CPU to performance mode..."
sudo sh -c 'echo performance > /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor' 2>/dev/null || echo "CPU governor not adjustable"



# Set thread priorities for better real-time performance
echo "Optimizing thread priorities..."
ulimit -r 50 2>/dev/null || echo "Real-time priority not available"

# Camera module optimization
echo "Checking camera module..."
vcgencmd get_camera

# Check available memory
echo "Available memory:"
free -h

# Check temperature (Raspberry Pi throttles at 80°C)
echo "Current temperature:"
vcgencmd measure_temp

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade required packages with optimizations
echo "Installing optimized packages..."
pip install --upgrade pip

# Install OpenCV with minimal dependencies for Raspberry Pi
pip install opencv-python-headless==4.8.1.78

# Install ultralytics with CPU optimizations
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.3.202

# Install other requirements
pip install requests numpy

# Set environment variables for better performance
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

echo "Optimization complete!"
echo ""
echo "Usage:"
echo "  For standard performance: python main.py"
echo "  For maximum performance:  python main_lite.py"
echo ""
echo "Tips for Raspberry Pi 4:"
echo "1. Ensure adequate cooling (fan/heatsinks)"
echo "2. Use a fast SD card (Class 10 or better)"
echo "3. Use a quality power supply (3A recommended)"
echo "4. Close unnecessary applications"
echo "5. Consider overclocking if properly cooled"
echo ""
echo "Monitor temperature with: watch vcgencmd measure_temp"
