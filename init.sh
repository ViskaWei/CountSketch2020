source ~/.bashrc

# Where's my Python
source /usr/local/miniconda3/bin/activate torch
export PYTHONPATH=.:../pysynphot:../SciScript-Python/py3

# Where's my PFS 
export ROOT=~/cancerHH/AceCanZ


# Work around issues with saving weights when running on multiple threads
export HDF5_USE_FILE_LOCKING=FALSE

# Disable tensorflow deprecation warnings
export TF_CPP_MIN_LOG_LEVEL=2

# Enable more cores for numexpr
export NUMEXPR_MAX_THREADS=32


cd $ROOT

echo "NIPS is still possible!! "
echo "YOU CAN DO THIS VISKA"
