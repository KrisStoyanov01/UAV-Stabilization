#!/bin/bash --login
#$ -cwd               # Run the job in the current directory
#$ -l nvidia_v100=1

date
echo "Starting experiment number: $experiment_number."

# Load modules
module load apps/binapps/anaconda3/2021.11
module load libs/cuda
module load libs/intel-18.0/hdf5/1.10.5_mpi              # Intel 18.0.3 compiler, OpenMPI 4.0.1
module load apps/binapps/pytorch/0.4.1-36-gpu
#module load tools/env/proxy

#pip install gym==0.19.0 --user
#pip install torch==1.4.0 --user
#pip install pybullet --user
#pip install dotmap --user
#pip install scipy==1.5.4 --user

./bash_scripts/drone_3D.sh

echo "Experiment $experiment_number complete."
    