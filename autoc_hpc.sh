#!/bin/bash
#PBS -N autovc-reimpl
#PBS -l select=1:ncpus=16:mem=128GB:ngpus=1:Qlist=ee:host=comp056
#PBS -l walltime=140:00:00
#PBS -m ae
#PBS -e output.err
#PBS -o output.out
#PBS -M 2086379@sun.ac.za

# make sure I'm the only one that can read my output
umask 0077
# create a temporary directory with the job ID as name in /scratch-small-local
TMP=/scratch-small-local/252671-hpc1-hpc/autovc_reimpl/ # E.g. 249926.hpc1.hpc
echo "Temporary work dir: ${TMP}"
cd ${TMP}
# Ensure miniconda is activated
echo "Activating conda"
source ../miniconda/bin/activate

# write my output to my new temporary work directory
echo "Creating environment"
conda activate py38
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TMP}/miniconda/lib/

echo "Environment created. Beginning training"
python train.py

# job done, copy everything back
echo "Copying from ${TMP}/ to ${PBS_O_WORKDIR}/"

# if the copy back succeeded, delete my temporary files
# [ $? -eq 0 ] && /bin/rm -rf ${TMP}
