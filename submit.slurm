#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --time=00:01:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --job-name=Simple
#SBATCH --gres=gpu:1

#Select file to run
export file="main_binary.py"
export args="--model resnet_binary --dataset cifar10 --gpus 0"

#Select how logs get stored
mkdir $SLURM_JOB_ID
export debug_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"
export benchmark_logs="$SLURM_JOB_ID/job_$SLURM_JOB_ID.log"

# Load Modules
module load python/python-3.6.3-gcc-7.1.0
module load cuda/cuda-9.0

# Enter Working Directory
cd $SLURM_SUBMIT_DIR
# Create Log FIle
echo $SLURM_SUBMIT_DIR
echo "JobID: %SLURM_JOB_ID" >> $debug_logs
echo "Running on $SLURM_NODELIST" >> $debug_Logs
echo "Running on $SLURM_NPROCS processors." >> $debug_logs
echo "Current working directory is `pwd`" >> $debug_logs

# Module debugging information
module list >> $debug_logs

date >> $benchmark_logs
echo "ulimit -l: " >> $benchmark_logs
ulimit -l >> $benchmark_logs

#Run job
python $file $args
echo "Program is finished with exit coded $? at: `date`"
sleep 3

date >> $benchmark_logs
echo "ulimit -l" >> $benchmark_logs
ulimit -l >> $benchmark_logs

mv job.$SLURM_JOB_ID.err $SLURM_JOB_ID/
mv job.$SLURM_JOB_ID.out $SLURM_JOB_ID/
