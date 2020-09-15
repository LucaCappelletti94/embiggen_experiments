#!/bin/bash
#SBATCH -t 20:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres gpu:4
#SBATCH --qos=training
#SBATCH --gres-flags=enforce-binding
#SBATCH --mail-user=vida.ravanmehr@jax.org

#conda activate base

python -V
module load cuda10.1/blas/10.1.243
module load cuda10.1/toolkit/10.1.243

python /projects/robinson-lab/vidar/embiggen_august/embiggen/Skipgram_graph_embedding.py
#python /projects/robinson-lab/vidar/embiggen_august/embiggen/test_gpu.py
#nvidia-smi
