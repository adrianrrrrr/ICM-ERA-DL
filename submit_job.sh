#!/bin/bash
#SBATCH -p gpi.compute
#SBATCH -J train_UNET
#SBATCH --mem=32gb
#SBATCH --gres=gpu:1
#SBATCH -e "logs/%x-%j.e"
#SBATCH -o "logs/%x-%j.o"
module load conda/2022.10
source .bashrc
conda activate erasdl_unet_cuda
python /home/usuaris/imatge/adrian.ramos/ICM-ERA-DL/main.py
