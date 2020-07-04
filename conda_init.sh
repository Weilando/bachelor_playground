# Script to set up machines (Linux, x86) at AWS to properly run the playground.
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# Install dependencies into environment bachelor_playground
conda create --name bachelor_playground --file conda_specs.txt
conda activate bachelor_playground
