# CPU-initiated High IOPS GNN Training

We study if disk-based GNN training can leverage high random read performance (IOPS) of modern SSDs via CPU-initiated IO rather than GPU-initiated IO.

## System Requirements

The computational environment we tested is as follows.

|Part     |Specifications |
|---------|---------------|
|CPU 0/1  |2 of Intel Xeon Gold 6430 (32 cores/CPU, 2.10 GHz) |
|DRAM     |DDR5 4800 MHz 512 GB (32 GB × 8 ch./CPU) |
|GPU      |NVIDIA GeForce RTX 5070 (12 GB, PCIe 5.0 x16) |
|SSD      |19 of Kioxia CM7-V (6.4 TB, 2.45 MIOPS) |
|OS       |Ubuntu 24.04.4 LTS, Linux kernel 6.8.0  |
|SW       |CUDA 12.9, PyTorch 2.8|

## Setup

1. Set up a virtual environment

   ```
   python3 -m venv .venv/chignn
   source .venv/chignn/bin/activate
   ```
   
1. Build SPDK

   ```
   sudo apt install -y meson
   bash scripts/build_spdk.sh
   ```

1. Build the SPDK data writer

   ```
   pushd src
   make
   popd
   ```
   
1. Install PyTorch and related modules

   PyTorch 2.7 or newer is required to support Blackwell GPUs. We tested PyTorch 2.8.
   ```
   pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
   pip install cmake==3.31.4 cython==3.0.12 torchmetrics
   ```

1. Clone the modified DGL

   The following steps are for anonymity. They will be replaced by `git clone` in the future.
   The long one-liner is a replacement for `git submodule update --init --recursive` as the latter does not work for the anonymized repository.
   ```
   wget -O dgl.zip https://anonymous.4open.science/api/repo/dgl-7761/zip
   mkdir dgl
   pushd dgl
   unzip ../dgl.zip
   counter=0; hash_list=("bfad207b448480783a1f428ae3d93d87032d8349" "e2bdd3bee8cb6501558042633fa59144cc8b7f5f" "f8d7d77c06936315286eb55f8de22cd23c188571" "e0f1b88b8efcb24ffa0ec55eabb78fbe61e58ae7" "4c47ca200209550c5628c89803591f8a753c8181" "80090603e43f6ddc870cc42e1403dd0af07744cc" "428802d1a5634f96bcd0705fab379ff0113bcf13" "709ddec37ff87e6087097ed6e49526dac21dcbc9" "f7dcc1ea60819475dffd3a45059e16f04381bee7" "4454de4b878f31d41c5b7578fe6ca24bba5ea3f4" "8bd6bad750b2b0d90800c632cf18e8ee93ad72d7" "7d9e85b6b2e9bf501021f857f2f3cbe43bc37c85" "1115dad3ffa0994e3f43b693d9b9cc99944c64c1"); while read -r line; do if [[ $line =~ path[[:space:]]*=[[:space:]]*(.*) ]]; then path=${BASH_REMATCH[1]}; fi; if [[ $line =~ url[[:space:]]*=[[:space:]]*(.*) ]]; then url=${BASH_REMATCH[1]}; git clone --recursive $url $path; pushd $path; git fetch origin ${hash_list[$counter]}; git checkout ${hash_list[$counter]}; popd; ((counter++)); fi; done < .gitmodules
   popd
   ```

1. Build the modified DGL

   ```
   bash scripts/build_dgl.sh
   ```

1. Load the SPDK driver

   The following command unbind NVMe SSDs from the standard NVMe driver and bind them to the SPDK driver.
   
   **The data on those SSDs will be lost.** 
   ```
   ./spdk/scripts/setup.sh
   ```
   
   
## Run

1. Run on DRAM

   Run GNN training by placing graph data on the host DRAM.
   Upon running the following command for the first time, graph data (Papers100M in this case) will be downloaded under the directory specified for `--root`.
   ```
   cd python
   python train.py --dataset=ogbn-papers100M --root=<path_to_data_dir> --sample-mode=sample_neighbors_all --num-sampling-buffers=8 --num-threads-sample=12 --num-threads-fetch=16 
   ```

2. Run on SSDs via SPDK
   
   Write the graph feature data to the SSDs bound to the SPDK driver.
   ```
   sudo src/write_npy_to_disk <path_to_data_dir>/ogbn-papers100M-seeds/preprocessed/data/node-feat.npy 0 1 128 1 0
   ```
   Run GNN training by placing feature data on the SSDs.
   ```
   cd python
   python train.py --dataset=ogbn-papers100M --root=<path_to_data_dir> --sample-mode=sample_neighbors_all --num-sampling-buffers=8 --num-threads-sample=12 --num-threads-fetch=2 --disk-based-feature=spdk --fetch-mode=3 --num-contexts=192 
   ```
