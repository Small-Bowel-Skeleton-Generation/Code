# Small Bowel Skeleton Generation via Diffusion Learning Based on Topology-Constrained Simulation

This repository provides the implementation for learning to generate 3D small-bowel skeletons using diffusion models with topology-constrained simulation. It includes data normalization utilities, a VAE for representation learning, and conditional diffusion models trained in low- and high-resolution stages for final generation.

## Environment Setup

- We recommend creating a fresh Conda environment (Python ≥ 3.8):
  - conda create -n tree-diffusion python=3.9 -y
  - conda activate tree-diffusion
- Install project dependencies:
  - pip install -r requirements.txt

Notes
- Please install a PyTorch build compatible with your CUDA version. The file <mcfile name="requirements.txt" path="<path-to-your-code>/requirements.txt"></mcfile> lists core packages; adjust torch/torchvision/torchaudio as needed.
- OCNN requires separate installation steps depending on your platform and CUDA. Refer to OCNN’s official instructions.
- Optional rendering utilities (pyrender/pyglet/pyrr/PyOpenGL) may require system OpenGL support and appropriate GPU drivers.

## Data Preparation and Preprocessing

The project expects two primary data types:
- Skeleton curve data (3D point sequences) for training the VAE.
- Mask volumes used as conditioning inputs for the diffusion model.

We provide utilities to normalize both:

1) Normalize skeleton curve point clouds
- Script: <mcfile name="repair_mesh.py" path="<path-to-your-code>/tools/repair_mesh.py"></mcfile>
- Supported modes: process_curve, sample_sdf, generate_dataset (runs process_curve then sample_sdf)
- Example commands:
  - python tools/repair_mesh.py --run process_curve --root_folder <ROOT_DATA_DIR> --curve_data_folder <RAW_CURVE_MAT_DIR> --num_samples 4000
  - python tools/repair_mesh.py --run sample_sdf --root_folder <ROOT_DATA_DIR> --sdf_size 128
  - python tools/repair_mesh.py --run generate_dataset --root_folder <ROOT_DATA_DIR> --curve_data_folder <RAW_CURVE_MAT_DIR> --num_samples 4000 --sdf_size 128

After completion, normalized curve samples will be stored under <ROOT_DATA_DIR>/dataset/<sample_id>/ as curve.npz containing points/tangents/normals/binormals/orders.

2) Normalize mask condition volumes
- Script: <mcfile name="mask_preprocess.py" path="<path-to-your-code>/tools/mask_preprocess.py"></mcfile>
- Supports single file or folder input; outputs .mat files with a boolean 3D array (key: mask). Default cubic size is 128.
- Example commands:
  - Single .nii.gz:
    - python tools/mask_preprocess.py -i /path/to/mask.nii.gz -o /path/to/output_dir -s 128
  - Folder of .nii.gz files:
    - python tools/mask_preprocess.py -i /path/to/nii_folder -o /path/to/output_dir -s 128

## End-to-End Workflow

Recommended order of operations:
1. Normalize skeleton curve data using tools/repair_mesh.py
2. Normalize mask condition data using tools/mask_preprocess.py
3. Train the VAE
4. Train the conditional diffusion model in low-resolution (lr) and then high-resolution (hr) stages
5. Run generation with the trained models

### 1) VAE Training and Generation
- Script: <mcfile name="run_snet_vae_skelet.sh" path="<path-to-your-code>/scripts/run_snet_vae_skelet.sh"></mcfile>
- Usage:
  - bash scripts/run_snet_vae_skelet.sh <mode> <category> [gpu_ids]
  - Arguments:
    - mode: train | generate | inference_vae
    - category: e.g., skeleton7
    - gpu_ids: e.g., 0 or 0,1,2 (optional)
- Examples:
  - Training: bash scripts/run_snet_vae_skelet.sh train skeleton7 0
  - Generation: bash scripts/run_snet_vae_skelet.sh generate skeleton7 0

This script invokes the main training entry point <mcfile name="train.py" path="<path-to-your-code>/train.py"></mcfile>. Please edit the script to set correct paths for checkpoints and logs (e.g., VQ_CKPT, CODE_BASE_DIR, LOGS_DIR).

### 2) Conditional Diffusion (LR/HR) Training and Generation
- Script: <mcfile name="run_snet_cond_skelet.sh" path="<path-to-your-code>/scripts/run_snet_cond_skelet.sh"></mcfile>
- Usage:
  - bash scripts/run_snet_cond_skelet.sh <mode> <stage_flag> <category> [gpu_ids]
  - Arguments:
    - mode: train | generate
    - stage_flag: lr | hr (low-/high-resolution stages)
    - category: e.g., skeleton7, skeleton7_rifle
    - gpu_ids: e.g., 0 or 0,1,2 (optional)
- Examples:
  - LR training: bash scripts/run_snet_cond_skelet.sh train lr skeleton7 0
  - HR training: bash scripts/run_snet_cond_skelet.sh train hr skeleton7 0
  - Generation: bash scripts/run_snet_cond_skelet.sh generate hr skeleton7 0

Please update the script headers to match your environment, including dataset roots, conditional data directories (COND_DIR_TRAIN / COND_DIR_GENERATE), pretrained checkpoints (PRETRAIN_CKPT / CKPT_GENERATE), VQ_CKPT, and base code/log directories.

## Configuration Notes
- The main training entry point is <mcfile name="train.py" path="<path-to-your-code>/train.py"></mcfile>. Command-line options are defined in <mcfile name="base_options.py" path="<path-to-your-code>/options/base_options.py"></mcfile>.
- YAML configuration files under configs/ specify dataset/model settings for different stages. During training, the active YAML files are copied into the experiment’s log directory for reproducibility.
- Logs, checkpoints, and visual artifacts are stored under logs/<experiment_name>/.

## Logs and Checkpoints (Download)

Pre-collected logs (including checkpoints, copied YAML configs, and sample results) are available for download:

- Google Drive: https://drive.google.com/drive/folders/1j3HB0JHNw9uWGZ_L78Axk5Ij19CpjgZU?usp=drive_link

After downloading, place the extracted logs folder under the project root (i.e., <project_root>/logs) if you wish to reuse the provided checkpoints/config snapshots.

## Pre-Run Checklist
- PyTorch is installed with a CUDA build matching your GPU drivers.
- All Python dependencies from <mcfile name="requirements.txt" path="<path-to-your-code>/requirements.txt"></mcfile> are installed (some packages may require system libraries, e.g., OpenGL, VC++/X11, etc.).
- Raw skeleton .mat curves and mask .nii.gz volumes are prepared and preprocessed into curve.npz and mask.mat, respectively.
- Paths and checkpoints have been updated in scripts under scripts/.

Windows Note
- The provided scripts are Bash (.sh). On Windows, consider running them via WSL or convert the launch commands for PowerShell. Alternatively, you can run <mcfile name="train.py" path="<path-to-your-code>/train.py"></mcfile> directly with the same command-line arguments.
