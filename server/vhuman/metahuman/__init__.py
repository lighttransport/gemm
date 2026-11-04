"""MetaHuman-derived statistical priors for the vhuman/GNM pipeline.

Host-side modules (numpy/torch): launch, sampling, generate, manifest, mesh_io, register, binding, *_prior_fit,
compare_render. Scripts under ue_scripts/ run inside the Unreal Editor's Python (unreal + stdlib only).

Policy: only statistics derived from locally generated MetaHuman Creator output leave this pipeline (means, PCA
bases, parameter distributions, renders used as references). Raw Epic assets, model data, meshes and textures stay
under the dataset root outside the repository and are never redistributed. No Epic cloud service is used.
"""
import os
from pathlib import Path

# Linux defaults; every root can be overridden from the environment (the Windows worker sets them, see winjob.py).
WINDOWS = os.name == 'nt'
UE_ROOT = Path(os.environ.get('VHUMAN_UE_ROOT', r'C:\Program Files\Epic Games\UE_5.8' if WINDOWS else '/mnt/nvme02/local/ue'))
UE_EDITOR_CMD = UE_ROOT / ('Engine/Binaries/Win64/UnrealEditor-Cmd.exe' if WINDOWS else 'Engine/Binaries/Linux/UnrealEditor-Cmd')
UE_RHI = os.environ.get('VHUMAN_UE_RHI', 'dx12' if WINDOWS else 'vulkan')    # Polaris (RX 570): Vulkan SM6 rejected, SM5 crashes
MH_PLUGIN = UE_ROOT / 'Engine/Plugins/MetaHuman/MetaHumanCharacter'
PROJECT = Path(os.environ.get('VHUMAN_MH_PROJECT', '/mnt/nvme01/work/vhuman-mh/MHGen/MHGen.uproject'))
DATASET_ROOT = Path(os.environ.get('VHUMAN_MH_DATA', '/mnt/nvme01/data/vhuman-mh'))   # nvme02 was at 30 GB free when this was created
MODELS_ROOT = Path(os.environ.get('VHUMAN_MH_MODELS', '/mnt/nvme02/models/vhuman-mh'))
UE_SCRIPTS = Path(__file__).resolve().parent / 'ue_scripts'
