"""MetaHuman-derived statistical priors for the vhuman/GNM pipeline.

Host-side modules (numpy/torch): launch, sampling, generate, manifest, mesh_io, register, binding, *_prior_fit,
compare_render. Scripts under ue_scripts/ run inside the Unreal Editor's Python (unreal + stdlib only).

Policy: only statistics derived from locally generated MetaHuman Creator output leave this pipeline (means, PCA
bases, parameter distributions, renders used as references). Raw Epic assets, model data, meshes and textures stay
under the dataset root outside the repository and are never redistributed. No Epic cloud service is used.
"""
from pathlib import Path

UE_ROOT = Path('/mnt/nvme02/local/ue')
UE_EDITOR_CMD = UE_ROOT / 'Engine/Binaries/Linux/UnrealEditor-Cmd'
MH_PLUGIN = UE_ROOT / 'Engine/Plugins/MetaHuman/MetaHumanCharacter'
PROJECT = Path('/mnt/nvme01/work/vhuman-mh/MHGen/MHGen.uproject')
DATASET_ROOT = Path('/mnt/nvme01/data/vhuman-mh')        # nvme02 was at 30 GB free when this was created
MODELS_ROOT = Path('/mnt/nvme02/models/vhuman-mh')
UE_SCRIPTS = Path(__file__).resolve().parent / 'ue_scripts'
