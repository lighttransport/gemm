"""Framework-free corrective assets; training machinery is loaded on demand."""
from .mlruntime import MLDeformer
from .contact_setup import (EYE_MARGIN_MM, TOOTH_MARGIN_MM, sample_controls,
                            tooth_spheres, tongue_spheres, region_mask, export_contacts)


def __getattr__(name):
    if name in ('Contacts', 'Solver', 'MLP2', 'train'):
        from . import mldeformer_training
        return getattr(mldeformer_training, name)
    raise AttributeError(name)
