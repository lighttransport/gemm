"""Qwen-Image 2.1 as an Image-to-3D preprocessing backend.

Import with cuda/qimg21 on sys.path:

    from qimg21_i23d import ops, views, backends
    backend = backends.select_backend("auto", references=1)
    ops.preprocess_object("photo.jpg", "object.png", backend)
    ops.generate_turntable(["object.png"], "dataset/", backend, views=24)

See cuda/qimg21/IMAGE_TO_3D.md. Generated views carry *requested* camera
metadata; they are not calibrated captures.
"""
from .backends import Backend, BackendError, GenRequest, GenResult, MockBackend, select_backend
from .imageops import MaskError
from .ops import (ViewParams, complete_occlusion, edit_object, generate_image_to_3d_dataset, generate_multiview,
                  generate_turntable, generate_view, generate_views, preprocess_object, texture_preprocess)
from .views import ViewSpec, ViewSpecError

__all__ = [
    "Backend", "BackendError", "GenRequest", "GenResult", "MockBackend", "select_backend", "MaskError",
    "ViewParams", "ViewSpec", "ViewSpecError", "complete_occlusion", "edit_object",
    "generate_image_to_3d_dataset", "generate_multiview", "generate_turntable", "generate_view",
    "generate_views", "preprocess_object", "texture_preprocess",
]
