import numpy as np
import pytest

from .metahuman.eye_sss_recombine import recombine


def test_recombine_keeps_specular_uncolored_and_interpolates_lighting():
    diffuse = np.array([[[.4, .3, .2], [.2, .4, .6]]])
    scattered = np.array([[[.8, .5, .4], [.4, .6, .8]]])
    color = np.array([[[.25, .5, .75], [.8, .4, .2]]])
    specular = np.full((1, 2, 3), .1)
    tint = np.array([0., .5, 1.])
    expected = np.array([[[.2, .3, .4], [.26, .3, .26]]])
    np.testing.assert_allclose(recombine(diffuse, scattered, color, specular, tint), expected, atol=1e-7)


def test_recombine_rejects_mismatched_or_invalid_inputs():
    rgb = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match='matching RGB'):
        recombine(rgb, rgb[:1], rgb, rgb, .5)
    with pytest.raises(ValueError, match='between 0 and 1'):
        recombine(rgb, rgb, rgb, rgb, 1.1)
    with pytest.raises(ValueError, match='finite'):
        recombine(rgb, rgb, rgb, rgb * np.nan, .5)
