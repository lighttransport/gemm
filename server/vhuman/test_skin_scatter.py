"""Linear-light scattering and recombination invariants."""
import unittest
import numpy as np
from .reconstruction.skin_scatter import (scatter_lighting, recombine_skin,
                                           suppress_indirect_spikes, smooth_scatter_correction)


class SkinScatterTests(unittest.TestCase):
    def test_symmetric_sampling_rotates_point_light_and_preserves_constant(self):
        n=65;mask=np.ones((n,n),np.float32);depth=np.full((n,n),.1,np.float32)
        normal=np.zeros((n,n,3),np.float32);normal[...,2]=1
        light=np.zeros_like(normal);light[n//2,n//2]=1
        filtered=scatter_lighting(light,mask,depth,normal,512.,samples=48,symmetric=True)
        np.testing.assert_allclose(filtered,np.rot90(filtered),atol=1e-7)
        constant=np.ones_like(light)
        np.testing.assert_allclose(scatter_lighting(constant,mask,depth,normal,512.,samples=48,symmetric=True),constant,atol=1e-6)
        with self.assertRaises(ValueError):
            scatter_lighting(light,mask,depth,normal,512.,samples=12,symmetric=True)

    def test_correction_filter_respects_surface_edges_and_constant_light(self):
        shape=(16,32);mask=np.ones(shape,np.float32);depth=np.full(shape,.5,np.float32)
        normal=np.zeros((*shape,3),np.float32);normal[...,2]=1
        correction=np.ones((*shape,3),np.float32);correction[:,16:]=10
        depth[:,16:]=.52
        filtered=smooth_scatter_correction(correction,mask,depth,normal,1.)
        np.testing.assert_allclose(filtered[:,15],1,atol=2e-6)
        np.testing.assert_allclose(filtered[:,16],10,atol=2e-6)
        depth[:]=.5;normal[:,16:,2]=-1
        filtered=smooth_scatter_correction(correction,mask,depth,normal,1.)
        np.testing.assert_allclose(filtered[:,15],1,atol=2e-6)
        np.testing.assert_allclose(filtered[:,16],10,atol=2e-6)
        normal[...,2]=1;constant=np.full_like(correction,3.)
        np.testing.assert_allclose(smooth_scatter_correction(constant,mask,depth,normal,1.),constant,atol=2e-6)
        np.testing.assert_array_equal(smooth_scatter_correction(correction,mask,depth,normal,0),correction)
        np.testing.assert_array_equal(smooth_scatter_correction(correction,mask,depth,normal,1e-320),correction)
        with self.assertRaises(ValueError):smooth_scatter_correction(correction,mask,depth,normal,9.)

    def test_correction_filter_follows_integer_image_translation(self):
        rng=np.random.default_rng(12);shape=(24,32)
        correction=rng.random((*shape,3),dtype=np.float32)
        mask=np.ones(shape,np.float32);mask[8:16,14:18]=0
        depth=np.full(shape,.5,np.float32);depth[:,20:]+=.01
        normal=np.zeros((*shape,3),np.float32);normal[...,2]=1
        expected=smooth_scatter_correction(correction,mask,depth,normal,1.)
        shift=lambda x:np.roll(x,1,axis=1)
        translated=smooth_scatter_correction(shift(correction),shift(mask),shift(depth),shift(normal),1.)
        np.testing.assert_allclose(translated[:,4:-4],shift(expected)[:,4:-4],atol=1e-6)

    def test_smoothed_correction_preserves_alpha_and_non_skin(self):
        shape=(16,24);mask=np.ones(shape,np.float32);mask[:,:5]=0
        depth=np.full(shape,.5,np.float32);normal=np.zeros((*shape,3),np.float32);normal[...,2]=1
        rng=np.random.default_rng(3);light=rng.random((*shape,3),dtype=np.float32)
        diff=rng.random((*shape,3),dtype=np.float32)*.1-.05
        colour=rng.random((*shape,3),dtype=np.float32);beauty=rng.random((*shape,4),dtype=np.float32)
        correction=smooth_scatter_correction(diff,mask,depth,normal,1.)
        result=recombine_skin(beauty,light,light+correction,colour,mask)
        np.testing.assert_array_equal(result[...,3],beauty[...,3])
        np.testing.assert_array_equal(result[mask==0],beauty[mask==0])

    def test_indirect_spike_guard_replaces_only_extreme_rgb_pixels(self):
        raw=np.array([[[.1,.1,.1],[2.,1.5,1.]],[[.05,.04,.03],[.2,.1,.05]]],np.float32)
        guide=np.array([[[.09,.09,.09],[.2,.1,.05]],[[.05,.04,.03],[.2,.1,.05]]],np.float32)
        filtered,selected=suppress_indirect_spikes(raw,guide)
        np.testing.assert_array_equal(selected,[[False,True],[False,False]])
        np.testing.assert_array_equal(filtered[0,1],guide[0,1])
        np.testing.assert_array_equal(filtered[~selected],raw[~selected])
        with self.assertRaises(ValueError):suppress_indirect_spikes(raw,guide[:1])
        with self.assertRaises(ValueError):suppress_indirect_spikes(raw,guide,ratio=float('nan'))
        floor=np.full(raw.shape[:2],.05,np.float32)
        filtered_scaled,selected_scaled=suppress_indirect_spikes(raw*.1,guide*.1,floor=floor*.1)
        np.testing.assert_array_equal(selected_scaled,selected)
        np.testing.assert_allclose(filtered_scaled,filtered*.1,rtol=1e-6)
        with self.assertRaises(ValueError):suppress_indirect_spikes(raw,guide,floor=floor[:1])

    def test_constant_light_and_empty_mask(self):
        light = np.full((16, 20, 3), [.8, .5, .2], np.float32)
        mask = np.ones((16, 20), np.float32); mask[:, :4] = 0
        z = np.full(mask.shape, .25, np.float32)
        normal = np.broadcast_to([0., 0., 1.], light.shape).copy()
        filtered = scatter_lighting(light, mask, z, normal, 500., samples=16)
        np.testing.assert_allclose(filtered, light, atol=2e-7)
        np.testing.assert_array_equal(scatter_lighting(light, mask * 0, z, normal, 500., samples=16), light)

    def test_eye_and_alpha_are_preserved(self):
        beauty = np.ones((8, 8, 4), np.float32)
        beauty[..., 3] = .75
        mask = np.ones((8, 8)); mask[:, 4:] = 0
        light = np.ones((8, 8, 3), np.float32)
        out = recombine_skin(beauty, light, light * 2, light * .25, mask, .5)
        np.testing.assert_allclose(out[:, :4, :3], 1.125)
        np.testing.assert_array_equal(out[:, 4:], beauty[:, 4:])
        np.testing.assert_array_equal(out[..., 3], beauty[..., 3])
        np.testing.assert_array_equal(recombine_skin(beauty, light, light * 2, light, mask, 0), beauty)

    def test_no_crossing_skin_and_eye_boundary(self):
        light = np.ones((24, 24, 3), np.float32)
        mask = np.zeros((24, 24), np.float32); mask[:, :12] = 1
        light[:, 12:] = 1000
        normal = np.broadcast_to([0., 0., 1.], light.shape).copy()
        out = scatter_lighting(light, mask, np.ones(mask.shape), normal, 500., samples=16)
        np.testing.assert_allclose(out[:, :12], 1., atol=1e-6)
        np.testing.assert_array_equal(out[:, 12:], light[:, 12:])

    def test_albedo_and_specular_are_not_filtered(self):
        # Constant irradiance must leave both pigment detail and a sharp highlight intact.
        rng = np.random.default_rng(7)
        albedo = rng.uniform(.05, .8, (12, 12, 3)).astype(np.float32)
        lighting = np.ones_like(albedo)
        specular = np.zeros_like(albedo); specular[6, 6] = [4, 3, 2]
        beauty = lighting * albedo + specular
        mask = np.ones((12, 12), np.float32)
        normal = np.broadcast_to([0., 0., 1.], albedo.shape).copy()
        scattered = scatter_lighting(lighting, mask, mask * .2, normal, 500, samples=16)
        np.testing.assert_allclose(recombine_skin(beauty, lighting, scattered, albedo, mask), beauty, atol=1e-7)

    def test_diffusion_conserves_energy_and_spreads_red_further(self):
        light = np.zeros((96, 96, 3), np.float32); light[48, 48] = 1
        mask = np.ones((96, 96), np.float32)
        normal = np.broadcast_to([0., 0., 1.], light.shape).copy()
        out = scatter_lighting(light, mask, mask * .2, normal, 100., (2., 1., .5), samples=48)
        np.testing.assert_allclose(out.sum((0, 1)), [1, 1, 1], atol=.002)
        yy, xx = np.mgrid[:96, :96]; r2 = (yy - 48) ** 2 + (xx - 48) ** 2
        moment = (out * r2[..., None]).sum((0, 1))
        self.assertGreater(moment[0], moment[1]); self.assertGreater(moment[1], moment[2])

    def test_native_thin_parts_and_alpha_are_preserved(self):
        from .reconstruction.skin_scatter import preserve_native_regions
        native = np.full((4, 6, 4), [.8, .2, .1, .7], np.float32)
        treated = np.full_like(native, [.2, .4, .6, .1])
        mask = np.ones((4, 6), np.float32); mask[:, :2] = 0; mask[:, 2] = .5
        out = preserve_native_regions(native, treated, mask)
        np.testing.assert_array_equal(out[:, :2], native[:, :2])
        np.testing.assert_array_equal(out[..., 3], native[..., 3])
        np.testing.assert_array_equal(out[:, 3:, :3], treated[:, 3:, :3])
        np.testing.assert_allclose(out[:, 2, :3], np.tile([.5, .3, .35], (4, 1)))
        with self.assertRaises(ValueError):
            preserve_native_regions(native, treated, mask + 1.)
