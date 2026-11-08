"""Regression checks for the optional production skin profile."""
import unittest
from unittest.mock import patch
import numpy as np
from .reconstruction.profile_assets import skin_profile_weights
from .reconstruction.skin_profile import enabled, preflight, default_overrides, compose


class SkinProfileTests(unittest.TestCase):
    def test_full_default_and_explicit_opt_out(self):
        defaults=default_overrides('full', ())
        self.assertEqual(defaults['skin_lighting_profile'],1)
        self.assertEqual(defaults['skin_normal_weighted_blend'],1)
        self.assertEqual(defaults['skin_micro_isotropic'],1)
        self.assertEqual(defaults['skin_normal_preserve_lids'],1)
        self.assertEqual(defaults['skin_pore_scale'],.5)
        self.assertEqual(defaults['eye_contact_skin_strength'],1)
        self.assertEqual(defaults['eye_contact_skin_width_m'],.0012)
        self.assertEqual(defaults['eye_contact_strength'],.8472126447375768)
        self.assertNotIn('skin_indirect_firefly_guard',defaults)
        self.assertNotIn('skin_scatter_correction_sigma_m',defaults)
        self.assertNotIn('skin_scatter_denoise',defaults)
        self.assertNotIn('skin_scatter_symmetric',defaults)
        disabled=default_overrides('full', (),{'skin_lighting_profile':0})
        self.assertEqual(disabled['skin_lighting_profile'],0)
        self.assertNotIn('skin_micro_isotropic',disabled)
        self.assertNotIn('skin_lighting_profile',default_overrides('baseline', ()))
        self.assertNotIn('skin_lighting_profile',default_overrides('full', ('2c',)))

    def test_previous_profile_flags_remain_explicit(self):
        old=dict(skin_lighting_profile=1,skin_normal_weighted_blend=0,skin_micro_isotropic=0,
                 skin_normal_preserve_lids=0,skin_pore_scale=1,eye_contact_skin_strength=0,
                 eye_contact_strength=.35739718109334984)
        result=default_overrides('full', (),old)
        for key,value in old.items():self.assertEqual(result[key],value)

    def test_native_and_refined_anatomical_weights(self):
        data = dict(skin_native_ids=np.array([0, 1]), skin_positions=np.zeros((3, 3)),
                    skin_bound_ids=np.array([[0, 1]]), skin_bound_weights=np.array([[.25, .75]]))
        np.testing.assert_array_equal(skin_profile_weights(data, [0, 1]), [1, 0, .25])

    def test_missing_refined_binding_rejected(self):
        with self.assertRaises(ValueError):
            skin_profile_weights(dict(skin_native_ids=np.array([0]), skin_positions=np.zeros((2, 3))), [0])

    def test_disabled_profile_needs_no_optional_dependency(self):
        preflight(dict(quality='baseline'))
        self.assertFalse(enabled({}))

    def test_nonfinite_enable_rejected(self):
        with self.assertRaises(ValueError):
            enabled(dict(material_overrides=dict(skin_lighting_profile=float('nan'))))

    def test_incompatible_material_rejected(self):
        with self.assertRaises(ValueError):
            preflight(dict(quality='baseline', material_overrides=dict(skin_lighting_profile=1)))

    def test_skin_contact_bounds_checked_before_render(self):
        with patch.dict('sys.modules',{'OpenEXR':object()}):
            for values in ({'eye_contact_skin_strength':-1},
                           {'eye_contact_skin_strength':1.01},
                           {'eye_contact_skin_strength':float('nan')},
                           {'eye_contact_skin_width_m':0},
                           {'eye_contact_skin_sigma_m':0},
                           {'eye_contact_upper_strength':.8},
                           {'eye_contact_upper_strength':.8,'eye_contact_lower_strength':2},
                           {'eye_contact_upper_strength':float('nan'),'eye_contact_lower_strength':.2},
                           {'eye_contact_thin_aperture':.5},
                           {'eye_contact_thin_aperture':float('nan')},
                           {'eye_contact_skin_coverage':.5},
                           {'eye_contact_skin_coverage':float('nan')},
                           {'skin_scatter_correction_sigma_m':-1},
                           {'eye_contact_tint':[1,-1,1]},
                           {'eye_contact_skin_tint':[1,1]}):
                with self.subTest(values=values),self.assertRaises(ValueError):
                    preflight(dict(quality='full',material_overrides=dict(skin_lighting_profile=1,**values)))

    def test_firefly_guard_requires_boolean_value(self):
        with patch.dict('sys.modules',{'OpenEXR':object()}):
            for value in (-1., .5, 3., float('nan')):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    preflight(dict(quality='full',material_overrides=dict(skin_lighting_profile=1,skin_indirect_firefly_guard=value)))

    def test_thin_aperture_option_composes_and_records(self):
        shape=(64,128);rgba=np.ones((*shape,4),np.float32);rgb=np.ones((*shape,3),np.float32)
        passes={name:rgb.copy() for name in ('skin_profile','skin_mask','eye_mask','DiffDir','DiffInd','DiffCol','Position','skin_geometric_normal')}
        passes['eye_mask'][:]=0;passes['eye_mask'][30,40:80,0]=.4
        rgba[30,40:80,:3]=0;passes['Combined']=rgba.copy()
        meta={'lens_mm':375.,'sensor_height_mm':24.,'camera_origin':[0,0,0],'camera_forward':[0,0,1]}
        options={'eye_contact_blur':1,'eye_contact_thin_aperture':1,'eye_contact_width_m':.001,
                 'eye_contact_sigma_m':.001,'eye_contact_strength':1,'eye_contact_tint':(1,1,1)}
        with patch('server.vhuman.reconstruction.skin_profile.scatter_lighting',side_effect=lambda light,*args,**kwargs:light):
            image,record=compose(rgba,passes,meta,options)
            disabled,disabled_record=compose(rgba,passes,meta,dict(options,eye_contact_thin_aperture=0))
        self.assertTrue(record['contact_thin_aperture'])
        self.assertFalse(disabled_record['contact_thin_aperture'])
        self.assertGreater(float(image[30,55,0]),float(disabled[30,55,0]))
        np.testing.assert_array_equal(image[...,3],rgba[...,3])
        options.update(eye_contact_thin_aperture=0,eye_contact_skin_coverage=1,
                       eye_contact_strength=0,eye_contact_skin_strength=1)
        with patch('server.vhuman.reconstruction.skin_profile.scatter_lighting',side_effect=lambda light,*args,**kwargs:light):
            image,record=compose(rgba,passes,meta,options)
            disabled,_=compose(rgba,passes,meta,dict(options,eye_contact_skin_coverage=0))
        self.assertTrue(record['contact_skin_coverage'])
        self.assertFalse(np.array_equal(image,disabled))
        np.testing.assert_array_equal(image[...,3],rgba[...,3])

    def test_firefly_guard_requires_guide_and_records_selection(self):
        shape=(4,4);rgba=np.ones((*shape,4),np.float32);rgb=np.ones((*shape,3),np.float32)
        passes={name:rgb.copy() for name in ('skin_profile','skin_mask','eye_mask','DiffDir','DiffInd','DiffCol','Position','skin_geometric_normal')}
        passes['Combined']=rgba.copy();passes['eye_mask'][:]=0
        passes['DiffInd'][:]=.1;passes['DiffInd'][1,1]=[2,2,2]
        meta={'lens_mm':50.,'sensor_height_mm':24.,'camera_origin':[0,0,0],'camera_forward':[0,0,1]}
        with self.assertRaisesRegex(ValueError,'guided indirect'):
            compose(rgba,passes,meta,{'skin_indirect_firefly_guard':1})
        passes['DiffIndDenoised']=np.full((*shape,3),.1,np.float32)
        with patch('server.vhuman.reconstruction.skin_profile.scatter_lighting',side_effect=lambda light,*args,**kwargs:light):
            image,record=compose(rgba,passes,meta,{'skin_indirect_firefly_guard':1})
        self.assertTrue(record['indirect_firefly_guard'])
        self.assertEqual(record['indirect_firefly_guard_mode'],1)
        self.assertEqual(record['guarded_opaque_skin_pixels'],1)
        np.testing.assert_array_equal(image,rgba)
        with patch('server.vhuman.reconstruction.skin_profile.scatter_lighting',side_effect=lambda light,*args,**kwargs:light):
            _,record=compose(rgba,passes,meta,{'skin_indirect_firefly_guard':2})
        self.assertEqual(record['indirect_firefly_guard_mode'],2)
        self.assertEqual(record['guarded_opaque_skin_pixels'],1)

    def test_correction_sigma_caps_projected_work(self):
        shape=(4,4);rgba=np.ones((*shape,4),np.float32);rgb=np.ones((*shape,3),np.float32)
        passes={name:rgb.copy() for name in ('skin_profile','skin_mask','eye_mask','DiffDir','DiffInd','DiffCol','Position','skin_geometric_normal')}
        passes['Combined']=rgba.copy();passes['eye_mask'][:]=0
        meta={'lens_mm':50.,'sensor_height_mm':24.,'camera_origin':[0,0,0],'camera_forward':[0,0,1]}
        with patch('server.vhuman.reconstruction.skin_profile.scatter_lighting',side_effect=lambda light,*args,**kwargs:light), \
             patch('server.vhuman.reconstruction.skin_profile.smooth_scatter_correction',side_effect=lambda delta,*args:delta) as filtering:
            image,record=compose(rgba,passes,meta,{'skin_scatter_correction_sigma_m':1.})
        self.assertEqual(record['correction_sigma_px'],8.)
        self.assertEqual(filtering.call_args.args[-1],8.)
        np.testing.assert_array_equal(image,rgba)

    def test_scatter_denoising_modes_and_validation(self):
        with patch.dict('sys.modules',{'OpenEXR':object()}):
            for value in (-1.,.5,3.,float('nan')):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    preflight(dict(quality='full',material_overrides=dict(skin_lighting_profile=1,skin_scatter_denoise=value)))
            with self.assertRaisesRegex(ValueError,'cannot be combined'):
                preflight(dict(quality='full',material_overrides=dict(skin_lighting_profile=1,skin_scatter_denoise=1,skin_indirect_firefly_guard=1)))
        shape=(4,4);rgba=np.ones((*shape,4),np.float32);rgb=np.ones((*shape,3),np.float32)
        passes={name:rgb.copy() for name in ('skin_profile','skin_mask','eye_mask','DiffDir','DiffInd','DiffCol','Position','skin_geometric_normal')}
        passes['Combined']=rgba.copy();passes['eye_mask'][:]=0
        meta={'lens_mm':50.,'sensor_height_mm':24.,'camera_origin':[0,0,0],'camera_forward':[0,0,1]}
        with self.assertRaisesRegex(ValueError,'guided diffuse'):
            compose(rgba,passes,meta,{'skin_scatter_denoise':1})
        passes['DiffIndDenoised']=rgb*.25
        with self.assertRaisesRegex(ValueError,'guided diffuse'):
            compose(rgba,passes,meta,{'skin_scatter_denoise':2})
        with self.assertRaisesRegex(ValueError,'cannot be combined'):
            compose(rgba,passes,meta,{'skin_scatter_denoise':1,'skin_indirect_firefly_guard':2})
        passes['DiffDirDenoised']=rgb*.5
        for mode,expected in ((0,2.),(1,1.25),(2,.75)):
            with self.subTest(mode=mode),patch('server.vhuman.reconstruction.skin_profile.scatter_lighting',side_effect=lambda light,*args,**kwargs:light) as scatter:
                image,record=compose(rgba,passes,meta,{'skin_scatter_denoise':mode})
                np.testing.assert_allclose(scatter.call_args.args[0],expected)
                np.testing.assert_array_equal(image,rgba)
                self.assertEqual(record['scatter_denoise_mode'],mode)

    def test_symmetric_mode_validation(self):
        with patch.dict('sys.modules',{'OpenEXR':object()}):
            for value in (-1.,.5,2.,float('nan')):
                with self.subTest(value=value),self.assertRaises(ValueError):
                    preflight(dict(quality='full',material_overrides=dict(skin_lighting_profile=1,skin_scatter_symmetric=value)))
