"""Eye/skin contact bands preserve distant detail, other surfaces and alpha."""
import unittest
import numpy as np
from .reconstruction.eye_contact import (_coverage_skin_contact_band, _thin_aperture_band,
                                         contact_band, eye_upper_weight, skin_contact_band,
                                         soften_contact)


class EyeContactTests(unittest.TestCase):
    def test_skin_coverage_band_avoids_binary_edge_jump(self):
        y,x=np.mgrid[:48,:48];depth=np.ones(x.shape,np.float32);frames=[]
        for shift in np.linspace(0,1,101):
            eye=np.clip(x+.5-(23.5+shift),0,1).astype(np.float32)
            frames.append(_coverage_skin_contact_band(eye,1-eye,depth,1000,.0012))
        self.assertLess(float(np.abs(np.diff(np.stack(frames),axis=0)).max()),.02)

    def test_skin_coverage_band_guards_specks_depth_and_large_width(self):
        eye=np.zeros((32,32),np.float32);eye[16,16]=.5;depth=np.ones_like(eye)
        self.assertFalse(_coverage_skin_contact_band(eye,1-eye,depth,1000,.0012).any())
        eye[16,16]=1;depth[16,16]=0;eye[16,17]=.25
        self.assertFalse(_coverage_skin_contact_band(eye,1-eye,depth,1000,.0012).any())
        depth[16,16]=.1
        self.assertFalse(_coverage_skin_contact_band(eye,1-eye,depth,1000,.0012).any())
        eye[:]=0;eye[8:24,16:24]=1;skin=1-eye
        depth[:]=1
        depth[:,:16]=1.1
        self.assertFalse(_coverage_skin_contact_band(eye,skin,depth,1000,.0012)[:,:16].any())
        depth[:]=1
        np.testing.assert_array_equal(_coverage_skin_contact_band(eye,skin,depth,1000,.032),
                                      skin_contact_band(eye,skin,depth,1000,.032))
        np.testing.assert_array_equal(_coverage_skin_contact_band(eye,skin,depth,1000,.0012),
                                      skin_contact_band(eye,skin,depth,1000,.0012))
        large_eye=np.zeros((512,512),np.float32);large_eye[:,:256]=1
        large_depth=np.ones_like(large_eye);large_skin=1-large_eye
        np.testing.assert_array_equal(_coverage_skin_contact_band(large_eye,large_skin,large_depth,1000,.031),
                                      skin_contact_band(large_eye,large_skin,large_depth,1000,.031))

    def test_skin_coverage_fades_at_depth_cutoff(self):
        eye=np.zeros((32,32),np.float32);eye[8:24,16:24]=1;skin=1-eye;frames=[]
        for gap in np.linspace(.007,.011,101):
            depth=np.ones_like(eye);depth[:,:16]+=gap
            frames.append(_coverage_skin_contact_band(eye,skin,depth,1000,.0012))
        self.assertLess(float(np.abs(np.diff(np.stack(frames),axis=0)).max()),.02)

    def test_distant_fractional_speck_does_not_change_gather_budget(self):
        eye=np.zeros((512,512),np.float32);eye[100:110,100:120]=1;eye[100:110,99]=.45
        skin=np.ones_like(eye);depth=np.ones_like(eye)
        baseline=_coverage_skin_contact_band(eye,skin,depth,1000,.031)
        eye[400,400]=.25
        np.testing.assert_array_equal(_coverage_skin_contact_band(eye,skin,depth,1000,.031),baseline)

    def test_unrelated_foreground_depth_does_not_expand_gather(self):
        eye=np.zeros((512,512),np.float32);eye[100:300,100:300]=1;eye[100:300,99]=.45
        skin=np.ones_like(eye);depth=np.ones_like(eye)
        baseline=_coverage_skin_contact_band(eye,skin,depth,1000,.0012)
        depth[450,450]=.01
        np.testing.assert_array_equal(_coverage_skin_contact_band(eye,skin,depth,1000,.0012),baseline)

    def test_skin_coverage_bounds_confidence_work(self):
        from unittest.mock import patch
        eye=np.ones((2048,2048),np.float32);depth=np.ones_like(eye);skin=np.ones_like(eye)
        with patch('server.vhuman.reconstruction.eye_contact._eye_depth_coverage_confidence',
                   side_effect=AssertionError('unbounded confidence calculation')):
            band=_coverage_skin_contact_band(eye,skin,depth,1000,.0012)
        self.assertFalse(band.any())

    def test_faint_background_does_not_force_work_fallback(self):
        eye=np.full((2048,2048),.001,np.float32);eye[1000:1020,1000:1020]=1
        eye[1000:1020,999]=.45;skin=1-eye;depth=np.ones_like(eye)
        improved=_coverage_skin_contact_band(eye,skin,depth,1000,.0012)
        legacy=skin_contact_band(eye,skin,depth,1000,.0012)
        self.assertGreater(float(np.max(np.abs(improved-legacy))),.001)

    def test_confidence_prefilter_keeps_near_threshold_sources(self):
        from unittest.mock import patch
        from .reconstruction.eye_contact import _eye_depth_coverage_confidence
        rng=np.random.default_rng(42);eye=np.zeros((64,64),np.float32)
        eye[12:52,12:52]=rng.uniform(.14,.15,(40,40));skin=1-eye;depth=np.ones_like(eye)
        pruned=_coverage_skin_contact_band(eye,skin,depth,1000,.0012)
        def unpruned(e,d,eligible=None):
            return _eye_depth_coverage_confidence(e,d)
        with patch('server.vhuman.reconstruction.eye_contact._eye_depth_coverage_confidence',side_effect=unpruned):
            full=_coverage_skin_contact_band(eye,skin,depth,1000,.0012)
        np.testing.assert_array_equal(pruned,full)

    def test_thin_aperture_fallback_is_continuous_and_optional(self):
        depth=np.ones((64,128),np.float32);weights=[]
        for coverage in np.linspace(.45,.55,101):
            eye=np.zeros_like(depth);eye[30,40:80]=coverage
            band=contact_band(eye,depth,1000,.001)
            weights.append(float(_thin_aperture_band(band,eye,depth,1000,.001)[30,55]))
        self.assertLess(max(np.abs(np.diff(weights))),.002)
        eye=np.zeros_like(depth);eye[30,40:80]=.4
        image=np.ones((*depth.shape,4),np.float32);image[...,3]=.7;image[eye>0,:3]=0
        base=soften_contact(image,eye,depth,1000,width_m=.001,sigma_m=.001,tint=(1,1,1))
        enabled=soften_contact(image,eye,depth,1000,width_m=.001,sigma_m=.001,tint=(1,1,1),thin_aperture=True)
        np.testing.assert_array_equal(base,image)
        self.assertGreater(float(enabled[30,55,0]),0)
        np.testing.assert_array_equal(enabled[eye==0],image[eye==0])
        np.testing.assert_array_equal(enabled[...,3],image[...,3])
        combined=soften_contact(image,eye,depth,1000,width_m=.002,sigma_m=.001,tint=(1,1,1),
                                skin_mask=np.ones_like(eye),skin_width_m=.002,skin_strength=1,
                                thin_aperture=True,skin_coverage=True)
        self.assertGreater(float(combined[30,55,0]),0)
        self.assertLess(float(combined[29,55,0]),1)
        np.testing.assert_array_equal(combined[...,3],image[...,3])

    def test_thin_aperture_rejects_specks_small_width_and_depth_gaps(self):
        depth=np.ones((64,128),np.float32);eye=np.zeros_like(depth);eye[30,40:45]=.4
        base=contact_band(eye,depth,1000,.001)
        np.testing.assert_array_equal(_thin_aperture_band(base,eye,depth,1000,.001),base)
        eye[30,40:80]=.4
        tiny=contact_band(eye,depth,1000,.00001)
        np.testing.assert_array_equal(_thin_aperture_band(tiny,eye,depth,1000,.00001),tiny)
        depth[30,60:80]=1.1
        base=contact_band(eye,depth,1000,.001)
        np.testing.assert_array_equal(_thin_aperture_band(base,eye,depth,1000,.001),base)

    def test_eye_axis_field_handles_separate_and_degenerate_apertures(self):
        y,x=np.mgrid[:80,:140];eye=((((x-38)/25)**2+((y-40)/10)**2<1)|
                                    (((x-102)/25)**2+((y-40)/10)**2<1)).astype(np.float32)
        depth=np.full(eye.shape,.5,np.float32);field=eye_upper_weight(eye,eye>0,depth)
        for centre in (38,102):
            self.assertGreater(float(field[33,centre]),.9)
            self.assertLess(float(field[47,centre]),.1)
        round_eye=(((x-70)**2+(y-40)**2)<100).astype(np.float32)
        neutral=eye_upper_weight(round_eye,round_eye>0,depth)
        np.testing.assert_allclose(neutral[round_eye>0],.5,atol=1e-6)
        vertical=((((x-70)/8)**2+((y-40)/25)**2)<1).astype(np.float32)
        neutral=eye_upper_weight(vertical,vertical>0,depth)
        np.testing.assert_allclose(neutral[vertical>0],.5,atol=1e-6)
        small=np.zeros_like(eye);small[10:12,10:12]=1
        np.testing.assert_allclose(eye_upper_weight(small,small>0,depth)[small>0],.5,atol=1e-6)
        nearly_closed=np.zeros_like(eye);nearly_closed[39:41,40:100]=1
        np.testing.assert_allclose(eye_upper_weight(nearly_closed,nearly_closed>0,depth)[nearly_closed>0],.5,atol=1e-6)
        noisy=np.zeros((200,200),np.float32);noisy[::2,::2]=1
        neutral=eye_upper_weight(noisy,noisy>0,np.full(noisy.shape,.5,np.float32))
        np.testing.assert_allclose(neutral[noisy>0],.5,atol=1e-6)

    def test_split_strength_is_optional_and_preserves_alpha(self):
        y,x=np.mgrid[:48,:64];eye=(((x-32)/20)**2+((y-24)/9)**2<1).astype(np.float32)
        image=np.ones((48,64,4),np.float32);image[eye>0,:3]=0;image[...,3]=.7
        depth=np.full(eye.shape,.5,np.float32);args=dict(width_m=.003,sigma_m=.001,strength=.5,tint=(1,1,1))
        base=soften_contact(image,eye,depth,1000,**args)
        equal=soften_contact(image,eye,depth,1000,upper_strength=.5,lower_strength=.5,**args)
        np.testing.assert_array_equal(base,equal)
        split=soften_contact(image,eye,depth,1000,upper_strength=.8,lower_strength=.2,**args)
        np.testing.assert_array_equal(split[eye==0],image[eye==0])
        np.testing.assert_array_equal(split[...,3],image[...,3])
        self.assertFalse(np.array_equal(split,base))
        nonzero=soften_contact(image,eye,depth,1000,width_m=.003,sigma_m=.001,strength=0,
                               tint=(1,1,1),upper_strength=.8,lower_strength=.2)
        self.assertFalse(np.array_equal(nonzero,image))
        with self.assertRaises(ValueError):soften_contact(image,eye,depth,1000,upper_strength=.8,**args)
        with self.assertRaises(ValueError):soften_contact(image,eye,depth,1000,upper_strength=.8,lower_strength=2,**args)

    def test_eye_axis_field_tracks_moderate_image_rotation(self):
        import cv2
        n=128;y,x=np.mgrid[:n,:n]
        eye=(((x-64)/40)**2+((y-64)/16)**2<1).astype(np.float32)
        depth=np.full(eye.shape,.5,np.float32);base=eye_upper_weight(eye,eye>0,depth)
        matrix=cv2.getRotationMatrix2D((64,64),30,1)
        rotated=cv2.warpAffine(eye,matrix,(n,n),flags=cv2.INTER_NEAREST)
        field=eye_upper_weight(rotated,rotated>0,depth)
        expected=cv2.warpAffine(base,matrix,(n,n),flags=cv2.INTER_LINEAR)
        interior=(rotated>0)&(cv2.warpAffine(eye,matrix,(n,n),flags=cv2.INTER_LINEAR)>.99)
        self.assertLess(float(np.mean(np.abs(field[interior]-expected[interior]))),.015)

    def test_eye_axis_field_is_stable_under_subpixel_translation(self):
        import cv2
        n=128;y,x=np.mgrid[:n,:n]
        eye=(((x-64)/40)**2+((y-64)/16)**2<1).astype(np.float32)
        depth=np.full(eye.shape,.5,np.float32);base=eye_upper_weight(eye,eye>0,depth)
        matrix=np.float32([[1,0,.5],[0,1,0]])
        moved=cv2.warpAffine(eye,matrix,(n,n),flags=cv2.INTER_LINEAR)
        field=eye_upper_weight(moved,moved>.5,depth)
        expected=cv2.warpAffine(base,matrix,(n,n),flags=cv2.INTER_LINEAR)
        interior=(moved>.99)&(cv2.warpAffine(eye,matrix,(n,n),flags=cv2.INTER_LINEAR)>.99)
        self.assertLess(float(np.mean(np.abs(field[interior]-expected[interior]))),.01)

    def test_nearly_closed_eye_fades_split_without_binary_jump(self):
        n=96;ss=8;yy,xx=np.mgrid[:n*ss,:n*ss];xx=(xx+.5)/ss;yy=(yy+.5)/ss
        depth=np.full((n,n),.5,np.float32);spreads=[]
        for radius in (1.6,1.5,1.4):
            high=((xx-48)/25)**2+((yy-48)/radius)**2<1
            eye=high.reshape(n,ss,n,ss).mean((1,3)).astype(np.float32)
            active=contact_band(eye,depth,1000,.0018)>0
            field=eye_upper_weight(eye,active,depth)[active]
            spreads.append(float(field.max()-field.min()))
        self.assertLess(max(np.abs(np.diff(spreads))),.06)

    def test_fractional_coverage_fades_contact_at_the_eye_edge(self):
        eye=np.zeros((16,24),np.float32);eye[4:12,8:16]=1
        eye[7,7]=.25;eye[8,7]=.5;eye[0,0]=.25
        depth=np.full(eye.shape,.5,np.float32)
        band=contact_band(eye,depth,1000,.002)
        self.assertAlmostEqual(float(band[7,7]),.25,places=6)
        self.assertAlmostEqual(float(band[8,7]),.5,places=6)
        self.assertEqual(float(band[0,0]),0.)
        image=np.ones((16,24,4),np.float32);image[...,3]=.7;image[eye>0,:3]=0
        out=soften_contact(image,eye,depth,1000,width_m=.002,sigma_m=.0005,strength=1,tint=(1,1,1))
        self.assertGreater(float(out[7,7,0]),0.)
        np.testing.assert_array_equal(out[eye==0],image[eye==0])
        np.testing.assert_array_equal(out[...,3],image[...,3])
        tiny=soften_contact(image,eye,depth,1000,width_m=1e-7,sigma_m=.0005,strength=1,tint=(1,1,1))
        self.assertTrue(np.isfinite(tiny).all())
        self.assertGreater(float(tiny[7,7,0]),0.)

    def test_subpixel_motion_does_not_jump_at_half_coverage(self):
        depth=np.full((32,32),.5,np.float32);x=np.arange(32,dtype=np.float32)
        weights=[]
        for offset in np.linspace(0,1,101):
            coverage=np.clip(x+.5-(15.5+offset),0,1)
            mask=np.broadcast_to(coverage,depth.shape)
            weights.append(float(contact_band(mask,depth,1000,.0012)[16,16]))
        self.assertLess(max(np.abs(np.diff(weights))),.011)

    def test_scale_mask_remains_based_on_original_opaque_band(self):
        from scipy.ndimage import distance_transform_edt
        eye=np.zeros((24,32),np.float32);eye[4:20,8:24]=1
        eye[4:20,8]=.7;eye[4:20,7]=.3
        depth=np.broadcast_to(np.linspace(.35,.75,32,dtype=np.float32),eye.shape)
        band,scale_mask=contact_band(eye,depth,1000,.0002,_return_scale_mask=True)
        valid=(eye>.5)&(depth>0);dist=np.maximum(distance_transform_edt(valid)-.5,0)
        t=np.clip(dist/(.0002*1000/depth),0,1)
        expected=(eye*valid*(1-t*t*(3-2*t))).astype(np.float32)>0
        np.testing.assert_array_equal(scale_mask,expected)
        self.assertTrue((band[eye==0]==0).all())

    def test_subpixel_fix_keeps_nonfractional_pixels_and_alpha(self):
        from scipy.ndimage import distance_transform_edt
        from unittest.mock import patch
        rng=np.random.default_rng(9);image=rng.random((24,32,4),dtype=np.float32)
        eye=np.zeros((24,32),np.float32);eye[4:20,8:24]=1
        eye[4:20,8]=.7;eye[4:20,7]=.3
        depth=np.broadcast_to(np.linspace(.35,.75,32,dtype=np.float32),eye.shape)
        def old_band(mask,z,focal,width,*,_return_scale_mask=False):
            valid=(mask>.5)&(z>0);dist=np.maximum(distance_transform_edt(valid)-.5,0)
            t=np.clip(dist/(width*focal/np.maximum(z,1e-6)),0,1)
            band=(mask*valid*(1-t*t*(3-2*t))).astype(np.float32)
            return (band,band>0) if _return_scale_mask else band
        args=dict(width_m=.0002,sigma_m=.0005,strength=.8,tint=(1,1,1))
        new=soften_contact(image,eye,depth,1000,**args)
        with patch('server.vhuman.reconstruction.eye_contact.contact_band',side_effect=old_band):
            old=soften_contact(image,eye,depth,1000,**args)
        np.testing.assert_array_equal(new[(eye==0)|(eye==1)],old[(eye==0)|(eye==1)])
        np.testing.assert_array_equal(new[...,3],old[...,3])

    def test_contact_only_and_constant_light(self):
        image = np.ones((32, 48, 4), np.float32)
        image[..., 3] = .7
        mask = np.zeros((32, 48), np.float32); mask[4:28, 8:40] = 1
        depth = np.ones(mask.shape, np.float32) * .5
        out = soften_contact(image, mask, depth, 1000, tint=(1,1,1))
        np.testing.assert_array_equal(out, image)
        band = contact_band(mask, depth, 1000, .001)
        self.assertEqual(band[16,24], 0)
        self.assertGreater(band[4,16], 0)
        self.assertTrue((band[mask == 0] == 0).all())

    def test_blends_skin_into_eye_margin_without_touching_centre(self):
        image = np.ones((32, 48, 4), np.float32); image[:,16:32,:3] = 0
        mask = np.zeros((32,48), np.float32); mask[:,16:32] = 1
        depth = np.ones(mask.shape, np.float32) * .5
        out = soften_contact(image, mask, depth, 1000, width_m=.002, sigma_m=.0005, strength=1, tint=(1,1,1))
        self.assertGreater(out[16,16,0], 0)
        np.testing.assert_array_equal(out[:,24,:3], image[:,24,:3])
        np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
        np.testing.assert_array_equal(out[...,3], image[...,3])
        np.testing.assert_array_equal(soften_contact(image, mask, depth, 1000, strength=0), image)

    def test_staged_geometry_projection_and_skin_exclusion(self):
        from .reconstruction.eye_contact import projected_eye_mask
        x=np.linspace(-.1,.1,9)
        lower=np.stack((x,x*0-.05,x*0),-1)
        upper=np.stack((x,x*0+.05,x*0),-1)
        data={'left_tearline_positions':np.repeat(lower[:,None],8,axis=1).reshape(-1,3),
              'left_tearline_upper_positions':np.repeat(upper[:,None],8,axis=1).reshape(-1,3)}
        camera=dict(origin=[0,0,1],rotation=np.eye(3),focal_px=100,size=[64,64])
        mask=projected_eye_mask(data,camera,(64,64))
        self.assertEqual(mask[32,32],1)
        self.assertEqual(mask[10,10],0)
        protected=projected_eye_mask(data,camera,(64,64),skin_mask=np.ones((64,64)))
        self.assertFalse(protected.any())

    def test_skin_side_softens_contact_and_preserves_other_pixels(self):
        image=np.full((32,48,4),.2,np.float32)
        image[:,16:32,:3]=1
        image[...,3]=np.linspace(0,1,48)[None]
        eye=np.zeros((32,48),np.float32);eye[:,16:32]=1
        skin=1-eye;depth=np.full(eye.shape,.5)
        out=soften_contact(image,eye,depth,1000,strength=0,tint=(1,1,1),
                           sigma_m=.0005,skin_mask=skin,skin_width_m=.002,skin_strength=1)
        self.assertGreater(out[16,15,0],image[16,15,0])
        np.testing.assert_array_equal(out[:,8],image[:,8])
        np.testing.assert_array_equal(out[eye==1],image[eye==1])
        np.testing.assert_array_equal(out[...,3],image[...,3])

    def test_skin_contact_rejects_foreground_surface_and_empty_eye(self):
        eye=np.zeros((24,40),np.float32);eye[:,16:28]=1
        skin=1-eye;depth=np.full(eye.shape,.5)
        depth[:,15]=.47  # An unrelated foreground surface beside the eye.
        band=skin_contact_band(eye,skin,depth,1000,.002)
        self.assertEqual(band[12,15],0)
        self.assertGreater(band[12,14],0)
        self.assertFalse(band[eye==1].any())
        self.assertFalse(skin_contact_band(eye*0,skin,depth,1000,.002).any())

    def test_skin_contact_requires_valid_coverage_and_strength(self):
        image=np.ones((8,8,4),np.float32);eye=np.ones((8,8),np.float32)
        depth=np.full(eye.shape,.5)
        with self.assertRaises(ValueError):
            soften_contact(image,eye,depth,1000,skin_strength=1)
        with self.assertRaises(ValueError):
            soften_contact(image,eye,depth,1000,skin_mask=eye,skin_strength=1.1)
        with self.assertRaises(ValueError):
            skin_contact_band(eye,eye[:-1],depth,1000,.0012)

    def test_independent_skin_sigma_keeps_eye_side_unchanged(self):
        image=np.full((32,48,4),.2,np.float32);image[:,16:32,:3]=1
        eye=np.zeros((32,48),np.float32);eye[:,16:32]=1
        skin=1-eye;depth=np.full(eye.shape,.5)
        common=dict(strength=.5,tint=(1,1,1),sigma_m=.0005,skin_mask=skin,
                    skin_width_m=.002,skin_strength=1.)
        base=soften_contact(image,eye,depth,1000,**common)
        split=soften_contact(image,eye,depth,1000,skin_sigma_m=.0002,**common)
        np.testing.assert_array_equal(base[eye==1],split[eye==1])
        np.testing.assert_array_equal(base[...,3],split[...,3])
        self.assertNotEqual(base[16,15,0],split[16,15,0])
        with self.assertRaises(ValueError):
            soften_contact(image,eye,depth,1000,skin_sigma_m=0,**common)

    def test_independent_skin_tint_preserves_skin_side(self):
        image=np.full((32,48,4),.2,np.float32);image[:,16:32,:3]=1
        eye=np.zeros((32,48),np.float32);eye[:,16:32]=1
        skin=1-eye;depth=np.full(eye.shape,.5)
        common=dict(strength=.5,sigma_m=.0005,skin_mask=skin,skin_width_m=.002,skin_strength=1.)
        base=soften_contact(image,eye,depth,1000,**common)
        split=soften_contact(image,eye,depth,1000,tint=(1,1,1),skin_tint=(1,.904661,.896269),**common)
        np.testing.assert_array_equal(base[eye==0],split[eye==0])
        np.testing.assert_array_equal(base[...,3],split[...,3])
        self.assertNotEqual(base[16,16,1],split[16,16,1])
        with self.assertRaises(ValueError):
            soften_contact(image,eye,depth,1000,skin_tint=(1,-1,1),**common)
