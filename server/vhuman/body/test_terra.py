"""Portable Terra tests; fixtures are numerical and require no licensed models."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from .. import gpu
from ..eye.glb import GLB, GLBBuilder
from ..service import ROOT, EyeService
from . import terra, mhr
from .motion import _export_glb
from .motion_format import document
from .musclemimic_kinematics import contact_metrics
from .terra_assets import reference
from .terra_motion import validate_source, load_reference, estimate_world_root
from .terra_terrain import Terrain, validate_spec, append_glb, Y_TO_Z


class TerraTest(unittest.TestCase):
    def test_request_defaults_and_known_terrain(self):
        terra.validate_request({"reference_motion":"walk"})
        terra.validate_request({"upload_id":"video","fixed_camera":True,"terrain":{"boxes":[]}})
        for request in ({}, {"reference_motion":"a","upload_id":"b"}, {"upload_id":"a"},
                        {"reference_motion":"a","terrain":"/etc/passwd"},
                        {"reference_motion":"a","seed":True}, {"reference_motion":"a","seed":-1},
                        {"reference_motion":"a","policy_actions":"guess"}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                terra.validate_request(request)

    def test_spec_bounds_and_seat_rejection(self):
        box = {"pos":[0,0,.1], "size":[1,1,.1], "name":"platform"}
        normalized = validate_spec({"boxes":[box]})
        self.assertEqual(normalized["boxes"][0]["name"], "terrain_box_0")
        self.assertEqual(validate_spec(normalized), normalized)
        for box in ({"pos":[0,0,0],"size":[0,1,1]}, {"pos":[0,0,np.nan],"size":[1,1,1]},
                    {"pos":[0,0,0],"size":[1,1,1],"pitch":2},
                    {"pos":[0,0,0],"size":[1,1,1],"name":"terrain_box_seat"}):
            with self.subTest(box=box), self.assertRaises(ValueError):
                validate_spec({"boxes":[box]})
        with self.assertRaises(ValueError):
            validate_spec({"boxes":[],"path":"private"})

    def test_step_height_with_uniform_world_transform(self):
        terrain = Terrain({"boxes":[{"pos":[0,0,.1],"size":[.5,.5,.1]}]},scale=2,offset=[1,.3,-2])
        source = np.array([[0,0,.2],[2,0,0]])
        world = source @ Y_TO_Z * 2 + terrain.offset
        height, normals = terrain.sample(world)
        np.testing.assert_allclose(height,[.7,.3])
        np.testing.assert_allclose(normals,[[0,1,0],[0,1,0]])
        np.testing.assert_allclose(terrain.clearance(world),0,atol=1e-12)

    def test_pitched_ramp_height_and_normals(self):
        slope = .5
        terrain = Terrain({"boxes":[{"pos":[0,0,0],"size":[1,1,.25],"pitch":-np.arctan(slope)}]})
        points = np.array([[0,1,0],[.2,1,0]])
        heights, normals = terrain.sample(points)
        np.testing.assert_allclose(heights[1]-heights[0],.2*slope,atol=1e-12)
        np.testing.assert_allclose(normals,np.tile([-slope,1,0],(2,1))/np.sqrt(1+slope*slope),atol=1e-12)
        report = contact_metrics(np.array([[[0,heights[0],0]],[[.1,heights[0]+.05,0]]]),
                                 np.ones((2,1),bool),terrain=terrain)
        self.assertGreater(report["stance_drift_p95_m"],.09)
        self.assertAlmostEqual(report["ground_penetration_max_m"],0)

    def test_skin_penetration_is_relative_to_raised_support(self):
        terrain = Terrain({"boxes":[{"pos":[0,0,.1],"size":[1,1,.1]}]})
        report = contact_metrics(np.array([[[0,.19,0]],[[0,.18,0]]]),np.ones((2,1),bool),terrain=terrain)
        self.assertAlmostEqual(report["ground_penetration_max_m"],.02)

    def test_mhr_reference_schema_and_fingerprint(self):
        source = {"model_params":np.zeros((5,204)),"identity":np.zeros(45),"times":np.arange(5)/30,
                  "root_world":np.tile([0,1,0],(5,1)),"mhr_fingerprint":np.array("a"*64)}
        validated = validate_source(source,"a"*64)
        self.assertEqual(validated["model_params"].shape,(5,204))
        for fields in ({"identity":np.zeros((5,45))},{"times":[0,0,1,2,3]},
                       {"root_world":np.full((5,3),np.nan)},{"mhr_fingerprint":np.array("b"*64)}):
            with self.subTest(fields=list(fields)), self.assertRaises(ValueError):
                validate_source(source|fields,"a"*64)
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            path = Path(folder)/"reference.npz"; np.savez(path,**source)
            loaded = load_reference(path,"a"*64)
            np.testing.assert_equal(loaded["root_world"],source["root_world"])
            source["identity"] = np.array([object()],dtype=object); np.savez(path,**source)
            with self.assertRaises(ValueError):
                load_reference(path)

    def test_world_root_does_not_flatten_successive_stances(self):
        n = 70; times = np.arange(n)/30
        elevation = np.r_[np.zeros(25),np.linspace(0,.2,10),np.full(35,.2)]
        relative = np.zeros((n,17,3)); relative[:,[11,12,15,16],1] = -1
        camera = np.zeros((n,3)); camera[:,1] = -(1+elevation)
        root, report = estimate_world_root(relative,camera,times,[0,0,0,0])
        self.assertGreater(np.mean(root[-15:,1])-np.mean(root[:15,1]),.15)
        self.assertGreater(report["stance_samples"],0)
        self.assertLess(report["camera_contact_rms_m"],.1)
        with self.assertRaises(ValueError):
            estimate_world_root(relative,camera,np.zeros(n),[0,0,0,0])

    def test_glb_embeds_static_terrain_and_preserves_root_animation(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
            avatar, motion = Path(folder)/"avatar.glb", Path(folder)/"motion.glb"
            builder = GLBBuilder(); builder.node("mhr_root",scale=[1.1]*3); builder.write(avatar)
            roots = np.array([[0,1,0],[.1,1.2,0]])
            quaternions = np.tile([0.,0,0,1],(2,1,1))
            _export_glb(avatar,motion,["mhr_root"],quaternions,np.array([0.,1.]),root_translation=roots)
            terrain = Terrain({"boxes":[{"pos":[0,0,.1],"size":[1,1,.1],"name":"step"}]})
            append_glb(motion,terrain)
            glb = GLB.load(motion)
            self.assertEqual(glb.doc["nodes"][0]["scale"],[1.1]*3)
            step = next(n for n in glb.doc["nodes"] if n.get("name")=="terrain_box_0")
            np.testing.assert_allclose(step["translation"],[0,.1,0])
            animation = glb.doc["animations"][0]
            channel = next(c for c in animation["channels"] if c["target"]["path"]=="translation")
            np.testing.assert_allclose(glb.accessor(animation["samplers"][channel["sampler"]]["output"]),roots)
            output = document(["mhr_root"],quaternions,[0.,1.],roots,kind="terra",terrain="terrain.json")
            self.assertEqual((output["version"],output["kind"],output["terrain"]),(2,"terra","terrain.json"))

    def test_reference_id_and_checksum_fail_closed(self):
        with self.assertRaises(ValueError):
            reference({"references":{}},"../secret")
        with self.assertRaises(ValueError):
            reference({"references":{}},"missing")
        self.assertFalse(terra.catalog(None)["available"])

    def test_control_clock_preserves_interval_and_short_quaternion_arc(self):
        import mujoco
        from scipy.spatial.transform import Rotation
        from .terra_timing import retime_qpos
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        qpos = np.tile(model.qpos0,(3,1)); qpos[:,0] = [0,.1,.2]
        qpos[:,3:7] = Rotation.from_euler('z',np.array([[170],[180],[190]]),degrees=True).as_quat(scalar_first=True)
        out, velocity, times = retime_qpos(model,qpos,.2)
        self.assertEqual(len(times),21); self.assertAlmostEqual(times[-1],.2)
        np.testing.assert_allclose(out[:,0],times,atol=1e-7)
        np.testing.assert_allclose(velocity[:,0],1,atol=1e-6)
        self.assertAlmostEqual(abs(out[10,3]),0,places=6)
        self.assertAlmostEqual(abs(out[10,6]),1,places=6)
        self.assertAlmostEqual(float(velocity[10,5]),np.deg2rad(100),places=5)

    def test_rejected_job_cancellation_and_cuda_lock(self):
        for cancelled,backend in ((False,"cpu"),(True,"cpu"),(False,"cuda")):
            with self.subTest(cancelled=cancelled,backend=backend), tempfile.TemporaryDirectory(dir=ROOT/"tmp") as folder:
                work=Path(folder); body=work/"heads/abc/body"; body.mkdir(parents=True)
                for name in ("avatar.glb","avatar.json","body_mhr.glb.json"):
                    (body/name).write_text("{}")
                assets=work/"model/safetensors"; assets.mkdir(parents=True)
                for name in mhr.ASSET_FILES:
                    (assets/name).touch()
                event=threading.Event()
                def worker(command,cancel,**kwargs):
                    self.assertEqual(command[command.index("--policy-device")+1],backend)
                    self.assertEqual(command[command.index("--policy-device-index")+1],"2")
                    stage=Path(command[command.index("--work")+1])
                    if cancelled:
                        event.set(); raise gpu.Cancelled("cancelled")
                    (stage/"simulation.json").write_text(json.dumps({"completed":False,"termination":"fall"}))
                    (stage/"baseline.json").write_text(json.dumps({"frames":[{},{}],"duration":.1}))
                    (stage/"terrain.json").write_text("{}")
                    raise RuntimeError("rejected")
                with gpu.execution(backend,2), patch.object(terra,"read_installation",return_value={}), \
                     patch.object(terra,"reference",return_value={}), patch.object(terra.job,"_run",side_effect=worker), \
                     patch.object(terra.gpu,"device_session",return_value=nullcontext()) as lock:
                    options=dict(model_dir=assets.parent,python=sys.executable,assets=work,device=backend)
                    args=(EyeService(work),{"head_id":"abc","reference_motion":"walk"},lambda *a:None,event)
                    if cancelled:
                        with self.assertRaises(gpu.Cancelled):
                            terra.fit(*args,**options)
                        self.assertEqual(list((body/"motions").iterdir()),[])
                    else:
                        result=terra.fit(*args,**options)
                        self.assertFalse(result["accepted"]); self.assertIsNone(result["motion_url"])
                        listing=EyeService(work).list_motions("abc")[0]["urls"]
                        self.assertIn("baseline",listing); self.assertIn("terrain",listing); self.assertNotIn("motion",listing)
                    self.assertEqual(lock.call_count,int(backend=="cuda"))


if __name__ == "__main__":
    unittest.main()
