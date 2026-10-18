"""Rebuild the vhuman CUDA development installation without running GPU jobs.

Run through setup_cuda.sh. Scratch stays in repo tmp/; Python storage is shared.
Existing research assets may be imported from an explicitly selected SSH host.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import urllib.request

from server.model_paths import MODELS_ROOT
from server.python_env import cache_path, environment_path
from .relocate_models import link_shared_models, relocate_models

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("envs", "assets", "tools", "build", "register", "scenes", "browser", "check")
BLENDER = "blender-5.2.2-linux-x64"
SOURCES = {
    "pytorch-src": ("https://github.com/pytorch/pytorch.git", "08187d9e0fba026dc8217405802ab5381dc88d90"),
    "flash-attention-src": ("https://github.com/Dao-AILab/flash-attention.git", "14c377950125c70b7a9dabf9c561fca53715ac7d"),
    "sageattention-src": ("https://github.com/thu-ml/SageAttention.git", "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"),
    "musclemimic-upstream": ("https://github.com/amathislab/musclemimic.git", "47b653db2bce25de834360d2ee6b2dd88142bd78"),
    "vhuman-gmr-upstream": ("https://github.com/amathislab/gmr_plus.git", "1b4eb9864360fa822b1290f34592da6393a4d496"),
}
CUTLASS_REVISION = "e05f953a5b3d38adc240df2ff928e0421c2abba3"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Setup:
    def __init__(self, args):
        self.args = args
        # Emscripten computes relative paths inside its cache. Resolve the repo's
        # tmp symlink so those paths are relative to the actual working directory.
        self.state = (ROOT / "tmp/vhuman-dev").resolve()
        self.python = environment_path(ROOT / "tmp/vhuman-cuda-venv", link=True) / "bin/python"
        self.uv_cache = cache_path()
        self.ssh = ["ssh", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes"]
        if args.ssh_identity:
            self.ssh += ["-i", str(args.ssh_identity.expanduser())]
        for relative in ("scratch", "cache/xdg", "logs", "tools/python", "tools/bin"):
            (self.state / relative).mkdir(parents=True, exist_ok=True)
        os.environ.update(
            TMPDIR=str(self.state / "scratch"), TMP=str(self.state / "scratch"),
            TEMP=str(self.state / "scratch"), UV_CACHE_DIR=str(self.uv_cache),
            XDG_CACHE_HOME=str(self.state / "cache/xdg"),
            UV_PYTHON_INSTALL_DIR=str(self.state / "tools/python"),
            UV_PYTHON_BIN_DIR=str(self.state / "tools/bin"),
            HF_HOME=str(self.state / "cache/huggingface"),
            EM_CACHE=str(self.state / "cache/emscripten"),
            VHUMAN_DATA_ROOT=str(args.data_root),
            VHUMAN_MODELS_ROOT=str(args.models_root),
            PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu",
            MUJOCO_GL="egl", UV_HTTP_TIMEOUT="300", UV_LINK_MODE="hardlink",
            CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
        )
        os.environ["PATH"] = str(self.state / "tools/bin") + os.pathsep + os.environ["PATH"]
        os.environ["LD_LIBRARY_PATH"] = str(args.cuda_home / "lib64") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        self.remote_root = None

    def run(self, *command, capture=False, cwd=ROOT, env=None):
        command = list(map(str, command))
        print("+ " + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=cwd, check=True, text=True,
                                stdout=subprocess.PIPE if capture else None, env=env)
        return result.stdout.strip() if capture else None

    def remote(self, *command):
        if not self.args.source_host:
            raise ValueError("Missing asset: supply --source-host or install it locally first")
        return self.run(*self.ssh, self.args.source_host, shlex.join(map(str, command)), capture=True)

    def source_root(self):
        if self.remote_root is None:
            self.remote_root = self.remote("readlink", "-f", self.args.source_repo)
            if not self.remote_root.startswith("/"):
                raise ValueError("source repository did not resolve to an absolute path")
        return self.remote_root

    def source_model(self, relative, legacy):
        """Prefer canonical source storage; accept pre-migration source nodes."""
        path = self.args.source_models + "/" + relative
        try:
            self.remote("test", "-e", path)
        except subprocess.CalledProcessError:
            return self.source_root() + "/" + legacy
        return path

    def transfer(self, source, target, *, directory=True, excludes=()):
        if shutil.disk_usage(target.parent if target.parent.exists() else self.state).free < 6 << 30:
            raise ValueError("Keep at least 6 GiB free before transferring another asset")
        target.parent.mkdir(parents=True, exist_ok=True)
        if directory:
            target.mkdir(parents=True, exist_ok=True)
        command = ["rsync", "-aL", "--partial", "--protect-args", "--info=progress2",
                   "-e", shlex.join(self.ssh)]
        command += ["--exclude=" + item for item in excludes]
        self.run(*command, self.args.source_host + ":" + source + ("/" if directory else ""),
                 str(target) + ("/" if directory else ""))

    def imported_file(self, source, target):
        """Verify imported files against a digest read from the selected source."""
        expected = self.remote("sha256sum", source).split()[0]
        if not target.is_file() or digest(target) != expected:
            self.transfer(source, target, directory=False)
        if digest(target) != expected:
            raise ValueError("Transferred file checksum mismatch: " + str(target))
        return expected

    def backup_path(self, path):
        for root, prefix in ((ROOT, Path()), (ROOT.resolve(), Path()),
                             ((ROOT / "tmp").resolve(), Path("tmp"))):
            try:
                return self.state / "backups" / prefix / path.relative_to(root)
            except ValueError:
                pass
        raise ValueError("Backup path is outside the managed project: " + str(path))

    def directory(self, path):
        """Preserve an obsolete broken symlink before creating local state."""
        if path.is_symlink() and not path.exists():
            backup = self.backup_path(path)
            if backup.exists() or backup.is_symlink():
                raise ValueError("Broken link backup already exists: " + str(backup))
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(backup))
        path.mkdir(parents=True, exist_ok=True)

    def link(self, path, target):
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        if path.exists() or path.is_symlink():
            if path.is_symlink() and not path.exists():
                backup = self.backup_path(path)
                if not backup.exists() and not backup.is_symlink():
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(backup))
                else:
                    raise ValueError("Refusing to replace link with an existing backup: " + str(path))
            else:
                raise ValueError("Refusing to replace existing installation: " + str(path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(target, path.parent))

    def checkout(self, path, url, revision):
        if not (path / ".git").exists():
            if path.exists() and any(path.iterdir()):
                raise ValueError("Refusing to replace non-Git source: " + str(path))
            self.run("git", "init", path)
            self.run("git", "-C", path, "remote", "add", "origin", url)
        head = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                              text=True, capture_output=True)
        if head.returncode:
            mirror = None
            if self.args.source_host:
                try:
                    relative = path.resolve().relative_to((ROOT / "tmp").resolve())
                    remote_path = self.source_root() + "/tmp/" + str(relative)
                    self.remote("git", "-C", remote_path, "cat-file", "-e", revision + "^{commit}")
                    mirror = self.args.source_host + ":" + remote_path
                except (ValueError, subprocess.CalledProcessError):
                    pass
            environment = dict(os.environ, GIT_SSH_COMMAND=shlex.join(self.ssh))
            self.run("git", "-C", path, "fetch", "--depth=1", mirror or "origin", revision, env=environment)
            self.run("git", "-C", path, "checkout", "--detach", revision)
        elif head.stdout.strip() != revision:
            raise ValueError("Source revision differs; preserve it and select a fresh path: " + str(path))
        return path

    def envs(self):
        if not self.python.exists():
            self.run("uv", "venv", "--python", "3.12", self.python.parent.parent)
        lock = ROOT / "server/vhuman/requirements-cuda-dev.lock"
        if self.args.refresh_lock or not lock.is_file():
            lock = self.state / "cuda-requirements.lock"
            self.run("uv", "pip", "compile", ROOT / "server/vhuman/requirements-cuda-dev.txt",
                     "--extra-index-url", "https://download.pytorch.org/whl/cu130",
                     "--index-strategy", "unsafe-best-match", "--output-file", lock)
        self.run("uv", "pip", "sync", "--python", self.python, lock,
                 "--extra-index-url", "https://download.pytorch.org/whl/cu130",
                 "--index-strategy", "unsafe-best-match")
        for name in ("vhuman-rig-venv", "vhuman-texture-venv", "qimg21-ref-venv"):
            self.link(ROOT / "tmp" / name, self.python.parent.parent)
        self.run(self.python, "-B", "-c", "import torch,diffusers,transformers,pxr,onnxruntime,xatlas,cv2,threadpoolctl; "
                 "assert torch.version.cuda and not torch.version.hip; "
                 "assert hasattr(cv2,'HOGDescriptor'); "
                 "print(torch.__version__, diffusers.__version__, transformers.__version__)")
        for name in ("musclemimic-upstream", "vhuman-gmr-upstream"):
            self.checkout(ROOT / "tmp" / name, *SOURCES[name])
        muscle_python = environment_path(ROOT / "tmp/vhuman-musclemimic-py312", link=True) / "bin/python"
        if not muscle_python.exists():
            self.run("uv", "venv", "--python", "3.12", muscle_python.parent.parent)
        self.run("uv", "pip", "install", "--python", muscle_python, "--no-deps",
                 "-r", ROOT / "server/vhuman/body/musclemimic-requirements.lock",
                 "--extra-index-url", "https://download.pytorch.org/whl/cpu", "--index-strategy", "unsafe-best-match")
        self.run("uv", "pip", "install", "--python", muscle_python, "--no-deps",
                 "-e", ROOT / "tmp/musclemimic-upstream", "-e", ROOT / "tmp/vhuman-gmr-upstream")
        from .body.terra_assets import REVISIONS
        for name, url in (("terra", "amathislab/terra"), ("musclemimic", "amathislab/musclemimic"),
                          ("holosoma", "amazon-far/holosoma")):
            self.checkout(ROOT / "tmp/terra-runtime" / name, "https://github.com/" + url + ".git", REVISIONS[name])
        self.run(sys.executable, "-B", ROOT / "ref/terra/setup.py", "--root", ROOT / "tmp/terra-runtime", "--install")

    def assets(self):
        models, data = self.args.models_root, self.args.data_root
        models.mkdir(parents=True, exist_ok=True)
        relocate_models(ROOT, models, apply=True)
        if self.args.shared_models_root:
            link_shared_models(self.args.shared_models_root, models, apply=True)
        data.mkdir(parents=True, exist_ok=True)
        if self.args.source_host:
            source = self.source_root()
            for suffix in ("_rig.json", "_rig.safetensors"):
                name = "sam3d_body_mhr_jit" + suffix
                self.imported_file(source + "/tmp/mhr-native/assets/" + name,
                                   models / "sam3d-body/safetensors" / name)
            self.imported_file(self.args.source_models + "/qwen-image-edit-2511/qwen-image-edit-2511-Q4_K_M.gguf",
                               models / "qwen-image-edit-2511/qwen-image-edit-2511-Q4_K_M.gguf")
            for name in ("ict-facekit", "depth-anything-v2-small"):
                self.transfer(self.source_model("vhuman-rig-models/" + name,
                                                "tmp/vhuman-rig/models/" + name), data / name)
            self.directory(ROOT / "tmp/vhuman-emotion")
            self.imported_file(self.source_model("vhuman/sensevoice-small-q8.gguf",
                                                 "tmp/vhuman-emotion/sensevoice-small-q8.gguf"),
                               models / "vhuman/sensevoice-small-q8.gguf")
            self.transfer(self.source_model("vhuman/mm-fullbody-base", "tmp/vhuman-musclemimic-checkpoint"),
                          models / "vhuman/mm-fullbody-base")
            self.transfer(self.source_model("vhuman/TERRA-4B", "tmp/terra-policy/TERRA-4B"),
                          models / "vhuman/TERRA-4B")
            for name in ("vhuman-ear-synth", "vhuman-contact-tone", "vhuman-mouth-contact",
                         "vhuman-mouth-gi", "vhuman-mouth-occlusion", "terra-native-review"):
                self.transfer(source + "/tmp/" + name, ROOT / "tmp" / name,
                              excludes=("*chrome-profile*/", "Singleton*", "*.sock"))
            for name, remote in (("musclemimic", "tmp/vhuman-musclemimic-assets"),
                                 ("terra", "tmp/terra-assets-v2")):
                receipt = json.loads(self.remote("cat", source + "/" + remote + "/installation.json"))
                write_json(self.state / ("source-" + name + ".json"), receipt)
                for identifier, entry in receipt["references"].items():
                    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", identifier):
                        raise ValueError("Invalid reference identifier")
                    self.imported_file(entry["path"], models / "vhuman/references" / name / (identifier + ".npz"))
        self.run(self.python, "-B", "-m", "server.vhuman.face_assets", "--root", data)
        rig = ROOT / "tmp/vhuman-rig"
        self.directory(rig)
        for name in ("gnm-v3", "ict-facekit", "depth-anything-v2-small"):
            target = data / name
            if not target.exists():
                raise ValueError("Required face installation missing: " + str(target))
        self.link(rig / "models", data)
        if models / "vhuman-rig-models" != data:
            self.link(models / "vhuman-rig-models", data)
        moge = models / "moge-2-vitl/native"
        if not (moge / "native.json").is_file():
            self.run(self.python, "-B", ROOT / "ref/vhuman/moge_reference.py", "--checkpoint",
                     models / "moge-2-vitl/model.pt", "--out", moge)

    def register(self):
        self.register_motion()

    def register_motion(self):
        for name, python, target in (
            ("musclemimic", ROOT / "tmp/vhuman-musclemimic-py312/bin/python", ROOT / "tmp/vhuman-musclemimic-assets"),
            ("terra", ROOT / "tmp/terra-runtime/venv/bin/python", ROOT / "tmp/terra-assets"),
        ):
            module = "server.vhuman.body." + name + "_assets"
            if (target / "installation.json").is_file():
                current = json.loads((target / "installation.json").read_text())
            else:
                current = None
            source_receipt = self.state / ("source-" + name + ".json")
            if not source_receipt.is_file():
                if current is not None:
                    self.run(python, "-B", "-m", module, "doctor", "--assets", target)
                    continue
                raise ValueError("Registration needs the existing provenance receipt: " + str(source_receipt))
            receipt = json.loads(source_receipt.read_text())
            command = [python, "-B", "-m", module, "register", "--assets", target,
                       "--checkpoint-revision", receipt["checkpoint_revision"]]
            if name == "musclemimic":
                if receipt.get("usage") != "noncommercial-research":
                    raise ValueError("Source MuscleMimic use was not registered")
                command += ["--upstream", ROOT / "tmp/musclemimic-upstream", "--gmr", ROOT / "tmp/vhuman-gmr-upstream",
                            "--checkpoint", self.args.models_root / "vhuman/mm-fullbody-base", "--accept-research-terms"]
            else:
                for project in ("terra", "musclemimic", "holosoma"):
                    command += ["--" + project, ROOT / "tmp/terra-runtime" / project]
                if receipt.get("checkpoint_use_acknowledged") is not True:
                    raise ValueError("Source Terra use was not acknowledged")
                flag = "--acknowledge-unspecified-checkpoint-terms" if receipt["checkpoint_license_status"] == "unspecified" else "--accept-checkpoint-terms"
                command += ["--checkpoint", self.args.models_root / "vhuman/TERRA-4B/checkpoint_24416",
                            "--mhr-assets", self.args.models_root / "sam3d-body/safetensors",
                            "--checkpoint-source-url", receipt["checkpoint_source_url"],
                            "--checkpoint-terms-url", receipt["checkpoint_terms_url"], flag]
            if current is None:
                self.run(*command)
            for identifier, entry in receipt["references"].items():
                if current is not None and identifier in current["references"]:
                    continue
                command = [python, "-B", "-m", module, "add-reference", "--assets", target, "--id", identifier,
                           "--trajectory" if name == "musclemimic" else "--motion",
                           self.args.models_root / "vhuman/references" / name / (identifier + ".npz"),
                           "--source-url", entry["source_url"], "--terms-url", entry["terms_url"]]
                if name == "musclemimic":
                    command += ["--accept-research-terms"]
                self.run(*command)
            self.run(python, "-B", "-m", module, "doctor", "--assets", target)

    def tools(self):
        blender = self.state / "tools" / BLENDER
        if not (blender / "blender").is_file():
            if not self.args.source_host or not self.args.source_blender:
                raise ValueError("Select --source-host and --source-blender for the saved scenes' Blender 5.2.2 installation")
            self.transfer(self.args.source_blender, blender)
        version = self.run(blender / "blender", "--version", capture=True)
        if not version.startswith("Blender 5.2.2"):
            raise ValueError("The imported scenes require Blender 5.2.2")
        self.link(self.state / "tools/bin/blender", blender / "blender")
        write_json(self.state / "blender.json", {"version": version, "binary_sha256": digest(blender / "blender")})
        three = self.state / "tools/three"
        if not (three / "package/package.json").is_file():
            three.mkdir(parents=True, exist_ok=True)
            metadata = json.loads(urllib.request.urlopen("https://registry.npmjs.org/three/0.163.0", timeout=60).read())
            import base64
            import tarfile
            archive = three / "three.tgz.partial"
            with urllib.request.urlopen(metadata["dist"]["tarball"], timeout=120) as src, archive.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            algorithm, expected = metadata["dist"]["integrity"].split("-", 1)
            if algorithm != "sha512" or base64.b64encode(hashlib.sha512(archive.read_bytes()).digest()).decode() != expected:
                raise ValueError("Three.js package checksum mismatch")
            with tarfile.open(archive) as tar:
                tar.extractall(three, filter="data")
            archive.unlink()
            write_json(three / "installation.json", metadata["dist"])
        if not shutil.which("em++"):
            emsdk = self.checkout(self.state / "tools/emsdk", "https://github.com/emscripten-core/emsdk.git",
                                  "14c18b569f55138fe4963924162244251f454fb0")
            self.run(emsdk / "emsdk", "install", "5.0.1")
            self.run(emsdk / "emsdk", "activate", "5.0.1")
            for name in ("em++", "emcc"):
                self.link(self.state / "tools/bin" / name, emsdk / "upstream/emscripten" / name)
        if (self.state / "tools/emsdk/.emscripten").is_file():
            os.environ["EM_CONFIG"] = str(self.state / "tools/emsdk/.emscripten")
        self.directory(ROOT / "tmp/vhuman-tools")
        self.run(sys.executable, "-B", ROOT / "vulkan/tools/setup_glslc.py", "--jobs", self.args.jobs)

    def build(self):
        for name in ("pytorch-src", "flash-attention-src"):
            self.checkout(ROOT / "tmp" / name, *SOURCES[name])
        self.checkout(ROOT / "tmp/flash-attention-src/csrc/cutlass",
                      "https://github.com/NVIDIA/cutlass.git", CUTLASS_REVISION)
        if self.args.gpu_arch != "sm_80":
            self.checkout(ROOT / "tmp/sageattention-src", *SOURCES["sageattention-src"])
        self.run("bash", ROOT / "cuda/qimg21/setup_cuda130.sh")
        make = ["make", "-j" + str(self.args.jobs)]
        for directory, targets in (
            ("cpu/vhuman", ["vhuman_models", "libvhuman_motion.so", "libvhuman_training.so", "libvhuman_landmarks.so"]),
            ("cpu/sam3d_body", ["mhr_decode"]), ("cpu/da2", ["da2_depth"]),
            ("cuda/vhuman", ["all"]), ("cuda/sam3d_body", ["test_cuda_sam3d_body", "mhr_decode"]),
            ("cuda/da2", ["da2_depth"]), ("speech", ["all", "cuda"]),
        ):
            self.run(*make, "-C", ROOT / directory, *targets)
        self.directory(ROOT / "ref/pixal3d/deps")
        if not (ROOT / "ref/pixal3d/deps/opencv/usr/include/opencv4/opencv2/core.hpp").is_file() or not (
            ROOT / "ref/pixal3d/deps/openblas/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so"
        ).is_file():
            self.run("sh", ROOT / "ref/pixal3d/setup_native.sh")
        self.run(*make, "-C", ROOT / "cpu/pixal3d")
        self.run(*make, "-C", ROOT / "cuda/pixal3d", "GPU_ARCH=" + self.args.gpu_arch,
                 "CUDA_PATH=" + str(self.args.cuda_home))
        targets = ["native-exact", "native-text-exact", "test_cuda_qimg21_vision", "native-vae",
                   "test_cuda_qimg21_fast", "libq21_fast_attention.so",
                   "libq21_fast_flash.so", "libq21_fast_int8.so"]
        if self.args.gpu_arch != "sm_80":
            targets += ["libq21_fast_sage.so"]
        if self.args.gpu_arch == "sm_120":
            targets += ["libq21_fast_fp4.so"]
        self.run(*make, "-C", ROOT / "cuda/qimg21", "GPU_ARCH=" + self.args.gpu_arch,
                 "CUDA_HOME=" + str(self.args.cuda_home), "PYTHON=" + str(self.python), *targets)
        self.run(*make, "-C", ROOT / "vulkan/vhuman", "GLSLC=" + str(ROOT / "tmp/vhuman-tools/bin/glslc"))
        self.emotion()

    def emotion(self):
        self.directory(ROOT / "tmp/vhuman-emotion")
        self.checkout(ROOT / "tmp/vhuman-emotion/source", "https://github.com/modelscope/FunASR.git",
                      "a57c05bfe2a91b5e0cb0983479634eba3e28ede5")
        self.checkout(ROOT / "tmp/vhuman-emotion/llama", "https://github.com/ggml-org/llama.cpp.git",
                      "803b7fcae893e9caaee3921779628fef83ac0965")
        sensevoice = self.state / "sensevoice-build"
        self.run("cmake", "-S", ROOT / "rdna4/sensevoice", "-B", sensevoice, "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DGGML_HIP=OFF", "-DGGML_CUDA=ON",
                 "-DCUDAToolkit_ROOT=" + str(self.args.cuda_home),
                 "-DCUDA_cudart_LIBRARY=" + str(self.args.cuda_home / "lib64/libcudart.so"),
                 "-DCUDA_cublas_LIBRARY=" + str(self.args.cuda_home / "lib64/libcublas.so"),
                 "-DCUDA_cublasLt_LIBRARY=" + str(self.args.cuda_home / "lib64/libcublasLt.so"),
                 "-DCUDA_culibos_LIBRARY=" + str(self.args.cuda_home / "lib64/libculibos.a"),
                 "-DCUDA_cuda_driver_LIBRARY=" + str(self.args.cuda_home / "lib64/stubs/libcuda.so"),
                 "-DCMAKE_CUDA_ARCHITECTURES=" + self.args.gpu_arch.removeprefix("sm_"),
                 "-DCMAKE_CUDA_COMPILER=" + str(self.args.cuda_home / "bin/nvcc"))
        self.run("cmake", "--build", sensevoice, "--target", "llama-funasr-sensevoice",
                 "--parallel", self.args.jobs)
        self.link(ROOT / "tmp/vhuman-emotion/runtime/llama-funasr-sensevoice",
                  sensevoice / "bin/llama-funasr-sensevoice")

    def check(self):
        # This historical name audits the shared CPU/CUDA/ROCm model layout.
        self.run(self.python, "-B", "-m", "server.vhuman.check_rocm_assets",
                 "--models-root", self.args.models_root)
        self.run(self.python, "-B", "-c", "from server.vhuman.reconstruction.provenance import validate_candidate; "
                 "import sys; print('Face candidate verified:', validate_candidate(sys.argv[1])['geometry_sha256'])",
                 ROOT / "tmp/vhuman-ear-synth/reconstruction_contact")
        self.run(self.python, "-B", "-m", "unittest", "server.vhuman.test_gnm_anatomy",
                 "server.vhuman.test_native_landmarks", "server.vhuman.test_mv_texture",
                 "server.vhuman.test_local_skin_color", "server.vhuman.test_lid_cleanup",
                 "server.vhuman.test_sequence_clearance", "server.vhuman.test_mobile_preprocess",
                 "server.vhuman.test_dev_setup")
        self.run(ROOT / "tmp/terra-runtime/venv/bin/python", "-B", "-m", "unittest",
                 "server.vhuman.body.test_motion", "server.vhuman.body.test_terra",
                 "server.vhuman.body.test_musclemimic")
        self.run("make", "-C", ROOT / "cpu/vhuman", "test")
        self.run("make", "-C", ROOT / "cpu/pixal3d", "test")
        self.run(self.python, "-B", "-c", "from server.vhuman.body.mhr import metadata; "
                 "import sys; print(len(metadata(sys.argv[1])['names']), 'MHR joints verified')",
                 self.args.models_root / "sam3d-body/safetensors")
        self.register_motion()

    def browser(self):
        package = ROOT / "tmp/vhuman-ear-synth/contact_mobile"
        detail = ROOT / "tmp/vhuman-ear-synth/contact_player/detail"
        inputs = [ROOT / "server/vhuman/mobile/native.cpp", ROOT / "server/vhuman/mobile/native.h",
                  package / "avatar.json", detail / "detail.json"]
        inputs += sorted((ROOT / "web").glob("vhuman_mobile*"))
        inputs += sorted((ROOT / "web").glob("vhuman_device_test.*"))
        inputs += [ROOT / "server/vhuman/mobile/browser.py"]
        fingerprint = hashlib.sha256("".join(digest(path) for path in inputs).encode()).hexdigest()
        out = self.state / "browser" / fingerprint[:16]
        if not (out / "native.wasm").is_file() or not (out / "config.json").is_file():
            if out.exists() and any(out.iterdir()):
                out.rename(out.with_name(out.name + ".incomplete-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")))
            self.run(self.python, "-B", "-m", "server.vhuman.mobile.browser", "--package", package,
                     "--out", out, "--three", self.state / "tools/three/package", "--detail", detail)
        for name in ("bindings", "contact", "occlusion", "hash", "audio"):
            self.run("node", ROOT / ("server/vhuman/mobile/test_browser_" + name + ".mjs"))
        write_json(self.state / "browser-build.json", {"out": str(out), "source_fingerprint": fingerprint,
                   "wasm_sha256": digest(out / "native.wasm"), "gpu_jobs_run": False})

    def scenes(self):
        source_receipt = self.state / "source-terra.json"
        if source_receipt.is_file():
            source_root = str(Path(json.loads(source_receipt.read_text())["sources"]["terra"]).parents[2])
        else:
            source_root = self.source_root()
        self.remote_root = source_root
        mappings = [source_root + "=" + str(ROOT), "/mnt/nvme01/models=" + str(self.args.models_root),
                    "/mnt/nvme02/models=" + str(self.args.models_root)]
        for name, filename in (("contact_scene", "head.blend"), ("contact_stress_scene", "head.blend"),
                               ("contact_default_usd", "portable.blend"), ("contact_stress_usd", "portable.blend")):
            source = ROOT / "tmp/vhuman-ear-synth" / name / filename
            target = self.state / "scenes" / name / filename
            if target.is_file():
                receipt = json.loads(target.with_suffix(".relocation.json").read_text())
                if digest(source) != receipt["source_sha256"] or digest(target) != receipt["target_sha256"]:
                    raise ValueError("Relocated scene changed; choose a fresh output: " + str(target))
                continue
            command = [self.state / "tools" / BLENDER / "blender", "-b", "--factory-startup",
                       "--python-exit-code", "1", "--python", ROOT / "server/vhuman/relocate_blender.py",
                       "--", "--input", source, "--out", target]
            for mapping in mappings:
                command += ["--map", mapping]
            self.run(*command)

    def activation(self):
        values = {
            "VHUMAN_MODELS_ROOT": self.args.models_root, "VHUMAN_DATA_ROOT": self.args.data_root,
            "VHUMAN_BLENDER": self.state / "tools" / BLENDER / "blender",
            "VHUMAN_THREE": self.state / "tools/three/package", "VHUMAN_RIG_PYTHON": self.python,
            "PYTHON": self.python,
            "TMPDIR": self.state / "scratch", "UV_CACHE_DIR": self.uv_cache,
            "XDG_CACHE_HOME": self.state / "cache/xdg", "EM_CACHE": self.state / "cache/emscripten",
            "HF_HOME": self.state / "cache/huggingface", "PYTHONDONTWRITEBYTECODE": "1",
        }
        if (self.state / "tools/emsdk/.emscripten").is_file():
            values["EM_CONFIG"] = self.state / "tools/emsdk/.emscripten"
        lines = ["# Generated by server/vhuman/setup_cuda.sh; source this file from any directory."]
        lines += ["export " + key + "=" + shlex.quote(str(value)) for key, value in values.items()]
        lines += ['export PATH=' + shlex.quote(str(self.state / "tools/bin") + ":" + str(ROOT / "tmp/vhuman-tools/bin")) + ':"$PATH"',
                  'export LD_LIBRARY_PATH=' + shlex.quote(str(self.args.cuda_home / "lib64")) + ':"${LD_LIBRARY_PATH:-}"']
        (self.state / "activate.sh").write_text("\n".join(lines) + "\n")
        server = [self.python, "-B", "-m", "server.vhuman.app", "--backend", "cuda",
                  "--models-root", self.args.models_root, "--rig-python", self.python, "--qwen-python", self.python,
                  "--musclemimic-python", ROOT / "tmp/vhuman-musclemimic-py312/bin/python",
                  "--musclemimic-assets", ROOT / "tmp/vhuman-musclemimic-assets", "--musclemimic-device", "cpu",
                  "--terra-python", ROOT / "tmp/terra-runtime/venv/bin/python",
                  "--terra-assets", ROOT / "tmp/terra-assets", "--terra-device", "cpu"]
        launcher = self.state / "run.sh"
        launcher.write_text("#!/bin/sh\nset -eu\n. " + shlex.quote(str(self.state / "activate.sh")) +
                            "\ncd " + shlex.quote(str(ROOT)) + "\nexec " + shlex.join(map(str, server)) + ' "$@"\n')
        launcher.chmod(0o755)
        config_file = self.state / "config.json"
        previous = json.loads(config_file.read_text()) if config_file.is_file() else {}
        configuration = {"root": str(ROOT), "models_root": str(self.args.models_root),
                   "data_root": str(self.args.data_root), "cuda_home": str(self.args.cuda_home),
                   "gpu_arch": self.args.gpu_arch, "gpu_validation": "pending",
                   "face_candidate": str(ROOT / "tmp/vhuman-ear-synth/reconstruction_contact"),
                   "source_repo": self.remote_root or previous.get("source_repo"),
                   "source_host": self.args.source_host or previous.get("source_host")}
        build = self.state / "build.json"
        if build.is_file():
            configuration["gpu_arch"] = json.loads(build.read_text())["gpu_arch"]
        write_json(config_file, configuration)
        body = (ROOT / "tmp/terra-native-review/avatar-varied").resolve()
        write_json(self.state / "terra-stand-request.json", {
            "avatar": str(body / "avatar.glb"), "avatar_json": str(body / "avatar.json"),
            "body_meta": str(body / "body_mhr.glb.json"), "reference_motion": "native-stand",
            "mhr_assets": str(self.args.models_root / "sam3d-body/safetensors"),
            "terrain": "auto", "policy_actions": "sampled", "seed": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phases", nargs="+", choices=PHASES + ("emotion",), default=list(PHASES))
    parser.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    parser.add_argument("--shared-models-root", type=Path, help="alias local weights from another volume instead of copying")
    parser.add_argument("--data-root", type=Path, help="face assets; defaults to MODELS_ROOT/vhuman-rig-models")
    parser.add_argument("--gpu-arch", choices=("sm_80", "sm_89", "sm_120"), default="sm_80")
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda-12.9"))
    parser.add_argument("--source-host")
    parser.add_argument("--source-repo", default="work/gemm/pixal3d")
    parser.add_argument("--source-models", default="/mnt/nvme01/models")
    parser.add_argument("--source-blender", default="/home/syoyo/local/" + BLENDER)
    parser.add_argument("--ssh-identity", type=Path, default=Path("~/.ssh/id_ed25519"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--refresh-lock", action="store_true", help="resolve requirements into tmp/ for review instead of using the committed CUDA lock")
    args = parser.parse_args()
    if args.data_root is None:
        args.data_root = args.models_root / "vhuman-rig-models"
    if args.jobs < 1 or args.jobs > 32:
        parser.error("--jobs must be between 1 and 32")
    if args.source_host and not re.fullmatch(r"[a-zA-Z0-9_.@-]+", args.source_host):
        parser.error("invalid SSH host")
    for key in ("models_root", "data_root", "cuda_home"):
        setattr(args, key, getattr(args, key).expanduser().absolute())
    if args.shared_models_root:
        args.shared_models_root = args.shared_models_root.expanduser().absolute()
    if any(phase in args.phases for phase in ("build", "emotion")) and not (args.cuda_home / "bin/nvcc").is_file():
        parser.error("select an installed full CUDA toolkit with --cuda-home")
    os.chdir(ROOT)
    setup = Setup(args)
    for tool in ("uv", "git", "rsync", "curl", "make", "cmake", "ninja", "gcc", "g++", "ffmpeg", "ffprobe", "node"):
        if not shutil.which(tool):
            parser.error("missing prerequisite: " + tool + " (see server/vhuman/CUDA_SETUP.md)")
    for phase in args.phases:
        print("\n=== vhuman CUDA setup: " + phase + " ===", flush=True)
        receipt = setup.state / (phase + ".json")
        status = {"started": datetime.now(timezone.utc).isoformat(), "status": "running",
                  "gpu_arch": args.gpu_arch, "gpu_jobs_run": False}
        write_json(receipt, status)
        try:
            getattr(setup, phase)()
        except Exception as exc:
            status.update(status="failed", error=str(exc))
            write_json(receipt, status)
            raise
        status.update(status="complete", completed=datetime.now(timezone.utc).isoformat())
        write_json(receipt, status)
        setup.activation()
    print("\nSetup complete for selected phases; GPU validation is pending.", flush=True)
    print("source " + str(setup.state / "activate.sh"), flush=True)


if __name__ == "__main__":
    main()
