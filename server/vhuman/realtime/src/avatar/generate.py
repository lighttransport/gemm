"""Clean identity provider. The sole reference is a newly generated neutral image."""
import json
from pathlib import Path
import time
from .provenance import sha256, verify_files

MODEL = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PROMPT = ("Photorealistic studio headshot of an original fictional adult woman, front view, "
          "face centered and fully visible, natural skin pores, hair behind ears, no jewelry, "
          "plain grey background, soft even fixed lighting, neutral expression, relaxed closed lips.")


def generate(output, cache, seed=7, expressions=False, resume=False):
    import torch
    from diffusers import Flux2KleinPipeline
    from ....rig.exprdata import EXPRESSIONS, KEEP
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    existing = None
    if (output / "manifest.json").exists():
        if not resume: raise FileExistsError("identity exists; choose a fresh output directory or explicit --resume")
        existing = json.loads((output / "manifest.json").read_text())
        if existing.get("model") != MODEL or existing.get("revision") != REVISION or existing.get("seed") != seed:
            raise ValueError("resume requires the same generator revision and seed")
        verify_files(existing["provenance"], output)
    pipe = Flux2KleinPipeline.from_pretrained(MODEL, revision=REVISION, cache_dir=str(cache), torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    specs = [("neutral", PROMPT, {})]
    if expressions: specs += [(name, KEEP + " " + prompt, controls) for name, (prompt, controls, _) in EXPRESSIONS.items()]
    receipts, records, neutral = [], [], None
    if existing:
        from PIL import Image
        receipts, records = existing["provenance"], existing["references"]
        neutral = Image.open(output / "neutral.png").convert("RGB")
    for index, (name, prompt, controls) in enumerate(specs):
        if any(record["name"] == name for record in records): continue
        kwargs = dict(prompt=prompt, height=512, width=512, num_inference_steps=4, guidance_scale=1.,
                      generator=torch.Generator("cpu").manual_seed(seed + index))
        if neutral is not None: kwargs["image"] = neutral
        started = time.monotonic()
        image = pipe(**kwargs).images[0]
        image.save(output / (name + ".png"))
        if neutral is None: neutral = image
        receipt = dict(path=name + ".png", source="https://huggingface.co/" + MODEL, revision=REVISION,
                       license="Apache-2.0", license_scope="original generated artifact released by this project; generator is Apache-2.0",
                       sha256=sha256(output / (name + ".png")), roles=["appearance-training", "rig-training"],
                       seed=seed + index, conditioning=[] if index == 0 else [receipts[0]["sha256"]])
        receipts.append(receipt)
        records.append(dict(name=name, path=receipt["path"], controls=controls, prompt=prompt,
                            generation_seconds=time.monotonic()-started, controls_are_approximate=True))
        (output / "manifest.json").write_text(json.dumps(dict(format="vhuman.clean_identity.v1", model=MODEL,
             revision=REVISION, seed=seed, references=records, provenance=receipts, requires_manual_identity_and_pose_QA=True), indent=2))
    return dict(output=str(output), images=len(records), model=MODEL, revision=REVISION)
