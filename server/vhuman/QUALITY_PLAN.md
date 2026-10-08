# Virtual-human closeout

Historical closeout below describes the September independence pass. The user
reopened face fitting and texture quality work on October 8, including eyeballs,
teeth, gums and tongue, with body/clothing/hair excluded. Current experiments,
validation evidence and remaining gates are tracked in
[QUALITY_ITERATIONS.md](QUALITY_ITERATIONS.md). Experimental candidates do not
replace the accepted material until the current gates pass.

The user accepted the existing fitting quality on 2026-09-28. The subsequent
independence pass replaces the asset-derived eye profile and engine-specific
material equations, defaults and exports. It does not reopen fitting research.

## Independent outputs

The active work directory is `tmp/vhuman-independent/`. Eye geometry now uses
user-editable physical measurements with synthetic defaults, an analytic angular
UV map, standard optical equations and original procedural materials. Refer to
[README.md](README.md) for the input schema and provenance boundaries.

Old outputs in `tmp/vhuman/` and the private pre-rewrite recovery bundle are
legacy local records, excluded from Git and from the active asset library.
They must not be distributed as independent outputs. Source portraits and raw
Pixal3D heads may be reused to regenerate the procedural eyes and skin maps;
their model/output terms still apply. No generated assets are committed.

Regenerated references (2026-09-28): man `291dfa911553`, woman `650ac67354cd`.
Both were visually checked in analytic and portable front views. Existing
fitting defects remain visible; these checks do not claim a quality improvement.

## Accepted limitations

Single-view fitting retains broad overhangs, local folds, uneven canthi and
lower-lid contour errors. Caruncles require a bounded hit on retained head
geometry; unsupported corners omit them. Skin masks do not provide full semantic
hair/beard/hairline segmentation. Optional illumination removal estimates broad
shading, not ground-truth albedo. Procedural pores and freckles remain default.
Automatic head fitting uses the synthetic default eye dimensions; the standalone
eye supports customized dimensions. No experimental reconstruction replaces the
production fitter.

## Validation

Run `python3 -m server.vhuman.test_all` from the repository root. Optional
triangulation coverage requires `requirements-remesh.txt`; browser coverage
requires Chrome and its web dependencies. Detailed local validation and security
scan logs are retained under `tmp/vhuman/dev/prepush/`.

Final validation: **88 tests passed in 225.222 seconds**, including browser
and optional triangulation coverage, with no skips. Command:
`PYTHONPATH=tmp/vhuman/dev/quality/deps:. python3 -m server.vhuman.test_all`.
`compileall` and `git diff --check` also passed.

The independent eye CPU/WebGL preview comparison measured mean absolute channel
difference 0.27/255 overall and 0.94/255 in the iris (384-pixel preview).

No push is authorized or performed.
