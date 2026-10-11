# Terra MHR extension

`setup.py` stages the three immutable upstream sources and applies `mhr.patch`
and `holosoma-smpl-optional.patch`. It does not download models or policy data.
See [worker setup and validation](../../server/vhuman/body/TERRA.md).

The Terra patch adds `terra.prepared.PreparedMotion` / `fit_prepared_motion`,
named source-joint indexing, precomputed orientation calibration and source
names in postprocessing. Existing SMPL APIs retain their original defaults.
The patched pipeline uses the existing SQP, contact/collision constraints,
materialization and stability retry policy. It does not create SMPL parameters
or licensed fitted-shape caches. Holosoma's optional SMPL import is separated
from its generic interaction-mesh solver.

Patched upstream files carry modification notices. Their upstream licenses,
copyright notices and NOTICE files remain in the staged checkouts. TERRA is
Apache-2.0; the maintained patches do not grant rights to model assets,
checkpoints, optional SMPL packages or motion datasets. Reference comparisons
must run separately with explicitly provided licensed assets.
