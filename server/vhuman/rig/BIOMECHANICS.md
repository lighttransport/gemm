# CPU face and mouth biomechanics

The opt-in full mode uses native LightGeom distributed muscle stress,
implicit tension-only fiber constraints, energy-based Neo-Hookean tissue,
volume constraints, a force-driven rigid mandible, and swept vertex/triangle and edge/edge contacts. PyTorch CPU fits
anatomy and captured expressions. Geometry targets drive inverse muscle
control; they are not added to the simulated result. This round changes
geometry and motion only.

This remains an experimental offline path. The retained subject passes source
conditioning and a short bilabial hold; wide-vowel, labiodental, tongue-tip and
larger brow/blink/gaze requests remain unqualified. Complete-utterance and
acoustic quality are unvalidated. Reports reject unsafe or inaccurate bakes
instead of treating a safe stationary fallback as successful articulation.

## Setup and use

```sh
sh server/vhuman/rig/setup_face_sources.sh --teacher-only
sh server/vhuman/setup_cpu_fit.sh python3.12 --full
PY=tmp/vhuman-cpu-venv/bin/python
$PY -m server.vhuman.cli --backend cpu --work WORK rig-fit-anatomy --head HEAD --threads 4
# Optional: --captures TAKE_ID ... for same-identity accepted video takes.
# Optional: --fit-sockets for bounded ocular geometry conditioning before meshing.
# Optional: --fit-mouth for bounded lip and hidden oral-lining conditioning.
# Optional: --fit-dentition for bounded rigid tooth-crown separation.
# Optional: --fit-tongue for local fold removal with the anterior tip fixed.
# Optional: --fit-tongue-root for bounded ventral root fitting; includes fold removal.
# Optional: --sliding-skin for skull-normal support with tangential skin tethers.
# Optional: --tangent-fibers to fit facial directions to the exterior shell.
# Optional: --harmonic-tongue-fibers for authored compartments and CPU harmonic directions.
# Optional: --extrinsic-tongue-cables for authored external tongue muscle supports.
# Combined geometry/mechanics candidate for review:
$PY -m server.vhuman.cli --backend cpu --work WORK rig-fit-anatomy --head HEAD --threads 4 \
    --fit-sockets --fit-mouth --fit-dentition --fit-tongue-root \
    --sliding-skin --tangent-fibers --harmonic-tongue-fibers
$PY -m server.vhuman.cli --backend cpu --work WORK rig-speech --backend cpu \
    --head HEAD --language ja --text 'こんにちは。' --motion-model biomechanics --threads 4
$PY -m server.vhuman.cli --backend cpu --work WORK rig-soft-tissue \
    --head HEAD --take TAKE_ID --mode full --threads 4
```

Use `--language en --text 'My name is Ryan.'` for English. Text generation,
alignment, motion fitting, simulation, review rendering and encoding use CPU
paths. Mixed server configurations should supply the CPU fitting interpreter
with `--rig-python`. HTTP jobs are `rig_fit_anatomy`, `rig_soft_tissue` with
`mode: full`, and `rig_speech` with `motion_model: biomechanics`.

For an isolated reconstruction without promoting the accepted head:

```sh
$PY -m server.vhuman.rig.anatomy fit CANDIDATE/rig --out NEW_ANATOMY_DIRECTORY --threads 4
$PY -m server.vhuman.rig.biomechanics CANDIDATE/rig TAKE_DIRECTORY \
    --package NEW_ANATOMY_DIRECTORY --out NEW_BAKE_DIRECTORY --threads 4
# Optional: --inverse-workers 4 runs sensitivity trials on four independent
# CPU bodies. It requires extra memory; the default remains one worker.
```

The standalone module, `rig-speech`, and full `rig-soft-tissue` accept
`--mechanics-solver coupled` (HTTP `mechanics_solver: coupled`) for the
experimental jaw/tissue/contact solve. It requires `--state-precision auto`
or `float64`. The default remains `partitioned`. For a separate review bake:

```sh
$PY -m server.vhuman.rig.biomechanics CANDIDATE/rig TAKE_DIRECTORY \
    --package NEW_ANATOMY_DIRECTORY --out NEW_BAKE_DIRECTORY --threads 4 \
    --mechanics-solver coupled --state-precision float64
```

Reports record the actual contact transport, jaw attachment mode, inertia,
damping and opening-dependent translation track. Reports also retain hashes of
the Python implementation files present at bake start, alongside the immutable
native solver hash. New anatomy packages fit that
track to alternate openings of the subject's native rigid dental rig and
validate withheld openings with a fixed hinge. This is geometric rig matching,
not a measured condylar trajectory. Existing packages retain the previous track
unless refitted. The coupled option uses the same force and volume acceptance
limits; choosing it does not qualify an otherwise inaccurate speech pose.

The low-level native session also offers experimental, disabled-by-default
`inertial_repair_sweeps` and `volume_rescue_limit` settings. They use local mass
projections to reduce connected-tissue locking while retaining the continuous
signed-volume guard. Worker sessions preserve these settings. They are not
exposed by the bake CLI pending broader subject and motion validation.
`--mechanics-solver block` (low-level `jaw_mode='coupled_block'`) projects all moving jaw
attachments simultaneously through a three-coordinate Schur complement. This
includes tissue reaction in the reduced inertia and applies one shared volume
guard fraction to tissue, jaw and XPBD multipliers. It avoids vertex-order
errors in fully bone-attached small tetrahedra. This is an experimental option;
the default remains `partitioned`, and every bake still has to pass all geometry
checks.

Anatomy preparation also accepts `--extrinsic-tongue-cables` (HTTP
`extrinsic_tongue_cables: true`). This adds six bilateral styloglossus,
hyoglossus and palatoglossus tension-only paths with distributed free-tissue
insertions. Existing geometry and internal fiber arrays remain unchanged.
Origins are inferred fixed head-local supports, including a fixed hyoid
approximation; they are not measured attachment sites or moving hyoid anatomy.
Force scales derive from the existing authored stress-volume fields and bundle
lengths. The package records these assumptions and rejects proper straight-path
crossings through teeth at rest. This is not muscle wrapping or continuous
path collision detection. Native support-reaction diagnostics report impulses
in N s. Synthetic force, moment, volume-guard and worker-replay checks pass;
full-subject tongue articulation with these paths remains unqualified.

`rig-speech` and full `rig-soft-tissue` also accept `--inverse-workers` (HTTP
`inverse_workers`). It must not exceed `threads`. Worker sessions restore the
same complete initial state for each sensitivity probe; the accepted trajectory
is advanced only on the main session. Results retain muscle order even when
probes finish in a different order. Cancellation joins native calls before
destroying snapshots or bodies. New solvers poll cancellation within passive
tetrahedral and active-fiber iterations, with transactional Python snapshot
rollback. Worker callbacks are installed only while a cancellation source is
attached. Legacy libraries still stop between substeps. Reports record native
checkpoint availability. Callback wrappers use weak owners and are released
after native destruction so closed worker arrays do not await cyclic GC.
Zero-iteration bakes do not allocate workers.
Workers are allocated lazily when a sensitivity probe is actually needed.
Their `--inverse-iterations` option (HTTP `inverse_iterations`, default 2)
sets 0–20 fitting iterations. The standalone biomechanics module uses
`--iterations` for the same setting. Zero iterations evaluate the authored
drive as a diagnostic; they do not establish expression or speech fidelity.

Inverse fitting stops early when the validated native trial already has
regional RMS below 75% of each acceptance limit, all-surface RMS below 0.25 mm,
extra contact residuals below 0.025 mm, valid triangle areas and zero crossings.
The final exported geometry still undergoes the complete independent checks.
For a source-clear neutral state, full-duration fitting proposals must also
retain valid exported triangle areas and zero head-local/exported crossings
before a trial can be committed. An unsafe prior falls back to the preceding
safe drive; line search can then find a bounded control. Derivative probes
remain estimates and are never committed. Already-intersecting source packages
retain diagnostic review bakes and still fail the final source/crossing gate.
Reports identify the filter and count rejected geometric trials.
When consecutive complete geometry targets match exactly, the next frame starts
from the previous fitted drive. This remains a new full native forward trial,
including current velocity, activation dynamics and contact checks. Unsafe
warm starts try the authored prior and previous drive without duplicate trials.
Reports record the seed source; no stored surface or physical state substitutes
for the new trajectory. The authored prior remains the regularization target.
Reports distinguish early stopping from completed fitting iterations and record
actual native steps, subdivision retries and allocated workers. Unused cached
responses still age with accepted frames.

`--sensitivity-reuse-frames N` (HTTP `sensitivity_reuse_frames`, default 0)
optionally reuses a local full-surface muscle response for up to N subsequent
frames. Samples can change between frames because the cache covers the entire
surface. Every proposed control still runs the complete native dynamics and
line search. A model that fails to improve the actual objective triggers fresh
native sensitivity trials. Cache age, owner and interval checks prevent
unbounded or incompatible reuse; cancellation discards the model. This is an
approximate optimizer and can choose different controls from always-fresh
fitting. It never substitutes cached geometry for the simulated result or
changes the geometry gates.

`--sensitivity-probe-substeps N` (HTTP `sensitivity_probe_substeps`, default 0)
optionally estimates the local response from N native 240 Hz steps (1–48),
using the full interval when N is zero or already covers it. An activation and
inertial response estimate is calibrated against a full-duration probe of the
muscle most relevant to the current target. That column uses its actual full
native response; other columns remain approximations. A useful calibration
trial is retained as a validated improvement. Every optimization proposal
still runs the complete frame interval and the unchanged geometry checks.
Reports record probe duration, calibration muscle, scale, direction agreement
and acceptance. This can change fitted controls and is not an exact substitute
for full-horizon derivatives.

Requirements: Linux/ELF, GCC, CMake, Ninja, FFmpeg, the existing CPU fitting
environment, and separately installed GNM/aligner/TTS assets. The existing native rig
teacher requires x86-64 AVX2/FMA. The standalone
native library is built in `tmp/vhuman-full-native-build/` and installed
atomically under `tmp/vhuman-full-native/`; it links CPU C code
and libm, with no graphics, GPU SDK, USD parser or PyTorch runtime linkage.
Each bake loads a verified copy under `tmp/vhuman-native-solvers/<sha256>.so`.
An ordinary rebuild cannot truncate a running solver or change its recorded
binary hash. Existing cached bytes are verified and never overwritten.
New solvers transport positions and velocities between native dynamics and
embedded contact correction in float64, preserving the internal state used by
the volume guard. This avoids a float32 proxy differing from the guarded body.
Attachment descriptors and exported surface samples retain their existing
float32 formats. Archived libraries without the optional double state API use
the legacy float32 bridge; `state_transport_precision` identifies the path.
Double transport can change a trajectory and requires its own bake validation.
`--state-precision auto|float32|float64` (HTTP `state_precision`) selects the
bridge explicitly for reproducible comparisons. `auto` uses double transport
when the native API is present. Explicit `float64` rejects an older library
without that API; workers retain the owner's selected precision.
For example, the retained `/m/` authored drive changes from a passing 0.158 mm
float32 gap to 0.202 mm and eight crossings with double transport. Its regional
tracking and volume checks still pass, but closure and crossings require
refitting; the float32 acceptance must not be transferred to that trajectory.

## Anatomy and fitting

Anatomy preparation requires GNM bind metadata and a geometry-matched
expression map. It keeps the full GNM skin, tongue, teeth, gingiva and eyes.
The tongue uses an authored convex volume envelope with a 0.25 mm support
margin; the original folded tongue surface is retained and transported exactly.
This solid is an inferred approximation, not a measured anatomical boundary.
Face and gingiva receive closed tissue shells
with reduced thickness near opposing surfaces. The pinned CPU fTetWild
bindings (`wildmeshing==0.4.1`) prepare volumes in a cancellable worker, with
a default 60-minute limit per region (`--mesh-timeout`, HTTP `mesh_timeout`,
60–7,200 seconds). Tongue and gingiva disable surface simplification
to preserve thin details. This offline step can be expensive. Invalid
tetrahedra, unstable bindings and oversized packages are rejected.
Verified regional volumes are reused under `tmp/vhuman-tetrahedra/` when
canonical geometry bytes, worker source, pinned mesher version, edge target,
thread count and surface-preservation setting match exactly. Each hit checks
the stored hash and positive-volume/size bounds; corrupt entries fail clearly.
Input changes always remesh, and original-surface embeddings are refit even on
a hit. Per-region `*-meshing-cache.json` receipts record reuse and volume hashes.
This avoids repeating tetrahedralization for muscle/attachment-only changes.
A real repeat CLI build with a 60-second worker budget reuses all four
regional volumes and preserves all 22 existing package arrays exactly. Its
only added array is the calibrated expression seed; physical geometry,
fibers, embeddings and attachments are unchanged. Source conditioning and
original-surface embedding still run, so a cache hit is not an instant build.
Bounded affine surface transport permits up to 0.1 mm outside the meshed boundary
with capped weight amplification; exported neutral vertices remain exact.
The centroid search expands from 64 to 256 and 1,024 candidates only for
unsupported samples; physical support and weight limits stay unchanged.
Difficult surface regions are not deleted during animation to make a take pass.
`rest_surface_preflight` reports proper intersections in the retained source
surface; preparation can succeed while source geometry still needs repair.

`rig-fit-anatomy --fit-sockets` (HTTP `fit_sockets: true`) conditions the
inferred hidden ocular cups around the fixed convex eye envelopes. A CPU
disk parameterization wraps each cup behind its globe; radial clearance and
visible triangle orientation constraints retain the original topology.
Visible lid displacement is limited to 1.5 mm and hidden cup displacement to
30 mm. Identity coefficients and eye vertices stay fixed. An independent native
BVH must verify zero skin/eye crossings, no additional skin self-crossings,
and fewer total source crossings before an edit is accepted. Already clear
sockets retain their exact source geometry. This is an
authored geometry prior for inferred internal skin, not measured orbital anatomy.
The package includes `socket_fit.json` and a hash-bound bind-space correction;
CPU rig and captured motion targets transport that correction once, before
the physical solve. The skin volume is rebuilt from the fitted source.

`--fit-mouth` (HTTP `fit_mouth: true`) first separates crossed upper/lower
lips along the head's authored up axis, with a 1.5 mm visible displacement
limit. It then fits the inferred lining outside a convex envelope of the fixed
tongue, gums and teeth, with a 30 mm hidden displacement limit. Swept skin
contact constraints oppose newly introduced self-crossings during this fit.
The native query returns bounded edge/triangle witnesses using the same
predicate as the bake gate; the final independent check requires fewer total
crossings, no increase in skin self-crossings, unchanged fixed anatomy and
preserved visible triangle orientation. `mouth_fit.json` records the result.
Already clear regions stay unchanged. This convex envelope is an authored
prior for hidden geometry, not an anatomical measurement.
Optional `--mouth-fairing .25` (HTTP `mouth_fairing`, 0–0.5) applies displacement
fairing on the hidden lining, decaying during fitting to preserve fixed lips
and the source shape. The default is zero, retaining the validated original
lining fit. Fairing can make meshing more expensive; the tested smoother
candidate needed a longer `--mesh-timeout 3600`. The fitted source
still has to pass the same crossing and displacement bounds before meshing.

`--fit-dentition` (HTTP `fit_dentition: true`) identifies disconnected tooth
crowns from source topology and separates overlapping crowns by rigid
translations, bounded to 1 mm per tooth. It preserves crown shape and all
non-dental vertices. Shared gingival seam vertices follow their tooth, and
the affected gingival volumes are rebuilt. Independent checks reject new
crossings at each gum, tongue, skin and eye interface, or new gingival
self-crossings. Shared gingival triangles must retain at least 10% of their
original oriented area. `dentition_fit.json` records each crown translation. Existing
crossings inside one tooth are rejected rather than deformed away.

`--fit-tongue` (HTTP `fit_tongue: true`) fairs only two edge rings around
existing tongue self-crossings, with a 1 mm displacement bound and the
frontmost fifth fixed. An oriented-area constraint opposes collapse and face
reversal. The final gate requires no tongue self-crossings, at least 10% of
each triangle's original oriented area, and no increased crossings at any
other anatomy interface. The unchanged convex tongue volume can be retained
when its meshing input matches exactly; surface embeddings are recomputed.
This conditions an authored source prior, not measured tongue anatomy.
`tongue_fit.json` records affected vertices and all interface checks.

`--fit-tongue-root` (HTTP `fit_tongue_root: true`) also conditions an inferred
ventral tongue base against the fixed lower arch. Staged radial fitting has a
9 mm subject-scale displacement limit, positive oriented-area bounds, and a
final independent surface check. The upper 60% of input tongue vertices,
including the dorsal tip, are restored exactly; other anatomy stays fixed.
The returned geometry must have no tongue self-crossings, no remaining
lower-arch crossings and no increased crossings at another interface.
`tongue_root_fit.json` records the stages, bounds and interface checks. This can
make a substantial hidden-base correction and must be reviewed as an authored
anatomical prior. It does not establish measured tongue shape. Its changed
convex volume envelope is remeshed and original-surface embeddings are refit.

These fitting flags can be combined. Bind corrections and accepted expression
corrections follow the CPU rig's weighted joint rotations before being added
to evaluated rig targets. Joint translations never act on correction vectors.

`--harmonic-tongue-fibers` (HTTP `harmonic_tongue_fibers: true`) solves linear
tetrahedral Laplace problems on CPU for longitudinal, transverse, vertical and
fan directions. Authored dorsal/ventral bands separate the longitudinal
families, and anterior/medial/posterior bands separate the genioglossus fields.
Each field redistributes its authored stress-volume integral before applying
the existing 3,500 Pa summed occupancy cap. Other tongue directions retain
their authored priors; skin, jaw paths, geometry and attachments stay fixed.
The construction follows the geometric idea in
[Gomez et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6050131/), using
our own implementation and authored boundary patches. It does not use MRI
segmentations or measured fiber directions. `tongue_fibers.json` records
boundaries, solver residuals, support and the remaining anatomical uncertainty.

`anatomy/anatomy.json` and `anatomy.npz` bind the anatomy to rig, geometry and
model hashes. Bake reports include a canonical metadata hash as well as the
array hash, distinguishing control-prior changes on the same mesh.
Original surface vertices use barycentric volume transport;
contact correction is distributed back through the same embedding. Teeth
remain rigid. Gingiva follows its dental arch while retaining tissue response.
The lower arch follows muscle-driven jaw rotation, lateral slide and forward
slide, with bounded condylar motion. Eyes follow fitted gaze transforms.

Facial/tongue fiber fields, mandibular muscle paths, hidden skull/palate
structure inferred from interior geometry, thickness, strength and stiffness
are authored priors. They are not measured physiology. Accepted captures can
refine the jaw hinge and expression mapping. Mapping fits hold out a take or
contiguous tail and use 2% improvement/5% regional regression gates. Identity
stays frozen; force/material scales are not inferred from monocular video.
Control targets use the actual evaluated CPU rig exterior, including jaw
skinning and correctives. Accepted expression fits change only the PCA
component of that target relative to the stored calibrated seed. The seed
is not added again to evaluated teacher geometry; legacy packages without
a seed retain their prior interpretation. Lower dental and tongue goals use one local jaw
transform. Direct captured targets factor out the complete GNM neck/head
transform before simulation and retain local eye motion. Captured cranial
rotation and translation are applied once after the solve; rigid eyes use
their recorded joint transforms relative to that head transform. These paths
use the exact captured transforms rather than the approximate semantic head
and gaze controls. Jaw-hinge fitting uses the same head-local factorization.
The morph-only
PCA map is not treated as a complete joint-motion model.

New anatomy packages label the authored atlas as
`vhuman.authored_surface_bands.v3`: facial anchors select frontal exterior
skin in a 3 mm XY neighborhood, preventing exact XY matches on the posterior
scalp. Central lip bands shorten horizontally, and separate lateral sectors
support pucker/funnel controls. The former
circular tangent could pull central lip margins apart during bilabial drive.
Previously prepared packages retain their own fiber fields and hashes; rebuild
anatomy to use the new atlas. Full speech and rounded-vowel validation remain
required before accepting an animation.

`--sliding-skin` (HTTP `sliding_skin: true`) replaces isotropic pinning on
frontal exterior skin with strong support along the inferred skull normal
and weaker tangential tethers. Upper scalp and side boundaries retain fixed
support. Mandibular, gingival, tongue and hidden lining attachments keep their
original constraints. Tangential stiffness scales with lumped nodal area;
the default density is an authored 10,000 N/m³. This is a local tangent-plane
support prior, not a measured skull contact or ligament model. It remains
opt-in, and its frames and compliances are hash-bound in the anatomy package.
The native solver projects anisotropic corrections through the same volume
guard. Attachment reactions are raw XPBD multipliers divided by substep
length squared; the legacy native function name includes “impulses”, but
its returned quantities have units kg·m.
The anterior eyelid band releases inferred skull pins, keeping posterior
orbital and canthal support. The connected tissue and globe contacts support
that band. Its geometric bounds and released node IDs are recorded under
`orbital_support_release`; this is an authored support model.

`--tangent-fibers` (HTTP `tangent_fibers: true`) fits facial directions to
nearby exterior tangent planes and removes deep lining from facial muscle
support. The `vhuman.authored_surface_bands.v5_ocular` atlas also wraps
orbicularis oculi directions around an ellipse fitted to exterior eyelid
bounds, before transporting them onto the curved surface. It preserves other
muscle families and keeps the original stress scale. This fits field geometry,
not measured fascicles, forces or connective-tissue properties.

The head-local neutral configuration assumes balanced external acceleration.
Cranial motion is applied once after the local solve. This mode does not model
airflow or vocal-tract acoustics. Simulation starts at 240 Hz and retries unsafe
steps with bounded subdivision. Authored viscous mandibular resistance damps
the partitioned jaw/tissue feedback; it is not measured physiology. Every
inverse-control trial restores tissue,
muscle, jaw and attachment state before evaluation.
An unsafe semantic seed falls back to the preceding accepted drive. The first
frame uses the equilibrated zero drive as its history, so an unsafe request
is not simply retried as its own fallback.
Accepted trials are committed from complete native snapshots instead of being
simulated a second time. Snapshot restoration includes the native implicit
projector's private activation, alongside activation dynamics, jaw state,
attachment targets and nodal state. Cancellation is checked between adaptive
steps; interrupted intervals roll back completely.
Regular timestamps retain eight 240 Hz steps per 30 Hz interval despite
floating-point subtraction noise; they do not acquire an accidental ninth step.
Rigid eye targets use the same exact gaze transforms as the geometric goals.
Prescribed rigid parts interpolate rotation and position across substeps;
adaptive retries bisect their target motion too. Endpoints remain exact and
intermediate eye shapes remain rigid, including at a 180-degree rotation.

New packages store `fit_importance` for geometric inverse fitting. Every part
retains uniform coverage, with additional spatially distributed lip samples
and moving frontal skin samples. Hidden lining cannot crowd out those extra
samples. Target RMS remains an unweighted coordinate error in millimetres;
the report identifies its sampling policy and count. Older packages retain
their original part-uniform sampling policy.
New packages include 64 additional spatial eyelid samples and a separate
eyelid target gate. A stationary eyelid must stay within its regional limit
when another expression moves the face.

Speech closure targets attract rest-selected medial samples toward opposing
lip triangles, extend displacement through a bounded geodesic collar, and
protect the commissures. Continuous quadratic oriented-area limits and
crossing-count checks constrain that fit. A feasible point target can still
leave a gap; separate contact penalties measure the actual simulated lips.
Labiodental targets and checks select inner lower-lip sites nearest the
frontal upper dental surface, following the incisor arc while preserving
lateral width. These targets never deform the accepted output directly.
The articulation receipt records the geometry policy. Isolated bilabial
closures have no intrinsic opening; neighboring vowels retain anticipatory
opening. Sub-frame closures survive coarse sampling, and Japanese
moraic nasals and geminate closures follow the next consonant's place.
Japanese aligner `f` maps to bilabial `ɸ`, retaining the original phone in the
receipt. Bilabial fricatives request a partial constriction rather than a
lip–teeth bite. The distinction follows the [IPA consonant chart](https://www.internationalphoneticassociation.org/IPAcharts/IPA_charts_EI/IPA_charts_EI.html)
and the Japanese realization described in [the NINJAL-hosted phonetic study](https://repository.ninjal.ac.jp/record/3681/files/jasa_149_2426.pdf).

Implicit actuation enables guarded inertial, material, fiber, volume and
explicit attachment corrections. Each local correction uses one common scale
for all affected vertices and its XPBD multiplier, checking every incident
tetrahedron. The signed-volume determinant is cubic along that correction;
interior extrema are checked so a positive endpoint cannot hide an intervening
inversion. Inertial prediction uses a common volume-safe scale. The native
floor remains 0.06 for headroom at legacy float32 transport/export boundaries; final
double-precision volume acceptance remains 0.05. Surface contact projection
still has its separate residual and volume checks and bounded rollback.
Embedded contact corrections now share one continuous volume-safe fraction
with their paired rigid-body reactions. Requested proxy corrections and the
actual transported surface correction are reported separately. Contact that
cannot converge within 0.05 mm still rejects the interval.
The native `Session` also has opt-in `embedded` and `embedded_volume` contact
transports; both require double-precision body state. `embedded` combines
shared-node gradients before computing contact mass and updates only affected
surface samples. `embedded_volume` additionally tries a bounded local mass
projection with incident volume gradients when a normal correction is blocked.
Its limits are 64 volume constraints, 272 nodes and 12 active-set expansions.
The existing cubic path guard remains authoritative. Internal volume reactions
balance force and torque; the accepted contact multiplier pairs the dental
reaction. These transports remain experimental and are not the public bake
default. A captured open-vowel contact replay clears its residual with local
volume compensation, but the complete vowel still fails later contact gates.
This mode allows up to twelve contact passes and requires zero proper crossings
in both native and float32 surfaces before accepting each substep. An interval
that still fails rolls back and subdivides within the existing bounded budget.

Conservative advancement no longer treats iteration-budget exhaustion as a
separation proof. It retains the last separating halfspace for an unresolved
grazing pair. This conservative fallback can add separation near a tangential
miss; it prevents silent tunneling without claiming an exact time of impact.
A retained skin/gingiva crossing that exceeded 128 advancement iterations
clears with zero local/exported crossings under the fixed detector. This is a
replay regression, not validation of a full utterance.

`Session(jaw_mode='coupled')` is an additional experimental native path. It
projects mandibular tissue attachments and the three jaw coordinates together,
after the material, muscle and volume constraints in each solver iteration.
Those anchors are removed from the fixed-target attachment list, and a separate
native projector slot preserves the active muscle projector. Joint limits and
the same signed-volume fraction scale the paired tissue/jaw correction. Dental
vertices follow the resulting rigid jaw transform. This reduces the delayed
attachment-force feedback; surface contact forces still feed the next jaw
prediction. Snapshots synchronize private jaw state and independent workers
retain mode, inertia and damping exactly.

Explicit `jaw_inertia` and `jaw_damping` triples configure the force prediction;
their defaults retain the previous jaw trajectory bit for bit in a 1,000-step
randomized compatibility check. Alternative resistance settings are authored
experiments, not measured physiological calibration. Analytical effective-mass
and active-limit cases pass, as do exact rollback/worker replay and address,
undefined-behavior and leak sanitizer runs. Full open-vowel tracking remains
unqualified, so this mode is not a public bake default.
The `coupled_volume` variant additionally solves a clipped attachment with
nearby signed-volume inequalities and the reduced-coordinate bounds. This lets
a blocked tissue node react back onto the jaw, rather than scaling both sides
of an infeasible correction to almost zero. Embedded coupled contacts use the
same local mass solve. Rigid dental contacts update jaw coordinates, all rigid
particles and their embedded surface samples together; their impulses are not
fed back a second time. Contacts within a single rigid jaw are invariant and
cannot drive that jaw. Fixed-axis separation proofs avoid conservative grazing
corrections when the entire swept features are provably separated.
Near-rest Neo-Hookean energies below arithmetic cancellation noise are skipped
to avoid fictitious neutral strain. These guards add CPU work and do not
establish force calibration or speech acceptance.
Conservative reference-ball bounds accelerate small corrections; leaving a
ball invalidates neighboring fast proofs and falls back to the cubic check.
Each connected tetrahedral tissue volume has its own inertial limit and
mass-weighted translation. A blocked tongue component cannot limit an unrelated
skin component or an isolated particle. Free components retain their own
center-of-mass momentum; components with pinned vertices retain their barrier.
Native rest-space fiber geometry is stored once per referenced tetrahedron,
and Python packs the exact native layout through array views. An idle implicit
projector is detached while guarded passive tissue stays enabled. Conservative
guard proofs are refreshed between material, fiber and volume phases. A cache
can skip only an already verified arithmetic-noise material exit with identical
vertex bits and material parameters; changed state or stiffness recomputes it.
That optional cache is bounded to 32 MiB per body.

## Bakes and review

- `physics.usda`: complete evaluated time-sampled meshes, without skeleton
  bindings or an additional cranial transform.
- `physics_points_0000.npz`, etc.: 32-frame chunks with timestamps, geometry,
  input excitations, actual dynamic muscle activations, jaw coordinates and
  rigid head/dental transforms. `physics_state.npz` includes muscle names.
- `physics_state.npz` and `physics_report.json`: topology, dynamic tracks,
  source/solver hashes, CPU time, memory, fit errors and acceptance results.
- `physics_review.mp4` and `physics_preview.png`: CPU front/profile/cutaway
  geometry; the movie includes available take audio.
- `articulation.json/npz`: phoneme-conditioned geometric goals, including
  anticipation, closures, tongue contacts and English diphthong transitions.

The report checks positive volume, neutral equilibrium, proper surface
crossings, residual contact correction, tooth rigidity, target RMS,
actual bilabial/labiodental gaps and independent USD replay. A failed check produces an explicitly
labelled `rejected_geometry` review bake. Human review remains required after
the geometry checks pass. Fitting improvements are not independent 3D or
physiological accuracy measurements.
Crossing and triangle-area checks use the actual exported float32 surface,
including its cranial transform. Every retained source triangle must remain
nondegenerate and retain at least 5% of its area. This magnitude check permits
rigid rotation; it does not mistake a rotated normal for a folded triangle.

Bilabial aperture uses symmetric point-to-triangle distances on rest-selected
samples in the medial 60% of the lip span, with the existing 0.2 mm closure
threshold. This measures the opposing surfaces even when their vertices do
not line up. Reports retain `vertex_pair_aperture_mm` as a diagnostic proxy.
Open medial geometry still fails even if the corners touch; proper surface
crossings are checked separately.

English labiodental events also require the medial inner lower-lip surface to approach
the frontal upper dental surface within 0.5 mm. Missing dental geometry fails
the check. Alveolar, dental-tongue, velar and bilabial-fricative contacts are
explicitly listed as unvalidated. Passing the tested contacts does not validate
all articulators or vocal-tract acoustics.

Each part, lip features and eyelid features have a separate target check. Moving skin
features are checked independently of the rest of the head. Regional coordinate
RMS must be below `min(2, max(0.25, 0.75 * neutral_target_RMS))` mm, so a large
requested movement must improve on the stationary geometry and unrequested
regions must remain stable. Targets remain authored rig geometry, not measured
human ground truth. The original 2 mm whole-sample bound remains in place.

Bakes are immutable: use a new output directory or take to rerun. Failed jobs
retain partial outputs and diagnostics. A rejected physical step records up to
16 remaining contact vertices (including source/part IDs), its smallest-volume
tetrahedra, substep duration and contact guard fraction in `failure.json`.
These describe the failed trial before rollback; the accepted state stays
unchanged. Accepted inverse frames retain evidence from up to two rejected
trials too, so an unsafe semantic prior followed by a safe fallback is explicit.
Compatible `animation.json` and its
control-track USD are approximations to the full bake. Full caches are not
inputs to the older patch-deformer distillation command.

## Checks

```sh
TMPDIR="$PWD/tmp" OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  $PY -B -m unittest server.vhuman.test_biomechanics server.vhuman.test_gnm_anatomy \
  server.vhuman.test_gnm_cpu server.vhuman.test_native_fit server.vhuman.test_speech server.vhuman.test_speech_en \
  server.vhuman.test_geometry_fit server.vhuman.test_oral_fit
```

The mechanics tests use independent synthetic invariants: muscle impulse
balance, rigid dental motion, fast surface crossings, volume safety, neutral
equilibrium, rejected-step rollback, deterministic replay, and full CPU bake,
audio mux and independent USD round-trip. A separate seeded native audit
checks 20,000 random partial/all-node correction paths on disjoint and
connected tetrahedra against independent determinant-polynomial extrema. All
retain the 0.06 native floor and a common correction fraction, including 483
limited paths with positive full endpoints. This tests the correction guard,
not physiological accuracy. Another 1,000 random rotated/skewed tetrahedron
and shear cases match independent explicit Piola-force assembly to below
4.2e-15 relative velocity error, with balanced linear/angular impulses on
the legacy float32-sampled positions. That audit does not establish nonlinear
implicit-fiber convergence. Full-subject captures still need
direct geometry and human review; synthetic tests do not establish realism.

## Retained subject validation

On the retained full-subject fixture, bounded source conditioning removes all
1,409 proper edge/triangle crossing events while retaining surface topology:

| Source conditioning | Remaining crossing events |
| --- | ---: |
| Original source | 1,409 |
| Ocular cavities | 1,091 |
| Mouth and hidden lining | 198 |
| Rigid tooth crowns | 137 |
| Local tongue folds | 110 |
| Ventral tongue root | 0 |

Visible socket shifts peak at 1.04 mm; inferred hidden cups reach 21.65 mm.
Visible lip shifts peak at 0.57 mm; hidden lining reaches 26.29 mm. Rigid crown
translations peak at 0.108 mm, local tongue fairing at 0.548 mm, and hidden
root fitting at 7.725 mm. The upper 60% of the tongue is restored exactly.
An independent brute-force edge/triangle test confirms 110 to zero events for
the root repair. These substantial hidden corrections are authored priors.

The conditioned package has 105,084 nodes and 371,300 tetrahedra. Its neutral
bake passes every geometry check, with zero source and animated crossings,
minimum volume ratio 0.938, neutral RMS 0.000098 mm and exact USD replay.
The v4 tangent/harmonic-tongue package also passes a pure `/m/` frame using the
current component-volume and embedded-contact guard: medial aperture is
0.158 mm, with rigid dental geometry, valid triangles and passing regional
target checks. Adding explicit eyelid samples retains that pass. The inverse
stopping rule accepts this native result before allocating sensitivity workers;
its report records one trial and zero fitting iterations.

A separate inverse-fitted `/m/` on that v4 anatomy passes all checks with
0.155 mm aperture. Its combined contact/tracking objective improves from
0.01509 to 0.00305; point-only RMS increases from 0.0307 to 0.0417 mm. This
measurement uses the archived pre-component native solver and is not an exact
comparison with the newer solver. Neither single-phoneme test establishes
complete-utterance, rounded-vowel or acoustic quality.

A fresh public CLI preparation with all geometry conditioning, sliding skin,
tangent facial fibers and harmonic tongue fibers produces 116,077 nodes and
405,358 tetrahedra, with zero source crossings. On that package the current
CPU solver and default double transport pass a separate zero-iteration `/m/`
bake: 0.172 mm medial aperture, minimum volume ratio 0.05999, neutral RMS
0.000118 mm, target RMS 0.0407 mm, valid triangles, zero animated crossings
and exact USD replay. This is a different volume mesh from the older v4
packages; its pass does not transfer to their double trajectories.
The repeat cached package adds the corrected calibration seed and retains
bit-identical simulated `/m/` geometry. The public CLI with two requested
fitting iterations and four requested workers passes in one full-duration
trial, stops at validated tolerances and allocates no sensitivity workers.
Its report records `auto` resolving to `float64`; no optimizer surrogate
replaces the native state. A subsequent three-frame `/m/` hold also
passes every geometry check, with 0.135 mm closure aperture, zero crossings
and exact USD replay. Its activation state evolves over time; each interval
uses one validated native trial with no sensitivity workers. The last two
intervals use the previous fitted drive as a warm start, rather than replaying
a static surface. Measured apertures are 0.172, 0.148 and 0.135 mm;
upper-left oris activation rises from 0.589 to 0.744 to 0.785 under constant
0.8 excitation. Geometry changes in both consecutive intervals. This still does not establish complete-utterance fidelity.

With the fitted dental translation track, the simultaneous jaw attachment
option (`--mechanics-solver block`) passes a separate public three-frame `/m/`
hold: final closure aperture is 0.113 mm, minimum volume ratio 0.05999,
zero surface crossings and exact USD replay. A same-configuration single-frame
comparison reduces aperture from 0.175 mm with sequential coupled attachments
to 0.136 mm with simultaneous attachments. These results qualify only the
retained bilabial fixture. The wider `/a/` request still exceeds the contact
residual limit, and the inverse-fitted brow fixture still fails regional
tracking. The longer brow experiment was interrupted at wrap-up and supplies
no final acceptance result.

The conditioned captured-motion and gentle-expression diagnostics pass the
physical, crossing and replay checks but correctly fail regional target
tracking. A larger gaze ramp still fails the contact residual limit near the
eyelids, including with released anterior orbital supports. The v5 ocular-ring
atlas and orbital support revision passes a separate inverse-fitted `/m/`
with 0.152 mm aperture on the float32 bridge. Its larger brow ramp is rejected
for regional tracking and 7–11 crossings in later hold frames; blink falls back
to a safe drive and fails tracking. This revision remains experimental.
A zero-iteration open-vowel diagnostic on the fresh cached package cannot
resolve the authored drive's skin contact at the smallest allowed step. It
falls back to a safe near-neutral drive and is rejected for target tracking
(overall RMS 7.43 mm). This is not accepted jaw-opening or vowel animation.
A separate zero-iteration `/f/` diagnostic on this mesh falls back to its
preceding safe drive and fails the labiodental and regional gates (7.05 mm
lip-to-teeth gap). The tongue-tip `/t/` diagnostic also falls back and fails
whole/region target tracking (2.10 mm overall RMS); alveolar contact remains
explicitly unvalidated. Geometry gates have not been relaxed.

## CPU measurements

Measurements here use an AMD Ryzen Threadripper 1950X (16 physical cores,
32 logical CPUs), GCC 15.2, Python 3.12.12, PyTorch 2.9.1+cpu, NumPy 2.2.6
and SciPy 1.16.3. Native builds use Release optimization. Whole-bake wall
times include preparation/export overhead and concurrent experiment load;
they are not isolated kernel throughput measurements.

Saving the accepted native trial instead of rerunning it reproduces a retained
three-frame bake bit for bit with the same solver and inputs. Wall time falls
from 1,132 s to 485 s (2.34x). Four independent full-mesh 1/240 s probes retain
bit-identical residuals and unchanged owner positions; measured trial time
falls from 23.7 s serial to 9.5 s concurrent (2.50x), with about 6,020 MiB peak
RSS for that older four-worker configuration. These are offline measurements.

Array-based packing of 1,608,286 native fibers retains identical 64,331,440
bytes while reducing setup from 11.38 s to 0.148 s and peak packing RSS from
1,015 to 175 MiB. Sharing per-tetrahedron active rest geometry reduces a
separate setup benchmark from 532 to 372 MiB before the optional material
cache, whose additional allocation is capped at 32 MiB per body. Whole-bake
performance depends strongly on anatomy, native trials and contact difficulty;
none of these measurements establishes real-time animation.
Releasing the copied ABI staging buffer reduces a retained session from about
371 to 310 MiB resident memory (1,608,286 fibers, about 61 MiB saved per body).
A same-target replay seeded with a previously fitted `/m/` drive reproduces
the full-subject float32 surface and controls bit for bit in one validated
trial, with zero worker allocation, versus 69 trials in its original fit.
This verifies seeded replay, not new-utterance quality or a general speedup.

On an earlier controlled 33 ms brow transient, normal/tangential attachments
increase mean lift from 0.024/0.027 mm to 0.340/0.345 mm toward roughly 1 mm
targets. That is an incomplete expression response, not measured physiology.
