# Third-party notices

The PBR Neutral tone-mapping adaptations in `eye/optics.py` and
`web/vhuman_eye_shader.js` follow Three.js `NeutralToneMapping`:
https://github.com/mrdoob/three.js/blob/r170/src/renderers/shaders/ShaderChunk/tonemapping_pars_fragment.glsl.js

Three.js license: https://github.com/mrdoob/three.js/blob/r170/LICENSE

The MIT License

Copyright © 2010-2024 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Other external dependencies are installed separately and retain their package
licenses. No third-party model weights or generated images are shipped here.

The optional GNM v3 model weights are fetched from
https://huggingface.co/google/gnm-v3 (Apache-2.0). The pinned
repository revision is `c01e90d298d82301f9fd18f54806be751775cb7c`; the
`v3_0/gnm_head.npz` SHA-256 is
`61d78bbfb4ad8e0b38495804a4caef3214d3df00f8c3f68761e63b41ce3747eb`.
The model card at https://huggingface.co/google/gnm-v3 declares Apache-2.0.

ICT-FaceKit Light meshes and shapes are fetched from
https://github.com/USC-ICT/ICT-FaceKit (MIT), pinned to commit
`da5f95a607f5e6b37755b38d3385d7f2853732e5`. Its license is at
https://github.com/USC-ICT/ICT-FaceKit/blob/master/LICENSE.
The download cache and fitted user assets are excluded from this repository.
# MediaPipe face model metadata

`native_landmarks.py` uses the 146-landmark subset and 52 blendshape names from
Google MediaPipe `v0.10.21`, `face_blendshapes_graph.cc` (Copyright 2023 The
MediaPipe Authors), licensed under Apache License 2.0:
https://www.apache.org/licenses/LICENSE-2.0

Source: https://github.com/google-ai-edge/mediapipe/blob/v0.10.21/mediapipe/tasks/cc/vision/face_landmarker/face_blendshapes_graph.cc

The C++ executor is an independent implementation. Model weights are downloaded
separately by `rig/setup_face_video.sh`, verified by SHA256, and remain outside
the repository. Exported graph receipts retain their source task/tensor hashes.
