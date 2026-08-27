# GLM5.3 Flash preflight

`~/models/glm53f` is a Hugging Face safetensors checkout, not the older GGUF
layout used by the existing GLM5.2 experiments. Do not copy the model to a
single node: it has 62 multi-gigabyte shards and is intended for the later
12-node run.

Run the metadata-only preflight on Fugaku instead:

```bash
python3 a64fx/glm5/glm53f_preflight.py ~/models/glm53f
```

It reads only `config.json`, the safetensors index, tokenizer metadata, and
`chat_template.jinja`. The current checkout reports 62/62 shards, 45 layers,
hidden size 4096, 288 routed experts with top-8 routing, mixed linear/full
attention layers, and a 1M-token position limit.

The text-only chat path is available in both `glm5_tokenizer.py` and the C11
headers `common/glm5_bpe.h` and `common/glm5_chat_template.h`. The C headers
load `tokenizer.json`, perform byte-level BPE, and mirror the template's basic
system/user/assistant/observation flow and generation prompt; tool-call JSON
serialization remains a later agent-runtime task. Compile its resource-light
unit test on an A64FX node with:

```bash
fcc -Nclang -O2 -I../../common test_glm53f_chat.c -o test_glm53f_chat
./test_glm53f_chat
```

The BPE test can be run with the real tokenizer without touching any model
shard:

```bash
cc -std=c11 -O2 -I../../common test_glm53f_bpe.c -o test_glm53f_bpe
./test_glm53f_bpe ~/models/glm53f/tokenizer.json
```

The static architecture contract has a small no-model unit test too:

```bash
cc -std=c11 -Wall -Wextra -Wpedantic -I../../common test_glm53f_arch.c -o test_glm53f_arch
./test_glm53f_arch
```

After the GLM5.3F graph and safetensors mapper land, the EP runner accepts the
same path directly with `--prompt-text TEXT`; it renders the C template and
tokenizes on every rank, so a separate Python tokenizer process is not needed.

The architecture contract is recorded in `common/glm53f_arch.h`. The existing
`common/glm5.h` constants describe the previous GLM5.2 harness and must not be
used to launch GLM5.3 Flash until its new graph, weight mapping, and 12-node
partitioning are implemented.

For a stronger, still metadata-only check, inspect the 62 safetensors headers
as well. This verifies the `model.language_model.layers.*` namespace, all 45
layer IDs, the mixed linear/sparse layer pattern, and the expected 288-expert
router inventory without reading tensor payloads:

```bash
python3 a64fx/glm5/glm53f_preflight.py --check-tensors ~/models/glm53f
```
