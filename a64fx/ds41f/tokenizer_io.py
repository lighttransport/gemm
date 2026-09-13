#!/usr/bin/env python3
"""Frontend text/token adapter; requires Hugging Face tokenizers, not PyTorch.

Plain text gets BOS. Chat messages use the checkpoint's explicit encoding.py.
"""
import argparse
import importlib.util
import json
import re
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    encode = sub.add_parser("encode")
    source = encode.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    source.add_argument("--messages-json", type=Path)
    encode.add_argument("--encoding", type=Path,
                        help="Trusted checkpoint encoding/encoding.py, required for messages")
    encode.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    encode.add_argument("--reasoning-effort", type=int, default=75)
    encode.add_argument("--output", required=True, type=Path)
    decode = sub.add_parser("decode")
    decode.add_argument("--rank0-log", required=True, type=Path)
    decode.add_argument("--prompt-ids", required=True, type=Path)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    if args.command == "encode":
        if args.messages_json:
            if not args.encoding or not 1 <= args.reasoning_effort <= 100:
                parser.error("messages require --encoding and reasoning effort in [1, 100]")
            messages = json.loads(args.messages_json.read_text())
            if not isinstance(messages, list) or not messages:
                parser.error("messages must be a nonempty JSON array")
            if any(not isinstance(message, dict) or
                   not isinstance(message.get("content"), str) for message in messages):
                parser.error("this runner accepts text-only message content")
            spec = importlib.util.spec_from_file_location("ds41f_checkpoint_encoding", args.encoding)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            text, media = module.encode_messages(messages, thinking_mode=args.thinking_mode,
                                                 reasoning_effort=args.reasoning_effort,
                                                 return_multi_modal_data=True)
            if media.get("images"):
                parser.error("vision inference is not implemented by this runner")
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if not ids or ids[0] != 0:
                parser.error("checkpoint encoder/tokenizer BOS contract mismatch")
        else:
            if args.encoding:
                parser.error("--encoding is only used with --messages-json")
            text = args.text if args.text is not None else args.text_file.read_text()
            ids = [0] + tokenizer.encode(text, add_special_tokens=False).ids
        if any(token < 0 or token >= 129280 for token in ids):
            parser.error("tokenizer does not match the DS4.1-Flash vocabulary")
        with args.output.open("x") as file:
            file.write(" ".join(map(str, ids)) + "\n")
        print(f"Encoded {len(ids)} tokens including BOS")
    else:
        prompt = [int(token) for token in args.prompt_ids.read_text().split()]
        if not prompt:
            parser.error("empty prompt")
        ids = []
        previous = -1
        for match in re.finditer(r"^TOKEN pos=(\d+) input=(\d+) next=(\d+) ",
                                 args.rank0_log.read_text(), re.MULTILINE):
            pos, _, token = map(int, match.groups())
            if pos != previous + 1:
                parser.error("non-contiguous or concatenated rank-zero log")
            previous = pos
            if pos + 1 >= len(prompt):
                ids.append(token)
        if not ids:
            parser.error("no generated tokens yet")
        print(tokenizer.decode(ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
