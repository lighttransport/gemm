#!/usr/bin/env python3
"""Extract unicode data from llama.cpp's unicode-data.cpp into C arrays."""

import re

def main():
    with open("/home/syoyo/work/llama.cpp/src/unicode-data.cpp", "r") as f:
        text = f.read()

    print("/* Auto-generated from llama.cpp/src/unicode-data.cpp */")
    print()

    # 1. unicode_ranges_flags
    m = re.search(r'unicode_ranges_flags\s*=\s*\{(.*?)\};', text, re.DOTALL)
    pairs = re.findall(r'\{(0x[0-9A-Fa-f]+),\s*(0x[0-9A-Fa-f]+)\}', m.group(1))
    print("static const struct { uint32_t start; uint16_t flags; } bpe_unicode_ranges_flags[] = {")
    for start, flags in pairs:
        print(f"{{{start}, {flags}}},")
    print("};")
    print("static const int bpe_unicode_ranges_flags_len = sizeof(bpe_unicode_ranges_flags)/sizeof(bpe_unicode_ranges_flags[0]);")
    print()

    # 2. unicode_set_whitespace
    m = re.search(r'unicode_set_whitespace\s*=\s*\{(.*?)\};', text, re.DOTALL)
    vals = re.findall(r'(0x[0-9A-Fa-f]+)', m.group(1))
    print("static const uint32_t bpe_unicode_whitespace[] = {")
    print(", ".join(vals))
    print("};")
    print("static const int bpe_unicode_whitespace_len = sizeof(bpe_unicode_whitespace)/sizeof(bpe_unicode_whitespace[0]);")
    print()

    # 3. unicode_map_lowercase
    m = re.search(r'unicode_map_lowercase\s*=\s*\{(.*?)\};', text, re.DOTALL)
    pairs = re.findall(r'\{(0x[0-9A-Fa-f]+),\s*(0x[0-9A-Fa-f]+)\}', m.group(1))
    print("static const struct { uint32_t from; uint32_t to; } bpe_unicode_lowercase[] = {")
    for f, t in pairs:
        print(f"{{{f}, {t}}},")
    print("};")
    print("static const int bpe_unicode_lowercase_len = sizeof(bpe_unicode_lowercase)/sizeof(bpe_unicode_lowercase[0]);")

if __name__ == "__main__":
    main()
