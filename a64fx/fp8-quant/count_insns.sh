#!/bin/bash
echo "=== Instruction Count Analysis ==="
echo ""

for func in fp16_to_fp8_e5m2_sve fp16_to_fp8_e4m3_sve fp32_to_fp8_e5m2_sve fp32_to_fp8_e4m3_sve; do
    echo "=== $func ==="
    # Count all instructions in the function
    total=$(sed -n "/^${func}:/,/^\\.size.*${func}/p" fp8_quant.s | grep -cE "^\s+[a-z]")
    
    # Count SVE instructions specifically
    sve=$(sed -n "/^${func}:/,/^\\.size.*${func}/p" fp8_quant.s | grep -cE "(ld1|st1|ptrue|whilel|and z|orr z|lsr z|lsl z|add z|sub z|cmp.*p|sel z|dup z|mov.*z|uzp|trn|movprfx|cpy z)")
    
    echo "  Total instructions: ~$total"
    echo "  SVE instructions: ~$sve"
    echo ""
done
