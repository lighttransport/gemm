# Generated RDNA4 mm0 BF16 direct-A schedule

This diagnostic keeps operand A in VGPRs loaded directly from global memory and
double-buffers only operand B through LDS.  It intentionally spends more global
bandwidth on A inside the CTA to reduce LDS traffic and wait pressure, matching
the major pattern observed in the hipBLASLt mm0 dump.
