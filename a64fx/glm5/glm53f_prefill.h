#ifndef GLM53F_PREFILL_H
#define GLM53F_PREFILL_H
/* The outer layer tile is independent of verification snapshots, arithmetic
 * panels, and collective payloads. Attention scratch stays bounded at 32. */
enum { GLM53F_PREFILL_MAX_TOKENS = 256, GLM53F_PREFILL_ATTN_TOKENS = 32 };
#endif
