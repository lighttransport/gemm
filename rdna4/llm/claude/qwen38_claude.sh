#!/usr/bin/env bash
# Run the Claude Code CLI against the local Qwen3.8-27B server started by
# rdna4/llm/run_qwen38_27b_codex.sh (Anthropic Messages API on /v1/messages).
#
#   rdna4/llm/claude/qwen38_claude.sh -p "fix the bug in mathutil.c" \
#       --allowedTools 'Read,Edit,Bash(cc:*)'
#
# The CLI runs with a clean environment and its own config directory
# (QWEN38_CLAUDE_CONFIG_DIR, default ~/.claude-qwen38), so it never mixes
# with an existing Claude Code login, settings or session transcripts.
set -euo pipefail
port="${QWEN38_API_PORT:-8090}"
host="${QWEN38_API_HOST:-127.0.0.1}"
config="${QWEN38_CLAUDE_CONFIG_DIR:-${HOME}/.claude-qwen38}"
context="${QWEN38_CONTEXT:-65536}"
max_output="${QWEN38_MAX_OUTPUT:-16384}"
model="${QWEN38_CLAUDE_MODEL:-qwen3.8-27b}"
mkdir -p "${config}"
exec env -i HOME="${HOME}" PATH="${PATH}" TERM="${TERM:-xterm-256color}" \
    LANG="${LANG:-C.UTF-8}" \
    CLAUDE_CONFIG_DIR="${config}" \
    ANTHROPIC_BASE_URL="http://${host}:${port}" \
    ANTHROPIC_AUTH_TOKEN="${QWEN38_CLAUDE_TOKEN:-local-qwen38}" \
    ANTHROPIC_MODEL="${model}" ANTHROPIC_DEFAULT_OPUS_MODEL="${model}" \
    ANTHROPIC_DEFAULT_SONNET_MODEL="${model}" ANTHROPIC_DEFAULT_HAIKU_MODEL="${model}" \
    CLAUDE_CODE_SUBAGENT_MODEL="${model}" \
    CLAUDE_CODE_MAX_CONTEXT_TOKENS="${context}" \
    CLAUDE_CODE_AUTO_COMPACT_WINDOW="${context}" \
    CLAUDE_CODE_MAX_OUTPUT_TOKENS="${max_output}" \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_TELEMETRY=1 \
    DISABLE_AUTOUPDATER=1 \
    claude "$@"
