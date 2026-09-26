"""Incremental Responses / Messages streams for the Qwen3.8 shim.

Reasoning and answer text reach the client while the model generates.  The
opaque replay blob must be emitted before its reasoning item / thinking block
closes, which is before the raw turn is known, so a stream carries a
reference (``q38ref1:<id>``) that the server resolves through its raw-turn
memory; the raw bytes are stored under the id when generation finishes.
Non-streaming responses keep the inline ``q38raw1:`` blob.
"""
import json
import uuid

from qwen_tools import call_events

REF_PREFIX = "q38ref1:"


def new_ref():
    return REF_PREFIX + uuid.uuid4().hex


def ref_id(blob):
    return blob if isinstance(blob, str) and blob.startswith(REF_PREFIX) else None


class MessagesLive:
    """Anthropic Messages stream: thinking block, text block, tool uses."""

    def __init__(self, emit, ref):
        self.emit, self.ref = emit, ref
        self.index = 0
        self.block = "thinking"
        self.content = ""
        emit("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "thinking", "thinking": "",
                                                       "signature": ""}})

    def _close(self):
        if self.block == "thinking":
            self.emit("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                              "delta": {"type": "signature_delta",
                                                        "signature": self.ref}})
        self.emit("content_block_stop", {"type": "content_block_stop", "index": self.index})
        self.index += 1
        self.block = None

    def _text(self, piece):
        if self.block == "thinking":
            self._close()
        if self.block is None:
            self.emit("content_block_start", {"type": "content_block_start", "index": self.index,
                                              "content_block": {"type": "text", "text": ""}})
            self.block = "text"
        self.emit("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                          "delta": {"type": "text_delta", "text": piece}})
        self.content += piece

    def feed(self, parts):
        for kind, piece in parts:
            if kind == "reasoning" and self.block == "thinking":
                self.emit("content_block_delta", {"type": "content_block_delta",
                                                  "index": self.index,
                                                  "delta": {"type": "thinking_delta",
                                                            "thinking": piece}})
            elif kind == "content":
                self._text(piece)

    def finish(self, text, tool_blocks, reason, use):
        if self.block == "thinking":
            self._close()
        if text.startswith(self.content) and len(text) > len(self.content):
            self._text(text[len(self.content):])
        if self.block == "text":
            self._close()
        for block in tool_blocks:
            self.emit("content_block_start", {"type": "content_block_start", "index": self.index,
                                              "content_block": {**block, "input": {}}})
            self.emit("content_block_delta", {"type": "content_block_delta", "index": self.index,
                                              "delta": {"type": "input_json_delta",
                                                        "partial_json": json.dumps(
                                                            block["input"], ensure_ascii=False)}})
            self.emit("content_block_stop", {"type": "content_block_stop", "index": self.index})
            self.index += 1
        self.emit("message_delta", {"type": "message_delta",
                                    "delta": {"stop_reason": reason, "stop_sequence": None},
                                    "usage": use})
        self.emit("message_stop", {"type": "message_stop"})


class ResponsesLive:
    """Responses stream: reasoning item, message item, then tool calls."""

    def __init__(self, emit, response_id, ref):
        self.emit, self.response_id, self.ref = emit, response_id, ref
        self.reasoning_id = "rs_" + uuid.uuid4().hex
        self.message_id = response_id + "-item"
        self.summary = ""
        self.summary_open = False
        self.reasoning_open = True
        self.message_open = False
        self.content = ""
        self.output = []
        emit({"type": "response.output_item.added", "output_index": 0,
              "item": {"type": "reasoning", "id": self.reasoning_id, "summary": []}})

    def _reasoning_item(self):
        return {"type": "reasoning", "id": self.reasoning_id,
                "summary": ([{"type": "summary_text", "text": self.summary}]
                            if self.summary else []),
                "encrypted_content": self.ref}

    def _close_reasoning(self):
        common = {"item_id": self.reasoning_id, "output_index": 0, "summary_index": 0}
        if self.summary_open:
            self.emit({"type": "response.reasoning_summary_text.done", **common,
                       "text": self.summary})
            self.emit({"type": "response.reasoning_summary_part.done", **common,
                       "part": {"type": "summary_text", "text": self.summary}})
        item = self._reasoning_item()
        self.emit({"type": "response.output_item.done", "output_index": 0, "item": item})
        self.output.append(item)
        self.reasoning_open = False

    def _text(self, piece):
        if self.reasoning_open:
            self._close_reasoning()
        common = {"item_id": self.message_id, "output_index": 1, "content_index": 0}
        if not self.message_open:
            self.emit({"type": "response.output_item.added", "output_index": 1,
                       "item": {"type": "message", "id": self.message_id, "role": "assistant",
                                "status": "in_progress", "content": []}})
            self.emit({"type": "response.content_part.added", **common,
                       "part": {"type": "output_text", "text": "", "annotations": []}})
            self.message_open = True
        self.emit({"type": "response.output_text.delta", **common, "delta": piece})
        self.content += piece

    def feed(self, parts):
        for kind, piece in parts:
            if kind == "reasoning" and self.reasoning_open:
                common = {"item_id": self.reasoning_id, "output_index": 0, "summary_index": 0}
                if not self.summary_open:
                    self.emit({"type": "response.reasoning_summary_part.added", **common,
                               "part": {"type": "summary_text", "text": ""}})
                    self.summary_open = True
                self.emit({"type": "response.reasoning_summary_text.delta", **common,
                           "delta": piece})
                self.summary += piece
            elif kind == "content":
                self._text(piece)

    def finish(self, text, calls):
        if self.reasoning_open:
            self._close_reasoning()
        if text.startswith(self.content) and len(text) > len(self.content):
            self._text(text[len(self.content):])
        if not self.message_open and not calls:
            self._text("")
        if self.message_open:
            part = {"type": "output_text", "text": self.content, "annotations": []}
            common = {"item_id": self.message_id, "output_index": 1, "content_index": 0}
            self.emit({"type": "response.output_text.done", **common, "text": self.content})
            self.emit({"type": "response.content_part.done", **common, "part": part})
            item = {"type": "message", "id": self.message_id, "role": "assistant",
                    "status": "completed", "content": [part]}
            self.emit({"type": "response.output_item.done", "output_index": 1, "item": item})
            self.output.append(item)
        base = len(self.output)
        for index, call in enumerate(calls):
            for event in call_events(self.response_id, [call]):
                if event["type"] in ("response.created", "response.in_progress"):
                    continue
                self.emit({**event, "output_index": base + index})
            self.output.append(call)
        return self.output
