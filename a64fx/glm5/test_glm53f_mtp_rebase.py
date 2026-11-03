#!/usr/bin/env python3
"""Pure state-machine regression for speculative MTP cache rebasing.

The integrated runner consumes ``ndraft`` inputs before verification.  A full
acceptance therefore needs one final input appended; a rejection must restore
the pre-cycle length and replay only the committed prefix.
"""

def commit_prefix(ndraft: int, accepted: int):
    assert 1 <= ndraft <= 4
    assert 0 <= accepted <= ndraft
    base = ["context"]
    inputs = ["first"] + [f"draft{i}" for i in range(ndraft)]
    speculative = base + inputs[:ndraft]
    committed = accepted + 1 if accepted < ndraft else ndraft + 1
    if accepted == ndraft:
        result = speculative + [inputs[ndraft]]
    else:
        result = base + inputs[:committed]
    assert result == base + inputs[:committed]
    return result


for drafts in range(1, 5):
    for accepted in range(drafts + 1):
        state = commit_prefix(drafts, accepted)
        print(f"drafts={drafts} accepted={accepted} length={len(state)-1} PASS")
