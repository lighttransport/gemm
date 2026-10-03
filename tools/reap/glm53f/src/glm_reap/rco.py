from __future__ import annotations

import numpy as np
from pathlib import Path
from .common import fingerprint


def allocate(costs, scores, budget):
    """Lagrangian discrete allocation followed by strict feasible repair."""
    groups = sorted(costs)
    if sum(min(costs[g]) for g in groups) > budget:
        raise ValueError("Even minimum precision exceeds the weight budget")
    def choose(multiplier):
        return {g: int(np.argmax(np.asarray(scores[g])-multiplier*np.asarray(costs[g]))) for g in groups}
    low, high = 0.0, 1.0
    for _ in range(80):
        mid = (low+high)/2
        selected = choose(mid)
        size = sum(costs[g][selected[g]] for g in groups)
        if size > budget:
            low = mid
        else:
            high = mid
    selected = choose(high)
    used = sum(costs[g][selected[g]] for g in groups)
    # Add affordable upgrades, ordered by task score gain per byte.
    upgrades = []
    for g in groups:
        i = selected[g]
        for j in range(len(costs[g])):
            delta = costs[g][j]-costs[g][i]
            gain = scores[g][j]-scores[g][i]
            if delta > 0 and gain > 0:
                upgrades.append((gain/delta, g, j))
    for _, g, j in sorted(upgrades, reverse=True):
        delta = costs[g][j]-costs[g][selected[g]]
        if delta > 0 and used+delta <= budget and scores[g][j] > scores[g][selected[g]]:
            selected[g], used = j, used+delta
    return selected, int(used)


def optimize(bank, runtime, batches, config, overhead=0):
    """Optimize choices using complete model answer-token cross entropy."""
    import torch
    costs = bank.costs()
    groups = sorted(g for g in costs if len(costs[g]) > 1)
    logits = torch.nn.ParameterDict({str(i): torch.nn.Parameter(torch.zeros(len(costs[g]), device=runtime.device)) for i, g in enumerate(groups)})
    optimizer = torch.optim.Adam(logits.parameters(), lr=config["rco_lr"])
    dual = 0.0
    budget = config["weight_budget_bytes"]-overhead
    fixed = sum(c[0] for c in costs.values() if len(c) == 1)
    if fixed+sum(min(costs[g]) for g in groups) > budget:
        raise ValueError("Native format minimum size exceeds budget")
    records = list(batches)
    if not records:
        raise ValueError("RCO requires answer-masked task examples")
    history = []
    torch.manual_seed(config["seed"])
    checkpoint_path = Path(config.get("work_dir", config["output"]))/"rco-state.pt"
    identity = fingerprint({"config": config, "costs": {g: list(map(int, c)) for g, c in costs.items()}, "reap": (Path(config.get("work_dir", config["output"]))/"reap.json").read_text()})
    first_step = 0
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=runtime.device, weights_only=True)
        if state["identity"] != identity:
            raise ValueError("RCO checkpoint source/config/candidate layout changed")
        logits.load_state_dict(state["logits"])
        optimizer.load_state_dict(state["optimizer"])
        first_step, dual, history = state["next_step"], state["dual"], state["history"]
        torch.set_rng_state(state["rng"].cpu())
        if torch.cuda.is_available() and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    for step in range(first_step, config["rco_steps"]):
        optimizer.zero_grad(set_to_none=True)
        ratio = step/max(1, config["rco_steps"]-1)
        temperature = config["rco_temperature"][0]*(config["rco_temperature"][1]/config["rco_temperature"][0])**ratio
        probabilities = {g: torch.nn.functional.gumbel_softmax(logits[str(i)], tau=temperature, hard=True) for i, g in enumerate(groups)}
        runtime.probabilities = probabilities
        expected = sum((p*torch.as_tensor(costs[g], device=runtime.device)).sum() for g, p in probabilities.items())+fixed
        record = records[step % len(records)]
        loss = runtime.loss(record[0], record[1], checkpoint=True, vision=record[2] if len(record) > 2 else None)
        violation = (expected-budget)/budget
        objective = loss+dual*violation+10*torch.relu(violation).square()
        objective.backward()
        optimizer.step()
        dual = max(0.0, dual+0.1*float(violation.detach().cpu()))
        history.append({"step": step, "task_loss": float(loss.detach().cpu()), "expected_bytes": int(expected.detach().cpu())})
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({"identity": identity, "next_step": step+1, "dual": dual, "history": history, "logits": logits.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, temporary)
        import os
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(checkpoint_path)
    scores = {g: logits[str(i)].detach().cpu().numpy() for i, g in enumerate(groups)}
    scores.update({g: np.zeros(1) for g, c in costs.items() if len(c) == 1})
    bank.selection, size = allocate(costs, scores, budget)
    runtime.probabilities = {}
    return {"selection": bank.selection, "payload_bytes": size, "overhead_reserve": overhead, "history": history}
