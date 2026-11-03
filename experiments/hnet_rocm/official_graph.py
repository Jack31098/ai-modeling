"""Check portable composition against the actual pinned upstream HNet.forward.

Executes only hash-checked class/function ASTs, excluding fused imports. Upstream
router/chunk/HNet control graph run unchanged; Isotropic and DeChunk use the
explicit portable operators. This checks composition, not native kernel parity.
"""

from .bootstrap import RUNTIME

import argparse
import ast
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import types
from typing import Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .reference import ReferenceHNet, dechunk, load_config
from .validate import MODEL, SAMPLE, metadata, route_metadata, save, sha


def upstream_scope():
    lock = json.loads(Path("experiments/hnet_rocm/source-lock.json").read_text(encoding="utf-8"))
    module_name = "experiments.hnet_rocm._checked_upstream"
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module
    scope = module.__dict__
    scope.update(torch=torch, nn=nn, F=F, dataclass=dataclass, Optional=Optional, Union=Union,
                 HNetConfig=object, IsotropicInferenceParams=object)
    selections = {"hnet__modules__dc.py": {"RoutingModuleOutput", "RoutingModuleState", "DeChunkState", "RoutingModule", "ChunkLayer"},
                  "hnet__models__hnet.py": {"STE", "ste_func", "HNetState", "HNet"}}
    for filename, names in selections.items():
        entry = next(e for e in lock["files"] if Path(e["local_path"]).name == filename)
        source = Path(entry["local_path"])
        if sha(source) != entry["sha256"]:
            raise ValueError("official source drift")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if {node.name for node in nodes} != names:
            raise ValueError("missing official AST definitions")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


class Linear:
    def __init__(self, ref, prefix):
        self.weight, self.bias = ref.w[prefix + ".weight"], ref.w[prefix + ".bias"]

    def __call__(self, x):
        return F.linear(x, self.weight, self.bias)


class Proxy:
    def __init__(self, ref, scope, prefix, layout, stage):
        self.ref, self.scope, self.prefix, self.stage_idx = ref, scope, prefix, stage
        self.d_model = ref.config["d_model"][stage]
        self.pad_dimension = ref.w.get(prefix + ".pad_dimension")
        self.is_innermost = len(layout) == 1
        def iso(name, arches):
            def run(x, **kwargs):
                assert x.shape[0] == 1
                return ref.isotropic(x[0], prefix + "." + name, arches, stage)[None]
            return run
        if self.is_innermost:
            self.main_network = iso("main_network", layout[0])
        else:
            self.encoder = iso("encoder", layout[0])
            self.decoder = iso("decoder", layout[2])
            self.main_network = Proxy(ref, scope, prefix + ".main_network", layout[1], stage + 1)
            self.residual_proj = Linear(ref, prefix + ".residual_proj")
            router = scope["RoutingModule"](self.d_model, device=next(ref.parameters()).device)
            router.load_state_dict({k.removeprefix(prefix + ".routing_module."): v
                                    for k, v in ref.w.items() if k.startswith(prefix + ".routing_module.")})
            self.routing_module = router
            self.chunk_layer = scope["ChunkLayer"]()
            self.residual_func = lambda out, residual, p: out * scope["STE"].apply(p) + residual

    def dechunk_layer(self, hidden_states, boundary_mask, boundary_prob, cu_seqlens, mask=None, inference_params=None):
        assert hidden_states.shape[0] == 1 and cu_seqlens is None and inference_params is None
        indices = torch.where(boundary_mask[0])[0]
        return dechunk(hidden_states[0], indices, boundary_prob[0, :, 1], boundary_mask.shape[1], self.ref.dechunk_mode)[None]

    def __call__(self, x, **kwargs):
        return self.scope["HNet"].forward(self, x, **kwargs)


def check(device):
    ref = ReferenceHNet(MODEL / "hnet_2stage_L.pt", load_config(), device=device, dechunk_mode="ema-fp32-step")
    scope = upstream_scope()
    proxy = Proxy(ref, scope, "backbone", ref.config["arch_layout"], 0)
    ids = torch.tensor(list(SAMPLE.encode("utf-8")[:-1]), device=device)
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        expected, routes = ref(ids)
        embedded = F.embedding(ids, ref.w["embeddings.weight"])[None]
        hidden, original_routes = proxy(embedded, mask=torch.ones(1, len(ids), dtype=torch.bool, device=device))
        actual = F.linear(hidden[0], ref.w["lm_head.weight"])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
        for a, b in zip(routes, original_routes):
            assert torch.equal(a["selected"], torch.where(b.boundary_mask[0])[0])
        difference = (actual - expected).abs().max().item()
    return {**metadata(), "status": "passed", "device": device, "logits_max_abs_difference": difference,
            "routes": route_metadata(routes), "source_lock_sha256": sha("experiments/hnet_rocm/source-lock.json"),
            "scope": "Original pinned HNet.forward/RoutingModule/ChunkLayer/STE executed unchanged via checked AST; portable Isotropic and dechunk substitutions. Not fused kernel validation or cached step testing."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = check(args.device)
    save(args.out, result)
    print(json.dumps({"status": result["status"], "device": args.device, "logits_max_abs_difference": result["logits_max_abs_difference"]}))
