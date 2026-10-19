"""Save/load training checkpoints: weights, optimizer state(s), value normalizer and extra metadata.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import torch

_STATE_DICT_KEY = "model_state_dict"


def save_checkpoint(
    path: str,
    net: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    value_rms_state: dict[str, float] | None = None,
    update: int = 0,
    config: Any = None,
    extra: dict[str, Any] | None = None,
    *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Write a resumable checkpoint (one optimizer, or separate actor/critic optimizers)."""
    if is_dataclass(config) and not isinstance(config, type):
        config = asdict(config)
    payload = {
        _STATE_DICT_KEY: net.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None and actor_optimizer is None and critic_optimizer is None
            else None
        ),
        "actor_optimizer_state_dict": (
            actor_optimizer.state_dict() if actor_optimizer is not None else None
        ),
        "critic_optimizer_state_dict": (
            critic_optimizer.state_dict() if critic_optimizer is not None else None
        ),
        "value_rms": value_rms_state,
        "update": update,
        "config": config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def _is_legacy_state_dict(payload: Any) -> bool:
    """True for old checkpoints that are a bare state_dict."""
    return isinstance(payload, dict) and _STATE_DICT_KEY not in payload


def _load_state_dict_partial(net: torch.nn.Module, state_dict: dict) -> tuple[list[str], dict[str, torch.Tensor]]:
    """Load only tensors whose name and shape match; return the skipped keys."""
    own_state = net.state_dict()
    compatible = {}
    loaded, skipped = [], {}
    for key, tensor in state_dict.items():
        if key in own_state and own_state[key].shape == tensor.shape:
            compatible[key] = tensor
            loaded.append(key)
        else:
            skipped[key] = tensor
    net.load_state_dict(compatible, strict=False)
    return loaded, skipped


def load_checkpoint(
    net: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
    strict: bool = True,
    *,
    actor_optimizer: torch.optim.Optimizer | None = None,
    critic_optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load weights (and optionally optimizer state) in place and return the checkpoint metadata."""
    payload = torch.load(path, map_location=device, weights_only=False)

    skipped_tensors: dict[str, torch.Tensor] = {}
    if _is_legacy_state_dict(payload):
        if strict:
            net.load_state_dict(payload)
        else:
            _, skipped_tensors = _load_state_dict_partial(net, payload)
        return {"update": 0, "config": None, "value_rms": None, "extra": {}, "skipped_tensors": skipped_tensors}

    if strict:
        net.load_state_dict(payload[_STATE_DICT_KEY])
        if actor_optimizer is not None or critic_optimizer is not None:
            if actor_optimizer is not None and payload.get("actor_optimizer_state_dict") is not None:
                actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            if critic_optimizer is not None and payload.get("critic_optimizer_state_dict") is not None:
                critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
        elif optimizer is not None and payload.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
    else:
        loaded, skipped_tensors = _load_state_dict_partial(net, payload[_STATE_DICT_KEY])
        print(
            f"load_checkpoint(strict=False): loaded {len(loaded)} matching-shape tensor(s), "
            f"reinitialized {len(skipped_tensors)} resized/new one(s): {list(skipped_tensors)}",
            flush=True,
        )
    return {
        "update": payload.get("update", 0),
        "config": payload.get("config"),
        "value_rms": payload.get("value_rms"),
        "extra": payload.get("extra") or {},
        "skipped_tensors": skipped_tensors,
    }
