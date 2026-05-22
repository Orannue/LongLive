# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal temporal RoPE helpers used by multi-shot generation."""

import torch


def _rope_base_angles_from_freqs(freqs_t: torch.Tensor) -> torch.Tensor:
    """Return temporal RoPE base angles without calling torch.angle.

    ``freqs_t`` is produced by ``rope_params(max_seq_len, dim)`` with the
    default theta=10000. Row 1 would contain ``polar(1, inv_freq)``. Computing
    that inverse-frequency vector directly avoids a CUDA complex ``angle``
    kernel that can fail with ``invalid argument`` on some driver/PyTorch
    combinations.
    """
    rope_dim = int(freqs_t.shape[1]) * 2
    return 1.0 / torch.pow(
        10000,
        torch.arange(0, rope_dim, 2, device=freqs_t.device, dtype=torch.float64).div(rope_dim),
    )


def select_temporal_offset_for_sample(
    temporal_offset,
    sample_idx: int,
    f: int,
    start_frame: int = 0,
):
    """Select the offset slice that applies to one sample.

    ``temporal_offset`` accepts a scalar, ``[B]`` per-sample constants,
    ``[F]`` shared per-frame offsets, or ``[B, F]`` per-sample per-frame
    offsets.  The returned value is still interpreted by
    ``compute_temporal_freqs`` so full-length and local slices both work.
    """
    if temporal_offset is None:
        return 0.0
    if torch.is_tensor(temporal_offset):
        if temporal_offset.ndim == 0:
            return temporal_offset
        if temporal_offset.ndim == 1:
            # Usually this is a shared [F] vector. If it is too short to cover
            # the requested frame range, treat it as [B] constants.
            if temporal_offset.numel() == f or temporal_offset.numel() >= start_frame + f:
                return temporal_offset
            return temporal_offset[sample_idx]
        if temporal_offset.ndim == 2:
            return temporal_offset[sample_idx]
        raise ValueError(
            "temporal_offset tensor must be scalar, [B], [F], or [B, F], "
            f"got shape={tuple(temporal_offset.shape)}"
        )
    if isinstance(temporal_offset, (list, tuple)):
        if not temporal_offset:
            return 0.0
        if isinstance(temporal_offset[0], (list, tuple)):
            return torch.as_tensor(temporal_offset[sample_idx])
        if len(temporal_offset) == f or len(temporal_offset) >= start_frame + f:
            return torch.as_tensor(temporal_offset)
        return temporal_offset[sample_idx]
    return temporal_offset


def compute_temporal_freqs(
    freqs_t: torch.Tensor,
    f: int,
    start_frame: int,
    t_scale: float,
    device: torch.device,
    method: str = "linear",
    original_seq_len: int | None = None,
    temporal_offset: float = 0.0,
) -> torch.Tensor:
    """Compute linear temporal RoPE freqs with an optional multi-shot offset."""
    if method != "linear":
        raise ValueError(f"Only linear temporal RoPE is supported in this release, got {method}.")
    if original_seq_len is not None:
        raise ValueError("original_seq_len is not used by the release linear RoPE path.")
    if temporal_offset is None:
        temporal_offset = 0.0
    if (
        t_scale == 1.0
        and not torch.is_tensor(temporal_offset)
        and float(temporal_offset) == 0.0
    ):
        return freqs_t[start_frame:start_frame + f]

    base_angles = _rope_base_angles_from_freqs(freqs_t)
    positions = torch.arange(f, device=device, dtype=torch.float64) + start_frame
    if torch.is_tensor(temporal_offset):
        offset = temporal_offset.to(device=device, dtype=torch.float64)
        if offset.ndim == 0:
            positions = positions + offset
        elif offset.ndim == 1:
            if offset.numel() == f:
                positions = positions + offset
            elif offset.numel() >= start_frame + f:
                positions = positions + offset[start_frame:start_frame + f]
            else:
                raise ValueError(
                    "temporal_offset length is too short for requested RoPE "
                    f"range: len={offset.numel()}, start={start_frame}, f={f}"
                )
        else:
            raise ValueError(
                "compute_temporal_freqs expects a scalar or 1D temporal_offset "
                f"after sample selection, got shape={tuple(offset.shape)}"
            )
    else:
        positions = positions + float(temporal_offset)
    positions = positions * t_scale
    angles = positions.unsqueeze(-1) * base_angles.unsqueeze(0)
    return torch.polar(torch.ones_like(angles), angles)
