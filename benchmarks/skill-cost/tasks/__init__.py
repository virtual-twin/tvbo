"""Benchmark task registry."""
from __future__ import annotations

from .whole_brain_sim import TASK as whole_brain_sim

TASKS = {
    whole_brain_sim.name: whole_brain_sim,
}

__all__ = ["TASKS", "whole_brain_sim"]
