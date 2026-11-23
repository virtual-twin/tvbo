"""Benchmark task registry."""

from __future__ import annotations

from .whole_brain_sim import TASK as whole_brain_sim
from .whole_brain_sim import TASK_GENERAL as whole_brain_sim_general
from .whole_brain_sim import TASK_MVP as whole_brain_sim_mvp

TASKS = {
    whole_brain_sim.name: whole_brain_sim,
    whole_brain_sim_general.name: whole_brain_sim_general,
    whole_brain_sim_mvp.name: whole_brain_sim_mvp,
}

__all__ = ["TASKS", "whole_brain_sim", "whole_brain_sim_general", "whole_brain_sim_mvp"]
