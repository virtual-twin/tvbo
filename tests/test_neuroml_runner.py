"""The jNeuroML runner must start the JVM headless.

`-nogui` suppresses jNeuroML's plot window but leaves AWT enabled, so on macOS the JVM opens a WindowServer connection and blocks there indefinitely: a simulation that finishes in a second instead sits until the subprocess timeout fires. pyNeuroML pairs `-nogui` with the headless property for this reason, and the adapter builds its own command line rather than going through it.
"""

from __future__ import annotations

import subprocess

import pytest

from tvbo.adapters import neuroml


class _Captured(Exception):
    """Carries the argv the runner assembled, so the test needs no jNeuroML jar."""

    def __init__(self, argv: list[str]):
        self.argv = argv


def test_jneuroml_is_launched_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    def _capture(argv, *args, **kwargs):
        if argv and argv[0] == "java":
            raise _Captured(list(argv))
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(neuroml._subprocess, "run", _capture)

    with pytest.raises(_Captured) as caught:
        neuroml.run_lems_example("LEMS_NML2_Ex21_CurrentBasedSynapses.xml")

    argv = caught.value.argv
    assert "-Djava.awt.headless=true" in argv
    assert argv.index("-Djava.awt.headless=true") < argv.index("-jar"), "the property must precede -jar to reach the JVM"
