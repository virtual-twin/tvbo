"""Exogenous stimuli for simulation experiments.

The public import location for :class:`Stimulus`, alongside the helpers that build one
from an ontology class or from an audio file.

There is no wrapper class: the stimulus's own methods live in
:mod:`tvbo.behaviour.perturbation` and are attached to the generated class itself, so a
stimulus carries them however it was built.
"""

import logging

import numpy as np

from tvbo.datamodel import schema as tvbo_datamodel
from tvbo.datamodel.schema import Stimulus
from tvbo.ontology import owl as ontology

try:
    import librosa
except ImportError:
    librosa = None

logger = logging.getLogger(__name__)

__all__ = ["Stimulus", "class2metadata", "load_acoustic_stimulus_from_audiofile"]


def _require_librosa():
    if librosa is None:
        raise ImportError(
            "Audio features require 'librosa'. Install with:\n  pip install tvbo[audio]\nOr: pip install librosa"
        )


def class2metadata(ontoclass):
    """Build `Stimulus` metadata from an ontology stimulus class.

    Reads the class's defining equation. The class name (identifier) and definition
    become the stimulus label and description, and every descendant `Parameter` is
    added with its default value and definition.

    Args:
        ontoclass: An owlready2 stimulus class whose `value`, `definition` and
            parameter descendants describe the perturbation.

    Returns:
        A datamodel `Stimulus` populated with the equation and parameters read
        from the ontology class.
    """
    onto_eq = ontoclass.value.first()
    metadata = tvbo_datamodel.Stimulus(label=ontoclass.name, description=ontoclass.definition.first())
    metadata.equation = tvbo_datamodel.Equation(rhs=onto_eq)
    parameters = ontology.intersection(
        ontoclass.descendants(include_self=False),
        ontology.onto.Parameter.descendants(include_self=False),
    )
    for p in parameters:
        pname = ontology.replace_suffix(p)
        metadata.parameters.update(
            {
                pname: tvbo_datamodel.Parameter(
                    name=pname,
                    value=p.defaultValue.first(),
                    description=p.definition.first(),
                )
            }
        )
    return metadata


def load_acoustic_stimulus_from_audiofile(file_path, sampling_rate=1000, duration="full"):
    """Load an audio file as a callable stimulus time course.

    Loads the waveform, resamples it to `sampling_rate`, normalises it to the
    `[-1, 1]` range, optionally truncates it to `duration`, and fits a smoothing
    spline over time (in milliseconds).

    Args:
        file_path: Path to the audio file to read (any format `librosa`
            supports).
        sampling_rate: Target sampling rate in Hz used for resampling and for
            converting sample indices to milliseconds.
        duration: `"full"` to keep the whole signal, or a length in
            milliseconds to truncate it to.

    Returns:
        A function of time (in milliseconds) that evaluates the interpolated,
        normalised audio amplitude and returns 0 outside the signal's span.

    Raises:
        ImportError: If the optional `librosa` dependency is not installed.
    """
    from scipy.interpolate import UnivariateSpline

    _require_librosa()
    audio, sr = librosa.load(file_path, sr=None)
    resampled_audio = librosa.resample(audio, orig_sr=sr, target_sr=sampling_rate)
    normalized_audio = resampled_audio / np.max(np.abs(resampled_audio))
    audio = normalized_audio if duration == "full" else normalized_audio[: int(duration / 1000 * sampling_rate)]

    t = np.arange(len(audio)) / sampling_rate * 1000
    audio_spline = UnivariateSpline(t, audio, s=0.01)

    def audio_fun(x):
        """Evaluate the audio spline at time `x`, returning 0 outside its span."""
        return (
            audio_spline(x)
            if np.isscalar(x) and t[0] <= x <= t[-1]
            else (np.where((x >= t[0]) & (x <= t[-1]), audio_spline(x), 0) if not np.isscalar(x) else 0)
        )

    return audio_fun
