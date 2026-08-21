"""Every curated model's dimensional standing, frozen.

The point of freezing it is that the numbers are *supposed* to move — as models
gain declarations they should shift from `underdetermined` towards `consistent` —
and a movement nobody intended is exactly what a corpus catches. Today 86 of the
108 registered models declare no unit anywhere, and 104 have no equation the
checker can settle either way — which is why it reports three answers instead of
two. A gate demanding consistency of all 108 would pressure invented declarations
into the published record, and a unit somebody made up to satisfy a test is worse
than no unit.

Update the table in the same commit as the model change, and say why.
"""

from __future__ import annotations

from collections import Counter

import pytest

from tvbo.analysis.units import CONSISTENT, INCONSISTENT, UNDERDETERMINED, check_units
from tvbo.classes.dynamics import Dynamics
from tvbo.data.registry import list_entries, resolve

VERDICTS = {
    "AdaptiveExponentialIF": (0, 0, 2),
    "Antidots": (0, 0, 4),
    "ArnoldCatMap": (0, 0, 2),
    "BetaTransformationMap": (0, 0, 1),
    "CakanObermayer": (0, 1, 2),
    "Chua": (0, 0, 3),
    "CoombesByrne": (0, 0, 4),
    "CoombesByrne2D": (0, 0, 2),
    "CoupledRoessler": (0, 0, 6),
    "CoupledStandardMaps3": (0, 0, 6),
    "DecoBalancedExcInh": (0, 0, 2),
    "DoublePendulum": (0, 0, 4),
    "DuffingForced": (0, 0, 2),
    "DumontGutkin": (0, 0, 8),
    "Epileptor": (0, 0, 6),
    "Epileptor2D": (0, 0, 2),
    "Epileptor3DStefanescuMcDonald": (0, 0, 2),
    "Epileptor5D": (0, 0, 6),
    "EpileptorCodim3": (0, 0, 3),
    "EpileptorCodim3SlowMod": (0, 0, 5),
    "EpileptorRestingState": (0, 0, 8),
    "FitzHughNagumo": (0, 0, 2),
    "FitzHughNagumo1969": (0, 0, 2),
    "ForcedPendulum": (0, 0, 2),
    "GastSchmidtKnosche_SD": (0, 0, 4),
    "GastSchmidtKnosche_SF": (0, 0, 4),
    "Generic2dOscillator": (0, 0, 2),
    "GenericLinear": (0, 0, 1),
    "Gissinger": (0, 0, 3),
    "GrebogiMap": (0, 0, 2),
    "GuckenheimerHolmes": (0, 0, 3),
    "HH_KineticScheme": (0, 0, 0),
    "HH_Tissue_Q10": (0, 0, 4),
    "Halvorsen": (0, 0, 3),
    "HenonHeiles": (0, 0, 4),
    "HenonMap": (0, 0, 2),
    "HindmarshRose": (0, 0, 3),
    "HodgkinHuxley": (0, 0, 4),
    "HodgkinHuxley_Q10": (0, 0, 4),
    "Hopfield": (0, 0, 2),
    "HyperRoessler": (0, 0, 4),
    "IaFCell": (0, 0, 1),
    "IkedaMap": (0, 0, 2),
    "IntegrateAndFire": (0, 0, 1),
    "Izhikevich2007": (0, 0, 2),
    "Izhikevich2007Cell": (0, 0, 2),
    "IzhikevichBurst": (0, 0, 2),
    "IzhikevichCell": (0, 0, 2),
    "JansenRit": (0, 0, 6),
    "JansenRit1995": (4, 0, 2),
    "KIonEx": (0, 0, 5),
    "Kuramoto": (0, 0, 1),
    "KuramotoModel2": (0, 0, 1),
    "Labyrinth": (0, 0, 3),
    "LarterBreakspear": (0, 0, 3),
    "Linear": (0, 0, 1),
    "LogisticMap": (0, 0, 1),
    "Lorenz63": (0, 0, 3),
    "Lorenz84": (0, 0, 3),
    "Lorenz96": (0, 0, 10),
    "LorenzBounded": (0, 0, 3),
    "LorenzDiffusionless": (0, 0, 3),
    "LotkaVolterraPredPrey": (0, 0, 2),
    "MagneticPendulum": (0, 0, 4),
    "MannevilleSimpleMap": (0, 0, 1),
    "ModelJansen1995": (4, 0, 2),
    "MontbrioPazoRoxin": (0, 0, 2),
    "MoreChaosExample": (0, 0, 3),
    "MorrisLecar": (0, 0, 2),
    "NLDCoupledLogisticMaps4": (0, 0, 4),
    "NoseHoover": (0, 0, 3),
    "PinskyRinzelCA3": (0, 0, 10),
    "PomeauMannevilleMap": (0, 0, 1),
    "QuadrupoleBosonHamiltonian": (0, 0, 4),
    "ReducedSetFitzHughNagumo": (0, 0, 4),
    "ReducedSetHindmarshRose": (0, 0, 6),
    "ReducedWongWang": (0, 0, 1),
    "ReducedWongWangExcInh": (0, 0, 2),
    "ReducedWongWangFunc": (0, 0, 1),
    "ReducedWongWangTvboptim": (0, 0, 1),
    "RiddledBasins": (0, 0, 4),
    "Rikitake": (0, 0, 3),
    "Roessler": (0, 0, 3),
    "RulkovMap": (0, 0, 2),
    "Sakarya": (0, 0, 3),
    "Shinriki": (0, 0, 3),
    "Spring": (2, 0, 0),
    "SprottDissipativeConservative": (0, 0, 3),
    "StandardMap": (0, 0, 2),
    "StefanescuJirsa2D": (0, 0, 4),
    "StefanescuJirsa3D": (0, 0, 6),
    "StommelThermohaline": (0, 0, 2),
    "StuartLandauOscillator": (0, 0, 2),
    "SupHopf": (0, 0, 2),
    "SwingingAtwood": (0, 0, 4),
    "TentMap": (0, 0, 1),
    "ThomasCyclical": (0, 0, 3),
    "TowelMap": (0, 0, 3),
    "TsodyksMarkram": (0, 0, 3),
    "Ueda": (0, 0, 2),
    "UlamRing4": (0, 0, 4),
    "VanDerPolForced": (0, 0, 2),
    "WilsonCowan": (0, 0, 2),
    "ZaslavskiiMap": (0, 0, 2),
    "ZerlautAdaptationFirstOrder": (0, 0, 5),
    "ZerlautAdaptationSecondOrder": (0, 0, 8),
    "ZetterbergJansen": (0, 0, 12),
    "hhcell_1": (0, 0, 4),
}
"""``model -> (consistent, inconsistent, underdetermined)`` equation counts.

Keyed by canonical database name, which is not always the file stem — `Jansen1995.yaml`
declares itself `ModelJansen1995` — and covers the subdirectories under
`database/models/` that a top-level glob misses.
"""

INCONSISTENT_MODELS = {
    "CakanObermayer": "adds mu_se (mV/ms) to E_A (mV); ratio is exactly 1000/second",
}
"""Models with a declaration a modeller has to settle, and what the checker found.

Not silently corrected: which of the two units is wrong is a modelling question,
and answering it in a test would be inventing an answer.
"""


def _counts(name):
    verdicts = check_units(Dynamics.from_file(str(resolve("Dynamics", name))))
    tally = Counter(verdict.status for verdict in verdicts)
    return tally[CONSISTENT], tally[INCONSISTENT], tally[UNDERDETERMINED]


@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_a_models_verdicts_are_unchanged(name):
    """A model's dimensional standing moves only when someone means it to."""
    assert _counts(name) == VERDICTS[name]


@pytest.mark.parametrize("name", sorted(VERDICTS))
def test_a_model_is_dimensionally_sound(name, request):
    """No curated model contradicts itself.

    The one that does is marked `xfail(strict=True)`, so fixing its declarations
    fails this test deliberately and prompts the freeze above to be updated in the
    same commit. A plain skip would let the fix land unnoticed.
    """
    if name in INCONSISTENT_MODELS:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=INCONSISTENT_MODELS[name]))

    assert _counts(name)[1] == 0


def test_the_corpus_covers_every_model():
    """A model added without a verdict is a gap, not a pass."""
    assert set(list_entries("Dynamics")) == set(VERDICTS)


def test_most_models_still_declare_nothing_to_check():
    """The reason the checker reports three answers rather than two.

    If this ever reads "all 106 consistent" without the declarations to back it,
    the checker has started guessing.
    """
    nothing_to_check = [name for name, (c, i, _) in VERDICTS.items() if not c and not i]

    assert len(VERDICTS) == 108
    assert len(nothing_to_check) == 104


def test_a_model_with_no_state_equations_reaches_no_verdicts():
    """`HH_KineticScheme` is the one model with nothing to check at all.

    An empty verdict list is the right answer for it — not a crash, and not a
    vacuous "consistent" that would count as a checked model in the tally above.
    """
    assert VERDICTS["HH_KineticScheme"] == (0, 0, 0)
    assert check_units(Dynamics.from_file(str(resolve("Dynamics", "HH_KineticScheme")))) == []
