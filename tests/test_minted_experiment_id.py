"""A minted experiment ``id`` is not a written slot: a save leaves it out and a reload mints afresh, while an id the author wrote stays written (register row WS1-b)."""

from tvbo import SimulationExperiment

G2D = {"name": "Generic2dOscillator", "iri": "tvbo:Generic2dOscillator"}


def test_a_minted_id_is_not_in_the_written_slots():
    minted = SimulationExperiment(dynamics=G2D)
    assert isinstance(minted.id, int)
    assert "id" not in minted.model_fields_set


def test_an_author_written_id_is_written():
    assert "id" in SimulationExperiment(id=41, dynamics=G2D).model_fields_set


def test_minting_counts_up_per_experiment():
    first, second = SimulationExperiment(dynamics=G2D), SimulationExperiment(dynamics=G2D)
    assert second.id == first.id + 1
