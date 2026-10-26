"""A panel letter in bsplot's standard place, centred on the top of the left spine, leaves the y axis's shared exponent readable.

matplotlib prints that exponent (``×10⁻³``) at the same corner, so the emitted letter helper moves it to the letter's right; a letter in any other corner leaves it where matplotlib put it.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

import tvbo.datamodel.pydantic as P  # noqa: E402
from tvbo.adapters import bsplot  # noqa: E402

STANDARD = {"option": "numbers", "fontsize": 16}
UPPER_RIGHT = {"option": "numbers", "fontsize": 16, "x_shift": 0.98, "y_shift": -0.02, "ha": "right", "va": "top"}


@pytest.fixture(scope="module")
def panel_number():
    figure = P.Figure(
        name="fig",
        layout="a",
        panels={
            "a": P.Panel(
                panel_key="a",
                kind="cartesian",
                layers=[P.Layer(used=P.DataRef(iri="tvbo:exp/None/exp-999", output="y"), encoding=P.Encoding(x="x", y="y"))],
            )
        },
    )
    helpers: dict = {}
    exec(compile(bsplot.render_code(figure, ".", "out.png"), "<figure>", "exec"), helpers)
    return helpers["_panel_number"]


def _extents(panel_number, placement):
    """The drawn letter's and exponent's display extents on a small panel whose y values need ×10⁻³."""
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=100)
    ax.plot([0.0, 1.0], [0.0005, 0.0011])
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(formatter)
    panel_number(ax, "g", placement)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    extents = (
        ax.texts[-1].get_window_extent(renderer),
        ax.yaxis.offsetText.get_window_extent(renderer),
        ax.yaxis.offsetText.get_text(),
        ax.bbox.x0,
    )
    plt.close(fig)
    return extents


def test_the_exponent_moves_right_of_a_letter_on_the_left_spine(panel_number):
    letter, exponent, text, _ = _extents(panel_number, STANDARD)
    assert text, "the panel must print an exponent for the test to mean anything"
    assert exponent.x0 >= letter.x1


def test_a_letter_in_another_corner_leaves_the_exponent_at_the_axes_edge(panel_number):
    _, exponent, text, left = _extents(panel_number, UPPER_RIGHT)
    assert text
    assert exponent.x0 == pytest.approx(left, abs=1.0)
