"""Custom quartodoc markdown renderer that supports ``Yields:`` sections.

quartodoc's stock ``MdRenderer`` renders ``Returns:`` and ``Raises:`` sections
but raises ``NotImplementedError: Unsupported type: DocstringSectionYields`` on
the ``Yields:`` section griffe emits for generator functions (e.g.
``tvbo.adapters.smallscale.lowering.connectivity_pairs``). ``Yields:`` is valid
Google-style docstring syntax, so this subclass adds the two missing dispatch
methods, rendering a ``Yields`` section as the same name/type/description table
quartodoc already uses for ``Returns``.

Wired in via the generated quartodoc config's ``renderer: _renderer.py`` key
(see ``scripts/tvbo_package_struct.py``). quartodoc imports this module from the
working directory and instantiates the class named ``Renderer``, so that name is
load-bearing. The leading underscore keeps Quarto from treating the file as a
project input page.
"""

from plum import dispatch
from quartodoc._griffe_compat import docstrings as ds
from quartodoc.renderers.md_renderer import MdRenderer, ParamRow


class Renderer(MdRenderer):
    # Distinct from MdRenderer.style ("markdown") so ``Renderer._registry``
    # doesn't reject the subclass at class-definition time.
    style = "markdown-yields"

    @dispatch
    def render(self, el: ds.DocstringSectionYields):
        rows = list(map(self.render, el.value))
        header = ["Name", "Type", "Description"]
        return self._render_table(rows, header, "returns")

    @dispatch
    def render(self, el: ds.DocstringYield):
        # Mirrors the DocstringReturn handler: a yielded value carries a
        # description and optional annotation, but no name or default.
        return ParamRow(
            el.name,
            el.description,
            annotation=self.render_annotation(el.annotation),
        )
