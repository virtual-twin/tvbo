# Copyright © 2024 Charité Universitätsmedizin Berlin.
# SPDX-License-Identifier: EUPL-1.2

"""Drop the imports and bindings a generated module does not use.

A backend template cannot know which imports its output will need: whether ``BoundedSolver`` appears depends on a state variable declaring ``domain.enforce``, whether ``optax`` appears depends on the experiment carrying an optimization. Writing that out as one ``% if`` per import means every new feature has to remember to extend a condition it does not otherwise touch, and the ones already written drift — the tvb and tvboptim headers between them carried nineteen names no render referenced.

So the templates emit the imports their features *may* need and this pass removes the ones the assembled module does not reference. The decision is made from the finished source, which is the only place the answer is actually known.

Imports are pruned **in place**, never hoisted. Order is load-bearing in generated code: the tvboptim module sets ``JAX_PLATFORMS`` in ``os.environ`` and only then imports jax, so moving that import above the assignment would silently change which device the experiment runs on.

The same reasoning applies to local scaffolding a template emits for downstream code that a given spec does not produce — ``n_nodes = weights.shape[0]`` ahead of thirty conditional uses, none of which fired. :func:`prune_dead_assignments` removes those, but only when the right-hand side cannot do anything besides compute a value: dropping ``initial_state = copy.deepcopy(state)`` would skip the copy, so a call is never touched however plainly unread its result is.

Both passes are deliberately conservative — they drop a name only when the module cannot plausibly refer to it. A string literal that parses as Python counts as a reference, so a name reached by ``getattr`` or an ``eval``-ed expression survives.
"""

from __future__ import annotations

import ast
import functools

__all__ = [
    "prune",
    "prune_dead_assignments",
    "prune_unused_imports",
    "unused_import_names",
]


def _bound_names(alias: ast.alias) -> str:
    """The name an ``import`` binds: ``import a.b`` binds ``a``, ``as c`` binds ``c``."""
    return alias.asname or alias.name.split(".")[0]


def _docstrings(tree: ast.AST) -> set[int]:
    """``id()`` of every docstring node, which describes the code rather than running it."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            out.add(id(first.value))
    return out


@functools.lru_cache(maxsize=4096)
def _names_in_string(text: str) -> frozenset[str]:
    """Names *text* refers to, if it is code; nothing if it is prose.

    A string can genuinely reach a name — ``getattr(mod, "TimeSeries")``, an ``eval``-ed expression — and such a string is always valid Python. Prose is not: ``"Coupling terms"`` and ``"Additive coefficient for the second state-variable"`` do not parse, and treating the words in them as references is what kept ``Coupling`` and ``Additive`` imported into generated models that only ever used them in a ``doc=``.

    Parsing rather than word-splitting is what separates the two, and it costs nothing in safety: every string that could actually resolve a name still does. Generated modules repeat the same literals across every function, so the answer is cached, and it is a ``frozenset`` because a cached value must not be mutated by its caller.
    """
    for mode in ("eval", "exec"):
        try:
            parsed = ast.parse(text, mode=mode)
        except (SyntaxError, ValueError):
            continue
        return frozenset(n.id for n in ast.walk(parsed) if isinstance(n, ast.Name)) | frozenset(
            n.attr for n in ast.walk(parsed) if isinstance(n, ast.Attribute)
        )
    return frozenset()


def _node_reads(node: ast.AST, prose: set[int]) -> frozenset[str]:
    """Names *node* itself refers to: a ``Name``'s identifier, or what a non-docstring string literal names when it is code.

    An attribute chain needs no case of its own: its root is a ``Name`` that every walk reaches in turn, and that root is the name an import binds (``jnp`` for ``jnp.exp``). A string contributes through :func:`_names_in_string`, so a class reached by name through ``getattr`` keeps its import. Docstrings (*prose*, from :func:`_docstrings`) are excluded outright: prose naming a class is not a use of it, and counting it as one kept ``AbstractMonitor`` imported into modules that never touch it.
    """
    if isinstance(node, ast.Name):
        return frozenset((node.id,))
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
        return _names_in_string(node.value)
    return frozenset()


def _referenced(tree: ast.AST, prose: set[int]) -> set[str]:
    """Every identifier the module could be referring to, as :func:`_node_reads` counts one.

    Decorators, base classes and annotations are already ``Name``/``Attribute`` nodes, and an import statement holds neither, so it never counts as a use of what it binds. Comments are not part of the AST, so a name mentioned only in one cannot keep its import alive — deliberate, since that is what ``# noqa`` is for.
    """
    return {name for node in ast.walk(tree) for name in _node_reads(node, prose)}


def _visible_reads(tree: ast.AST, prose: set[int]) -> set[str]:
    """Names read somewhere a module-level binding would actually resolve them.

    A read inside a function that binds the same name locally resolves to *that* binding, not the module's — Python decides this per function, so a single nested ``import os`` makes every ``os`` in that function local. Counting such reads against the module keeps a top-level ``import os`` that nothing outside the function uses.

    Only a *function* body shadows, and only its own body. A class body does not form a scope its methods can see, so ``pi = 3`` in a class leaves ``x * pi`` in a method resolving to the module's ``pi``. Decorators, argument defaults and annotations are evaluated in the enclosing scope, before the function's locals exist, so they are read with the outer set too. Treating either as shadowing dropped an import the generated module then failed on with ``NameError``.
    """
    found: set[str] = set()

    def note(node: ast.AST, shadowed: frozenset[str]) -> None:
        # A local binding shadows a read by name, never one a string makes.
        if not (isinstance(node, ast.Name) and node.id in shadowed):
            found.update(_node_reads(node, prose))

    def walk(node: ast.AST, shadowed: frozenset[str]) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            outer = shadowed
            inner = shadowed | frozenset(_assigned_once(node))
            for child in ast.iter_child_nodes(node):
                # The signature is evaluated where the function is defined, not inside it.
                where = outer if child is not getattr(node, "body", None) else inner
                if isinstance(child, ast.arguments) or child in getattr(node, "decorator_list", []):
                    where = outer
                elif isinstance(child, ast.stmt):
                    where = inner
                note(child, where)
                walk(child, where)
            return
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                continue
            note(child, shadowed)
            walk(child, shadowed)

    walk(tree, frozenset())
    return found


_BLOCK_FIELDS = ("body", "orelse", "finalbody", "handlers")


def _block_sizes(tree: ast.AST) -> dict[int, int]:
    """``id(stmt)`` → how many statements share its block.

    Removing the only statement of a block leaves a header with no suite, which is a :class:`SyntaxError` rather than a tidier module: ``if flag:`` followed straight by ``return`` does not compile, and neither does an emptied ``try:``.
    """
    sizes: dict[int, int] = {}
    for node in ast.walk(tree):
        for field in _BLOCK_FIELDS:
            block = getattr(node, field, None)
            if isinstance(block, list):
                statements = [s for s in block if isinstance(s, ast.stmt)]
                for statement in statements:
                    sizes[id(statement)] = len(statements)
    return sizes


def _owns_its_lines(node: ast.stmt, lines: list[str]) -> bool:
    """Whether *node*'s source lines hold nothing but *node*.

    Rewriting by line is only safe when the lines are the statement.
    ``n = w.shape[0]; m = g()`` puts an impure call on the same line, and deleting by line would drop that call while leaving the name it bound undefined — precisely the change :func:`_is_pure` exists to prevent, and it parses, so nothing downstream catches it. A single-line suite (``if x: n = 1``) fails the same check, because the span re-parses as the ``if``, not as the assignment.
    """
    import textwrap

    end = node.end_lineno or node.lineno
    span = "\n".join(lines[node.lineno - 1 : end])
    if "noqa" in span:
        return False
    try:
        parsed = ast.parse(textwrap.dedent(span))
    except SyntaxError:
        return False
    return len(parsed.body) == 1 and type(parsed.body[0]) is type(node)


def _last_in_block(node: ast.stmt, sizes: dict[int, int]) -> bool:
    """Whether removing *node* would leave its block empty.

    A header with no suite is a :class:`SyntaxError`, not a tidier module: ``if flag:`` followed straight by ``return`` does not compile, and neither does an emptied ``try:``. Such a statement is left alone rather than replaced with ``pass``, which would trade a lint warning for a line that means nothing.
    """
    return sizes.get(id(node), 1) < 2


def _import_nodes(tree: ast.AST) -> list[ast.Import | ast.ImportFrom]:
    """Top-level and nested import statements, in source order."""
    return sorted(
        (n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))),
        key=lambda n: (n.lineno, n.col_offset),
    )


def _unused_aliases(tree: ast.AST):
    """Each import statement binding a name nothing refers to, with the aliases it should keep.

    ``from __future__`` imports and star imports are never reported, because dropping either changes semantics. A module-level import is only kept by a read a module-level binding can reach (:func:`_visible_reads`); a nested one by any reference at all.
    """
    prose = _docstrings(tree)
    used = _referenced(tree, prose)
    visible = _visible_reads(tree, prose)
    for node in _import_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if any(a.name == "*" for a in node.names):
            continue
        reachable = visible if node.col_offset == 0 else used
        keep = [a for a in node.names if _bound_names(a) in reachable]
        if len(keep) < len(node.names):
            yield node, keep


def unused_import_names(source: str) -> set[str]:
    """Names *source* imports but never refers to. Empty when *source* does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {_bound_names(a) for node, keep in _unused_aliases(tree) for a in node.names if a not in keep}


def prune_unused_imports(source: str) -> str:
    """Return *source* with unreferenced imports removed, everything else untouched.

    A statement importing several names keeps the ones that are used; a statement whose names are all unused is dropped whole. ``from __future__`` imports, star imports and lines carrying a ``noqa`` comment are always kept — the first two because dropping them changes semantics, the last because it is how a template says an import is deliberate.

    Source that does not parse is returned unchanged: reporting a syntax error is :func:`tvbo.codegen.style.format_source`'s job, and it gives a better message.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    sizes = _block_sizes(tree)
    lines = source.split("\n")
    replacements: dict[int, list[str]] = {}

    for node, keep in _unused_aliases(tree):
        if not _owns_its_lines(node, lines):
            continue
        if not keep and _last_in_block(node, sizes):
            continue
        indent = " " * node.col_offset
        if keep:
            kept = (
                ast.Import(names=keep)
                if isinstance(node, ast.Import)
                else ast.ImportFrom(module=node.module, names=keep, level=node.level)
            )
            text = [indent + ast.unparse(ast.fix_missing_locations(kept))]
        else:
            text = []
        end = node.end_lineno or node.lineno
        replacements[node.lineno - 1] = text
        for i in range(node.lineno, end):
            replacements[i] = []

    if not replacements:
        return source
    out: list[str] = []
    for i, line in enumerate(lines):
        out.extend(replacements[i] if i in replacements else [line])
    return "\n".join(out)


_ACTING_NODES = (ast.Call, ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)
"""Expression forms that can do something besides compute a value."""


def _is_pure(node: ast.AST) -> bool:
    """Whether evaluating *node* can only produce a value.

    A call is the line between the two: ``weights.shape[0]`` computes, while ``copy.deepcopy(state)`` copies, and dropping the second because nothing reads its result would change what the program does rather than only what it says. Walrus, ``await`` and ``yield`` bind or suspend, so they count as acting too.
    """
    return not any(isinstance(sub, _ACTING_NODES) for sub in ast.walk(node))


_SCOPES = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)

_PRUNABLE_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef)
"""Scopes whose bindings are private enough to remove.

Only a function's locals qualify. A class body's assignments are its attributes, read by
whatever consumes the class and never by the module that defines it —
``COUPLING_INPUTS = {...}`` looks exactly as unread as dead scaffolding, and removing it
left the generated dynamics advertising no coupling inputs at all. Module-level names are
excluded for the same reason: another module may import them.
"""


def _own_nodes(scope: ast.AST):
    """Every node belonging to *scope* itself, not descending into nested scopes.

    Binding counts have to be per scope: ``n_nodes = weights.shape[0]`` appears once in each of a dozen generated functions, and counting them together makes every one look rebound and so untouchable.
    """
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _assigned_once(scope: ast.AST) -> dict[str, int]:
    """How many times each name is bound in *scope*, by any binding form."""
    counts: dict[str, int] = {}

    def bump(name: str) -> None:
        counts[name] = counts.get(name, 0) + 1

    for node in _own_nodes(scope):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        bump(sub.id)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            if isinstance(node.target, ast.Name):
                bump(node.target.id)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            for sub in ast.walk(node.target):
                if isinstance(sub, ast.Name):
                    bump(sub.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bump(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bump(_bound_names(alias))
        elif isinstance(node, ast.withitem) and isinstance(node.optional_vars, ast.Name):
            bump(node.optional_vars.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bump(node.name)
        elif isinstance(node, ast.arg):
            bump(node.arg)
    return counts


def _read_names(tree: ast.AST, prose: set[int]) -> set[str]:
    """Names *tree* loads or declares ``global``/``nonlocal``, plus the names a non-docstring string refers to."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
        elif not isinstance(node, ast.Name) or isinstance(node.ctx, ast.Load):
            names.update(_node_reads(node, prose))
    return names


def prune_dead_assignments(source: str) -> str:
    """Remove ``name = <pure expression>`` statements whose *name* is never read.

    Only the plainest form is considered, and every condition must hold: the binding sits directly in a function body (:data:`_PRUNABLE_SCOPES`), it has a single ``Name`` target, its right-hand side cannot do anything but compute (:func:`_is_pure`), the name is bound exactly once in that scope, and nothing in that scope reads it — including a nested function that closes over it, and including a non-docstring string that parses as code.

    Requiring a single binding keeps loop accumulators and rebound temporaries intact, and the purity check is what separates the scaffolding this is meant to remove from a call whose effect the program depends on.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.split("\n")
    sizes = _block_sizes(tree)
    prose = _docstrings(tree)
    drop: set[int] = set()

    for scope in (n for n in ast.walk(tree) if isinstance(n, _PRUNABLE_SCOPES)):
        read = _read_names(scope, prose)
        bound = _assigned_once(scope)
        for node in _own_nodes(scope):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if target.id in read or bound.get(target.id, 0) != 1:
                continue
            if not _is_pure(node.value):
                continue
            if _last_in_block(node, sizes) or not _owns_its_lines(node, lines):
                continue
            end = node.end_lineno or node.lineno
            drop.update(range(node.lineno - 1, end))

    if not drop:
        return source
    return "\n".join(line for i, line in enumerate(lines) if i not in drop)


def prune(source: str) -> str:
    """Run every pruning pass over generated *source*, in dependency order.

    Assignments go first: removing one can make an import unused, and removing an import never makes an assignment dead.
    """
    return prune_unused_imports(prune_dead_assignments(source))
