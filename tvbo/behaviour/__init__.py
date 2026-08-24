"""Hand-written behaviour attached to the generated model classes.

Each module here holds one class's user-facing helpers as a plain mixin. ``hatch_build`` discovers them by name — ``EventBehaviour`` attaches to ``Event`` — and makes each a base of both generated forms, so the helpers are present on every object however it was built:
loaded through LinkML, validated through Pydantic, or constructed by hand. Adding behaviour to a class is creating a file here; there is nothing to register.

The mapping lives in the mixin's name rather than in the schema on purpose. The schema is language-neutral and is what the OWL export is generated from, so a Python import path there would state one language's mechanism as if it were a fact about the model.

A mixin may hook construction, by defining ``__post_init__``. ``@dataclass`` writes ``__init__`` on the generated class itself, so a base cannot reach that; but the generated ``__post_init__`` ends in ``super().__post_init__(**kwargs)`` and the mixin sits directly before ``YAMLRoot``, so the hook runs once the generated normalization is done, on every construction path. It must forward to ``super()`` in turn. What it cannot do is see the keywords as they were passed — defaults are already applied by then, so a hook cannot tell an explicitly passed default from an absent one, and it cannot rewrite a keyword the class has no slot for.

Three rules keep this working:

* A mixin is a plain class, never a ``BaseModel`` subclass. Two models in the bases would
  merge their ``model_config``, letting a mixin silently change validation for the class
  it is attached to.
* A mixin must not import :mod:`tvbo.datamodel` at module scope. The generated modules
  import these, so a module-scope import back into them is a cycle; import inside the
  method that needs it.
* A mixin's name must end in ``Behaviour`` and its stem must name a class the schema
  defines, or the build fails — otherwise it would attach to nothing, silently.

One mixin attaches by a schema rule instead of by its name: :class:`IriEnrichable`, which ``hatch_build`` gives to every class the schema declares an ``iri`` on. It lives in :mod:`tvbo.behaviour._enrich`, under a leading underscore, because the name rule above would otherwise look for a class called ``Enrich``. A behaviour mixin may refine what it provides — ``DynamicsBehaviour._from_ontology`` is how a model reaches the ontology — and reach the generic implementation through ``super()``, since behaviour is listed first.
"""
