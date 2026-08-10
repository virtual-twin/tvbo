"""Hand-written behaviour attached to the generated model classes.

Each module here holds one class's user-facing helpers as a plain mixin. ``hatch_build``
discovers them by name — ``EventBehaviour`` attaches to ``Event`` — and makes each a base
of both generated forms, so the helpers are present on every object however it was built:
loaded through LinkML, validated through Pydantic, or constructed by hand. Adding
behaviour to a class is creating a file here; there is nothing to register.

The mapping lives in the mixin's name rather than in the schema on purpose. The schema is
language-neutral and is what the OWL export is generated from, so a Python import path
there would state one language's mechanism as if it were a fact about the model.

Three rules keep this working:

* A mixin is a plain class, never a ``BaseModel`` subclass. Two models in the bases would
  merge their ``model_config``, letting a mixin silently change validation for the class
  it is attached to.
* A mixin must not import :mod:`tvbo.datamodel` at module scope. The generated modules
  import these, so a module-scope import back into them is a cycle; import inside the
  method that needs it.
* A mixin's name must end in ``Behaviour`` and its stem must name a class the schema
  defines, or the build fails — otherwise it would attach to nothing, silently.
"""
