"""What a refusal says, whether it is raised as itself or wrapped by a record's validation."""

from pydantic import ValidationError


def said(error: BaseException) -> str:
    """The message *error* carries: the one error a record's validation wraps, or *error*'s own."""
    if isinstance(error, ValidationError) and error.error_count() == 1 and "error" in error.errors()[0].get("ctx", {}):
        return str(error.errors()[0]["ctx"]["error"])
    return str(error)
