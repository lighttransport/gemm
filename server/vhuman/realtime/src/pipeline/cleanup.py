"""Attempt every release while preserving the original runtime failure."""
import sys


def close_resources(*resources):
    primary = sys.exception()
    errors = []
    for resource in resources:
        if resource is not None:
            try:
                resource.close()
            except Exception as error:
                errors.append(error)
    if primary is not None:
        for error in errors:
            primary.add_note(f"Cleanup failed: {type(error).__name__}: {error}")
    elif errors:
        for error in errors[1:]:
            errors[0].add_note(f"Additional cleanup failure: {type(error).__name__}: {error}")
        raise errors[0]
