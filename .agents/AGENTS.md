# Project Rules

## Coding Standards & Behavioral Constraints

- **Forbidden Constructs (`hasattr`, `isinstance`)**:
  `hasattr` and `isinstance` are strictly forbidden across the codebase. Do not use defensive type-checking or attribute-probing loops (such as `while hasattr(obj, ...)` or `isinstance(obj, ...)`). Rely on clean contract interfaces, protocol typing, or explicit polymorphism instead. Usage of `hasattr` or `isinstance` requires prior, explicit approval from the user.

- **Strict Non-Nullability (No Defensive `None`s)**:
  Defensive nullability and unnecessary `Optional` / `| None` parameters or attributes are forbidden. When in doubt, nothing is nullable. Contracts and data structures must be explicit, concrete, and non-null.
