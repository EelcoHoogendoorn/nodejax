# Project Rules

## Coding Standards & Behavioral Constraints

- **Forbidden Constructs (`hasattr`, `isinstance`)**:
  `hasattr` and `isinstance` are strictly forbidden across the codebase. Do not use defensive type-checking or attribute-probing loops (such as `while hasattr(obj, ...)` or `isinstance(obj, ...)`). Rely on clean contract interfaces, protocol typing, or explicit polymorphism instead. Usage of `hasattr` or `isinstance` requires prior, explicit approval from the user.
