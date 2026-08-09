# CLAUDE.md

## Style
- Comments and docstrings state what code IS and why — never what it
  isn't, doesn't do, or used to be ("no X anywhere", "unlike before",
  "without any Y"). No process narrative. Design history belongs in
  docs/canon_design.md, not in code.
- When in doubt, don't add defaults to input parameters. A default is
  a decision made for the caller; hyperparameters, optimizers and
  sizes are the caller's to make. Default only what is definitional to
  the abstraction.
- An explicit user directive outranks any benchmark result. No single
  test measurement ever justifies reverting or reinterpreting a
  directive (e.g. "the D channel is de/dt, no time constants"). When a
  measurement conflicts with a directive: keep the directive, report
  the measurement, recalibrate the test's expectations to the mandated
  configuration.

## Engineering
- This is a difficult, novel, engineered library, not a throwaway
  script. Reach for the RIGHT abstraction, not the expedient one. The
  cost of janking leaky abstractions in like a bedroom coder compounds
  and grinds the work to a halt.
- The existing code, tests, and design docs are days of poorly
  supervised vibecoding under active cleanup. They carry NO authority
  as precedent: "matching the existing pattern" is never a
  justification. Judge every inherited pattern before extending it;
  when it smells, flag it for death rather than replicate it.
- Keep abstractions clean and encapsulated. Do not leak kind-specific
  logic onto a base type: a distinct kind of thing gets its own subtype
  that dispatches by type, never `if self.members is not None`-style
  branches scattered on the base (e.g. composite behaviour belongs on a
  Composite(NodeDef) subtype, not on NodeDef). A base advertised as a
  container/contract must stay one.
- When a clean design needs a bigger change than the quick hack, say so
  and do the clean change; do not smuggle logic into the wrong layer to
  save effort.
- Vocabulary discipline: a term not explicitly defined in the glossary of
  docs/redesign_again.md is not used in code or docstrings, ever. Extend
  the glossary first. ("canonical" is retired; the word is "contract".)
- Condemned constructs — NEVER write any of these without the user's
  explicit approval in the current conversation:
  - metadata attributes stapled onto functions (`fn._tag = ...`);
    existing sites are marked `# FN-TAG antipattern` and are dying, not
    a pattern to extend. Metadata lives on the def / in the specs.
  - `getattr(x, name, default)` — the read side of the same disease:
    absence silently masked by a default.
  - `isinstance` — type-switching that belongs in polymorphism (a
    method the type answers), not branches at call sites.

## Interaction
- Never push to a remote without asking permission first, every time.
  Local commits are fine; anything leaving the machine is not yours to
  decide.
- No unsolicited lifestyle advice. Do not suggest pausing, resting,
  stopping, banking a milestone, session length, or picking work up
  "fresh" another time. When told to continue, continue and do the
  work. No meta-commentary on pacing or whether to keep going.

## Environment
- Type: Conda
- Name: `nodejax`
- Definition: `environment.yml`

## Commands
- **Build/Update Env**: `conda env update -f environment.yml --prune`
- **Test**: `/opt/miniconda3/envs/nodejax/bin/python -m pytest nodejax/test`
- **Test Single**: `/opt/miniconda3/envs/nodejax/bin/python -m pytest {file}`

## Prose
- All prose you produce is written in full sentences: commas, colons,
  and periods, never dash-interrupted fragments. This is not a
  code-comment rule; it covers docs, docstrings, commit messages, and
  chat. Everything.
- Before referencing a concept by name, ask whether the reader is
  actually on the same page: has the term been defined or explained in
  the text BEFORE this point, or is it common knowledge for the
  audience? A coinage from your own commit messages, chat, or head is
  not established vocabulary. Define it in place or use plain words.
- Docstrings and docs never reference the development conversation.
  They are read by strangers with no context: no 'as discussed', no
  'the new line', no narrative that assumes the reader watched the
  code evolve.
- Forbidden words and constructs, same scope:
  - quietly
  - honest / honestly (as filler praise for one's own prose)
  - load-bearing
  - contrapositions in every sentence; thats not X - thats Y