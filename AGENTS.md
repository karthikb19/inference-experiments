# Repository guide

- Use `uv`: `uv sync`, `uv run pytest`, and `uv run ruff check .`.
- Keep production code in `src/inference_experiments/` and tests in `tests/`.
- Write small, typed functions with explicit inputs, outputs, and failures; avoid hidden state.
- Keep inline code comments to one line and use them only to explain why code exists.
- Do not use multi-line comments to explain a change; keep that context in documentation or commits.
- Model structured data precisely. Avoid bare `dict`, `Any`, and opaque shapes such as
  `list[dict[str, Any]]`; prefer a typed model (`pydantic`, `TypedDict`, or a named tuple).
  Use `Any` only as a last resort when the schema cannot be expressed.
- Add or update focused tests for every behavior change, including edge and failure cases.
- Tests must be deterministic, fast, and offline; mock network, Modal, clocks, randomness, and expensive Torch work.
- Run `uv run ruff format --check .`, `uv run ruff check .`, and `uv run pytest` before handing off.


## Dev Environment - CS Servers

Run Python and GPU checks inside the user's Apptainer gdevbox environment. The host sandbox Python environment is not sufficient for GPU validation.

Start an interactive Bash shell so the aliases in ~/.bashrc are available.
Run module load apptainer.
Run gdevbox and wait for the container shell prompt.
Inside the container, run cd ~/inference-experiments.
Run source .venv/bin/activate before running Python commands.
For the terminal tool, bash -ic 'module load apptainer; gdevbox' with a PTY opens the container. Send subsequent commands to that same terminal session. If container setup fails with a socket operation not permitted error in the sandbox, request elevated execution through the tool's approval mechanism.

The verified container path is /root/inference-experiments; the host workspace path is /u/ycb7rx/inference-experiments. After activation, sys.executable is /root/inference-experiments/.venv/bin/python, and sys.prefix != sys.base_prefix is true. Activation must be repeated in new container shell sessions.

## Git workflow
- you will need to do this in the specified dev environment
- Prefix every branch name with `kb/`.
- Create draft pull requests when work is ready to share, and keep their status
  and links visible in handoffs.
- Write accurate, durable PR descriptions. State what changed, why it changed,
  the user or developer impact, relevant historical context or root cause, and
  the validation performed. Update the description when the scope changes.

## Project knowledge

Keep durable project context, plans, and decisions in `knowledge/`.

- Add architecture decision records (ADRs) to `knowledge/decisions/` as
  `YYYY-MM-DD-<name>.md`, using `knowledge/decisions/TEMPLATE.md`.
- Every ADR must include, in this order: Context, Decision, Alternatives,
  Consequences, and Surface Areas. ADRs are not required for bug fixes,
  documentation fixes, dependency bumps, or changes without a design decision.
  When uncertain, add a concise three-line ADR instead: it is cheap and keeps
  the decision record clear.
- Use `knowledge/scratch/` for informal scratch notes, working plans, and
  temporary project thinking.
