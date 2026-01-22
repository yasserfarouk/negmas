# Agent Guidelines for negmas

## ⚠️ CRITICAL: GIT PUSH POLICY ⚠️

**NEVER NEVER NEVER NEVER PUSH WITHOUT EXPLICIT INSTRUCTION**

**ABSOLUTELY NO EXCEPTIONS. NEVER PUSH TO GITHUB WITHOUT THE USER EXPLICITLY SAYING "PUSH" OR GIVING A DIRECT COMMAND TO PUSH.**

**DO NOT PUSH EVEN IF:**
- All tests pass locally
- All tests pass on CI
- The changes look good
- You think it's ready
- You want to check CI

**ALWAYS:**
1. Run tests locally
2. Show results to user
3. **WAIT FOR EXPLICIT "PUSH" COMMAND**
4. Only then run `git push`

## Build/Test Commands
```bash
pytest                                    # Run all tests
pytest tests/core/test_sao.py             # Run single test file
pytest tests/core/test_sao.py::test_name  # Run single test function
pytest -k "pattern"                       # Run tests matching pattern
NEGMAS_FASTRUN=True pytest                # Fast test mode
```

## Lint/Format Commands
```bash
ruff check --fix --unsafe-fixes           # Lint and auto-fix
ruff format                               # Format code
pre-commit run --all-files                # Run all pre-commit hooks
```

## Code Style
- **Line length**: 88 chars, **indent**: 4 spaces, **quotes**: double
- **Python target**: 3.10+ (use `|` for unions: `int | None`)
- **Imports**: `from __future__ import annotations` first, then stdlib, third-party, local
- **Naming**: PascalCase for classes, snake_case for functions/variables, `_prefix` for private
- **Types**: Use type hints everywhere; use `TYPE_CHECKING` block for type-only imports
- **Classes**: Use `@define` from attrs for data classes; every module exports via `__all__`
- **Errors**: Raise `ValueError`/`TypeError` with descriptive messages; use `negmas.warnings` for deprecations
- **Source location**: Main code in `src/negmas/`, tests in `tests/`

## Agent Sandbox
- **File Storage**: All files created for documentation, internal testing, or any other purpose must be placed in the `coding_agents/` directory. This is to avoid polluting the root directory.

## Git Workflow
- **Do not push**: Never push commits to origin without explicit user approval. Commit changes locally, but wait for the user to say "push" or similar before running `git push`.
- **Always add tests**: Every new feature must have corresponding tests. Add tests to `tests/core/` for core functionality.
- **Test naming**: Use `test_<feature_name>` naming convention for test functions.
- **Run tests**: Always run relevant tests after implementing a feature to verify it works.

## Publications List Maintenance

**Task:** Periodically update the publications list (approximately every 2 weeks).

**Files to update:**
- `docs/publications.rst` - Full comprehensive list with all details
- `README.rst` - Condensed "Papers Using NegMAS" section

**Search queries to use (Google Scholar):**
- `"negmas" negotiation`
- `"SCML" "supply chain" negotiation agent`
- `"supply chain management league" ANAC`
- `"automated negotiating agents competition" SCML`

**Entry format for publications.rst:**
```rst
- Author1, A., Author2, B. (Year).
  `Paper Title <URL>`_.
  In: *Venue Name*. Publisher. *Cited by N*
```

**Process:**
1. Search using the queries above
2. Filter for papers that actually use/cite NegMAS or SCML
3. Add new entries to the appropriate category in `docs/publications.rst`
4. For significant papers (high citations or major venues), also add to `README.rst`
5. Update the "Last updated" date in publications.rst
6. Verify the documentation builds: `cd docs && make html`

## Ecosystem Maintenance

**Task:** Periodically review and update the NegMAS ecosystem documentation (approximately every 2 weeks).

**Files to update:**
- `docs/overview.rst` - Ecosystem section in main documentation
- `README.rst` - Ecosystem section in project README

**Repositories to check (all under github.com/yasserfarouk/):**

| Category | Repositories |
|----------|--------------|
| Competition Frameworks | `anl`, `anl2025`, `scml` |
| Agent Repositories | `anl-agents`, `scml-agents` |
| Bridges & Adapters | `negmas-geniusweb-bridge`, `negmas-negolog`, `geniusbridge` |
| Extensions | `negmas-llm`, `negmas-rl-tutorial` |
| Visualization | `scml-vis` |
| Language Bindings | `jnegmas` |

**Process:**
1. Check each repository for recent updates, new releases, or status changes
2. Look for new repositories in the yasserfarouk GitHub account related to NegMAS
3. Verify all repository links are still valid
4. Update descriptions if repository purposes have changed
5. Add any new ecosystem projects to the appropriate category
6. Remove or mark as deprecated any archived/abandoned projects
7. Ensure both `docs/overview.rst` and `README.rst` are in sync
