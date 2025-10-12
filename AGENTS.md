# Advice for AI/LLM Agents

See: <https://agents.md/>

## Code style

- use code style configured in `pyproject.toml`
- use concise type annotations
- preserve code comments unless instructed

## Dev environment

- Activate the `chewacla` conda environment
  - Create environment: `docs/source/install.rst`

## Testing

- Find the CI plan in the .github/workflows folder.
- Write parametrized pytests
  - Test for exception (as parameter) or no exception in a context manager
    - parameter for pytest.raises(exception) or does_not_raise() for no exception
    - from contextlib import nullcontext as does_not_raise
  - when using pytest.raises(match=text), enclose with re.escape(text)
  - label all tests with the class name
  - Avoid creating tests in test classes

## Docs

- Linux: `make -C docs html`
- Windows: `docs/make.bat html`
