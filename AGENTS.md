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
    - each parameter set has an 'id'
    - when using pytest.raises(match=text), enclose with re.escape(text)
  - label all tests with the class name
  - Avoid creating tests in test classes

Example:

```python
from contextlib import nullcontext as does_not_raise
@pytest.mark.parametrize(
    "initial, set_value, expected, context",
    [
        pytest.param(None, "default", "default", does_not_raise(), id="default"),
        pytest.param(
            None,
            "invalid_mode",
            None,
            pytest.raises(ValueError, match=re.escape("Invalid mode: invalid_mode")),
            id="invalid_mode",
        ),
    ],
)
def test_Chewacla_mode_setter(initial, set_value, expected, context):
    with context:
        c = Chewacla({"a": "x+"}, {"d": "y+"})
        if initial is not None:
            c.mode = initial
        c.mode = set_value
        if expected is not None:
            assert c.mode == expected
```

## Docs

- Linux: `make -C docs html`
- Windows: `docs/make.bat html`
