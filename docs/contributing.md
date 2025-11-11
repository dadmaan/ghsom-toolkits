# Contributing to GHSOM Toolkits

Thank you for your interest in contributing to GHSOM Toolkits! This document provides guidelines and instructions for contributors.

## Ways to Contribute

- 🐛 **Bug Reports**: Report issues you encounter
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve or add documentation
- 🎨 **Visualizations**: Add new plot types or improve existing ones
- 🧪 **Tests**: Increase test coverage
- 🔧 **Bug Fixes**: Fix reported issues
- ⚡ **Performance**: Optimize code performance

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ghsom-toolkits.git
cd ghsom-toolkits
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install all optional dependencies
pip install -e .[all]
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

## Development Workflow

### Code Style

We use **Black** for formatting and **Ruff** for linting:

```bash
# Format code
black ghsom_toolkits/ tests/

# Lint code
ruff check ghsom_toolkits/ tests/

# Type check
mypy ghsom_toolkits/
```

**Code Style Guidelines:**
- Line length: 100 characters
- Use type hints for public APIs
- Follow PEP 8
- Write docstrings for all public functions (Google style)

### Writing Tests

Add tests for new features or bug fixes:

```python
# tests/test_my_feature.py
import pytest
from ghsom_toolkits import my_function

def test_my_function():
    """Test my_function with valid input."""
    result = my_function(input_data)
    assert result == expected_output

def test_my_function_edge_case():
    """Test my_function with edge case."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

Run tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ghsom_toolkits --cov-report=html

# Run specific test file
pytest tests/test_my_feature.py

# Run specific test
pytest tests/test_my_feature.py::test_my_function
```

### Documentation

Update documentation for any changes:

```python
def my_function(param: str, option: int = 5) -> bool:
    """
    Brief description of the function.

    Longer description with more details about what the function does,
    its purpose, and any important behavior.

    Parameters
    ----------
    param : str
        Description of param
    option : int, optional
        Description of option (default: 5)

    Returns
    -------
    bool
        Description of return value

    Raises
    ------
    ValueError
        When param is invalid

    Examples
    --------
    >>> from ghsom_toolkits import my_function
    >>> my_function("test", option=10)
    True
    """
    pass
```

## Contribution Guidelines

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal code to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**:
   - Python version (`python --version`)
   - ghsom-toolkits version
   - Operating system
   - Relevant dependencies

**Example:**

```markdown
## Bug Description
`visualize_ghsom_hierarchy()` fails when lookup table is empty.

## Steps to Reproduce
```python
from ghsom_toolkits import visualize_ghsom_hierarchy
visualize_ghsom_hierarchy(model, {}, "output.png")
```

## Expected Behavior
Should raise a descriptive error message.

## Actual Behavior
Raises cryptic `KeyError`.

## Environment
- Python 3.10.5
- ghsom-toolkits 0.1.0
- Ubuntu 22.04
```

### Feature Requests

When requesting features, include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other solutions you've considered
4. **Examples**: Code examples of desired usage

### Pull Requests

1. **Create an Issue First**: Discuss major changes before implementing
2. **Follow Code Style**: Use Black and Ruff
3. **Add Tests**: Maintain or increase test coverage
4. **Update Documentation**: Document new features
5. **Keep PRs Focused**: One feature/fix per PR
6. **Write Clear Commits**: Use descriptive commit messages

**PR Checklist:**

- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] No linting errors
- [ ] Type hints added
- [ ] CHANGELOG.md updated

**Commit Message Format:**

```
<type>: <short summary>

<optional detailed description>

<optional footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat: add radar chart for model comparison

fix: handle empty lookup tables in visualize_ghsom_hierarchy

docs: improve quickstart tutorial with more examples

test: add tests for adapter edge cases
```

## Project Structure

```
ghsom-toolkits/
├── ghsom_toolkits/          # Main package
│   ├── __init__.py
│   ├── plotting/            # Visualization functions
│   ├── interactive/         # Dashboard and exploration
│   ├── analysis/            # Comparison and reporting
│   ├── adapters/            # ghsom-py integration
│   ├── core/                # Internal data structures
│   ├── export/              # Export utilities
│   └── utils/               # Helper functions
├── tests/                   # Test suite
│   ├── plotting/
│   ├── interactive/
│   ├── analysis/
│   ├── adapters/
│   └── core/
├── docs/                    # Documentation
│   ├── api/                 # API reference
│   ├── tutorials/           # Tutorials
│   └── ...
├── examples/                # Example scripts
├── pyproject.toml           # Package configuration
└── README.md
```

## Testing

### Running Tests Locally

```bash
# All tests
pytest

# With coverage
pytest --cov=ghsom_toolkits --cov-report=term-missing

# Specific module
pytest tests/plotting/

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Coverage

Aim for >90% coverage for new code:

```bash
pytest --cov=ghsom_toolkits --cov-report=html
open htmlcov/index.html  # View coverage report
```

## Documentation

### Building Docs Locally

```bash
# Install mkdocs
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve docs locally
cd ghsom-toolkits
mkdocs serve

# Open http://localhost:8000
```

### Documentation Structure

- `docs/index.md` - Homepage
- `docs/installation.md` - Installation guide
- `docs/quickstart.md` - Quick start guide
- `docs/tutorials/` - Step-by-step tutorials
- `docs/api/` - API reference
- `docs/gallery.md` - Visualization gallery

## Code Review Process

1. **Automated Checks**: CI runs tests, linting, type checking
2. **Maintainer Review**: A maintainer reviews your PR
3. **Feedback**: Address any comments or requested changes
4. **Approval**: Once approved, your PR will be merged

## Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.x.0`
4. Push tag: `git push origin v0.x.0`
5. GitHub Actions automatically builds and publishes to PyPI

## Community Guidelines

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on the code, not the person
- Assume good intentions

## Questions?

- **Discussions**: [GitHub Discussions](https://github.com/dadmaan/ghsom-toolkits/discussions)
- **Issues**: [GitHub Issues](https://github.com/dadmaan/ghsom-toolkits/issues)
- **Email**: See package metadata

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to GHSOM Toolkits!** 🎉
