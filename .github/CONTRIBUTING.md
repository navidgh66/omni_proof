# Contributing to OmniProof

Thank you for your interest in contributing to OmniProof!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/navidgh66/omni_proof.git
   cd omni_proof
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (GEMINI_API_KEY, PINECONE_API_KEY, etc.)
   ```

## Running Tests

Run all tests:
```bash
pytest -v --tb=short
```

Run only unit tests:
```bash
pytest tests/unit/ -v
```

Run only integration tests:
```bash
pytest tests/integration/ -v
```

Run with coverage:
```bash
pytest --cov=src/omni_proof --cov-report=term-missing
```

## Code Quality

Lint your code:
```bash
ruff check src/ tests/
```

Format your code:
```bash
ruff format src/ tests/
```

Type checking (optional):
```bash
mypy src/
```

## Pull Request Guidelines

1. **Keep PRs focused**: One feature or fix per PR
2. **Write tests**: Add unit tests for new functionality
3. **Follow existing patterns**: Match the project's architecture and style
4. **Update documentation**: Add docstrings and update README if needed
5. **Run tests locally**: Ensure all tests pass before submitting
6. **Write clear commit messages**: Use descriptive commit messages

## Code Style

- Python 3.11+ features are encouraged
- Follow PEP 8 (enforced by Ruff)
- Use type hints where possible
- Keep line length to 100 characters
- Write docstrings for public APIs

## AI-Assisted Development

This project includes a `CLAUDE.md` file and `.claude/` directory for [Claude Code](https://claude.ai/claude-code) users. These provide project context for AI-assisted development. Non-Claude-Code users can safely ignore these files.

## Questions?

Open an issue for discussion before starting major changes.
