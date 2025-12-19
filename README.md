# spurs



## Install

`pip install spurs`

`conda install spurs`

## How to use

Installation will create a command line script `spurs`:

```bash
spurs 20150608_20170808.int
```
By default, will output to file `20150608_20170808.unw` matching the name.

To increase the tolerance (from `pi/10` radians) for faster convergence, showing iteration stats:
```bash
spurs 20150608_20170808.int -o 20150608_20170808.unw --tol .5 --debug
```

See `spurs --help` for all options.

Note that for input interferograms which aren't complex, float32 binary format, `gdal` must be installed. E.g. for a VRT input:

```bash
spurs 20150608_20170808.vrt -o 20150608_20170808.unw
```



spurs is an open source implementation of [1]:

## Development Setup

### Prerequisites

- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver

Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Getting Started

1. Clone the repository:
```bash
git clone https://github.com/scottstanie/spurs.git
cd spurs
```

2. Install dependencies:
```bash
uv sync
```

This will create a virtual environment in `.venv` and install all dependencies from `uv.lock`.

3. Install with development dependencies:
```bash
uv sync --extra dev
```

This installs additional tools like `pytest`, `ruff`, `mypy`, and `jax`.

### Running Tests

```bash
uv run pytest
```

For tests with JAX backend:
```bash
uv sync --extra jax
uv run pytest
```

### Running the CLI Locally

```bash
uv run spurs <input-file>
```

### Code Quality

Run linting:
```bash
uv run ruff check .
```

Run type checking:
```bash
uv run mypy spurs
```

### Optional Dependencies

- **JAX backend**: For GPU-accelerated unwrapping
  ```bash
  uv sync --extra jax
  ```

### Adding Dependencies

```bash
uv add <package-name>
```

This will update both `pyproject.toml` and `uv.lock`.

## References

1. Chartrand, Rick, Matthew T. Calef, and Michael S. Warren. "Exploiting Sparsity for Phase Unwrapping." IGARSS 2019-2019 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2019.
