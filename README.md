

The project consists of a Python module, `async_sim.components`, which provides objects that are composed
into applications found under `examples`.

Our algorithms are built atop v0.6 of the [LEAP library](https://leap-gmu.readthedocs.io/) (see `leap_ec` [on PyPI](https://pypi.org/project/leap-ec/))


## Setup

Optionally set up a virtual environment (ex. via `python -m venv ./venv && source venv/bin/activate`).

Then install the package:

```bash
pip install -e .
```

Ensure the tests pass:

```bash
pip install pytest
pytest
```