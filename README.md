# cieran

![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)
[![Python package](https://github.com/matthhong/cieran/actions/workflows/python-package.yml/badge.svg)](https://github.com/matthhong/cieran/actions/workflows/python-package.yml)

Cieran is a Python package for designing visualization colormaps via active preference learning.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install cieran.

```bash
pip install cieran
```

## Usage

See altair-demo.ipynb for more details.

```python
from cieran import Cieran
cie = Cieran(draw=draw_map)
cie.set_color("#f88253")

cieran.teach(); cieran.search()
```

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)

## Acknowledgements
Cieran borrows code from the [APReL](https://github.com/Stanford-ILIAD/APReL) package licensed under the MIT License. We would like to thank the original authors for their contributions.
