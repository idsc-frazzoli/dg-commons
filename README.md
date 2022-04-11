# dg-commons (Driving Games common tools)

[![CircleCI](https://circleci.com/gh/idsc-frazzoli/dg-commons/tree/master.svg?style=svg&circle-token=19e654261b71d1fa32c2991574d17dde93a23502)](https://circleci.com/gh/idsc-frazzoli/dg-commons/tree/master)
[![codecov](https://codecov.io/gh/idsc-frazzoli/dg-commons/branch/master/graph/badge.svg?token=jqhkIa4fzB)](https://codecov.io/gh/idsc-frazzoli/dg-commons)
[![PyPI version](https://badge.fury.io/py/dg-commons.svg)](https://badge.fury.io/py/dg-commons)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package contains common tools that ease the development and quick prototyping of autonomous vehicles with a focus
on multi-agent planning and decision-making.

## Highlights

<table>
<tr>
<td>

![solutions-multilaneint5psets-solutions-solver2puresecuritymNEfact2noextra-joint-joint-0](https://user-images.githubusercontent.com/18750753/162696592-3ad8801d-21d8-4b5d-856f-fd799278a5bb.gif)

[Multi-agent simulation](#multi-agent-simulation)

<td>

![collision_gif]()

[Collision resolution](#collision-resolution)

<td>

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/18750753/162698234-cdbee571-2a84-462e-95e8-d6f8b7cdad09.gif)
[Probabilistic sensors](#probabilistic-sensors)


</tr>
</table>

### Simulator

todo... This simulator builds on a clear separation between the agent and the corresponding model.

Philosophy: Every dynamic entity of the simulation has a model and a corresponding agent

### Collision resolution

todo...

### Commonroad integration

todo...

## Installation

The package is distributed on PyPI. You can simply install it via

```shell
pip install dg-commons
```

to install also the developer tools use `pip install dg-commons["all"]`.

## Pre-commit hook

Install pre-commit with
```shell
pip install pre-commit
pre-commit install
```

Run pre-commit with
```shell
pre-commit run --all-files
```


## Contributors

...

## Publications

The tools contained in this package have contributed to the following publications:

- [**Posetal Games: Efficiency, Existence, and Refinement of Equilibria in Games with Prioritized Metrics**](https://ieeexplore.ieee.org/document/9650727) - _A. Zanardi*, G. Zardini*, S. Srinivasan, S. Bolognani, A. Censi, F. Dörfler, E. Frazzoli_

- ...
