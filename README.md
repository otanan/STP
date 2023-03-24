<!-- Filename:      README.md -->
<!-- Author:        Jonathan Delgado -->
<!-- Description:   GitHub README -->

<!-- Header -->
<h2 align="center">Stochastic Thermodynamics in Python (STP)</h2>
  <p align="center">
    Python library to construct random quantities and track their information-theoretic properties. These objects include continuous time rate matrices, discrete time transition matrices, and matrices representing 3-state self assembly models.
    <br />
    <br />
    Status: <em>in progress</em>
    <!-- Documentation link -->
    ·<a href="https://stochastic-thermodynamics-in-python.readthedocs.io/en/latest/"><strong>
        Documentation
    </strong></a>
    <!-- Notion Roadmap link -->
    ·<a href="https://otanan.notion.site/01f791e958c04bfeaf33cd066f3971c1?v=2ca4c56ad7404d59b2bd751e3fcaaee6"><strong>
        Notion Roadmap »
    </strong></a>
  </p>
</div>


<!-- Project Demo -->
![Screenshot](https://jdelgado.net/images/stochastic-thermodynamics/typical-set.webp "Self Assembly Typical Set")
<!-- ![Screenshot](https://jdelgado.net/images/stochastic-thermodynamics/ts_hist_animation.gif "Self-Assembly State Histogram") -->


<!-- ## Table of contents
* [Contact](#contact)
* [Acknowledgments](#acknowledgments) -->


## Installation

This package is pip-installable.

1. Install via pip.
   ```sh
   python3 -m pip install stp
   ```
1. Import the package
   ```python
   import stp
   ```


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Generating a 3-state time-dependent self-assembly model with this package is as simple as writing
```python
import numpy as np
import stp
# Dimensionless, time-dependent parameter for self assembly matrix
alpha = lambda t : np.cos(t) + 2
W = stp.self_assembly_rate_matrix(alpha)

# The initial matrix
print(W(0))
# [[-2.  3.  9.]
# [ 1. -3.  0.]
# [ 1.  0. -9.]]

# A later matrix
print(W(1))
# [[-2.          2.54030231  6.45313581]
# [ 1.         -2.54030231  0.        ]
# [ 1.          0.         -6.45313581]]
```


_For more examples, please refer to the [Documentation].

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

Refer to the [Notion Roadmap] for the state of the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact
Created by [Jonathan Delgado](https://jdelgado.net/).


<p align="right"><a href="#readme-top">Back to top</a></p>

[Notion Roadmap]: https://otanan.notion.site/01f791e958c04bfeaf33cd066f3971c1?v=2ca4c56ad7404d59b2bd751e3fcaaee6
[Documentation]: https://stochastic-thermodynamics-in-python.readthedocs.io/en/latest/