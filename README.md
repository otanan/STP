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
  </p>
</div>


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

**3-state self-assembly rate matrix** (time-dependent)
```python
import numpy as np
import stp

alpha = lambda t: np.cos(t) + 2
W = stp.self_assembly_rate_matrix(alpha)
print(W(0))  # [[-2.  3.  9.] ...]
```

**Stationary distribution**
```python
R = stp.rand_transition_matrix(3)
p_star = stp.get_stationary_distribution(R, discrete=True)
```

**Shannon entropy**
```python
p = stp.rand_p(3)
H = stp.info.entropy(p)
```

**KMC path sampling**
```python
W = stp.self_assembly_rate_matrix(alpha=1.5)
p = stp.rand_p(3)
paths = stp.KMC(W, p, num_paths=100, path_length=10, seed=42)
```

_For more examples, please refer to the [Documentation]._

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

Refer to the [Notion Roadmap] for the state of the project.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact
Created by [Jonathan Delgado](https://jdelgado.net/).


<p align="right"><a href="#readme-top">Back to top</a></p>

[Notion Roadmap]: https://otanan.notion.site/01f791e958c04bfeaf33cd066f3971c1?v=2ca4c56ad7404d59b2bd751e3fcaaee6
[Documentation]: https://stochastic-thermodynamics-in-python.readthedocs.io/en/latest/