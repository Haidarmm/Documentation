---
title: Package Structure
weight: 1
---

## Folder Structure

OpenVQE consists of **two main modules that are inside the "openvqe" folder:**:

- `UCC Family/`: This module includes different classes and functions to generate the fermionic cluster operators (fermionic pool) and the qubit pools, and to get the VQE optimized energies in the cases of active and non-active orbital selections.
- `adapt/` includes two sub-modules:
  - `Fermionic-ADAPT/`: containing functions performing the fermionic-ADAPT-VQE algorithmic steps in the active and non-active space selections;
  - `Qubit-ADAPT/`: containing functions that perform the qubit-ADAPT-VQE algorithmic steps calculation in the active and non-active space orbital selections.

## SubFolder Structure

- `common_files/`: stores all the internal functions needed to be imported for executing the two modules.
- `notebooks`: allows the user to run and test the above two modules: `UCC Family/` and `adapt/`.

![image](/uploads/output.jpg)


## Import folder / subfolder

Page `Check the import`:



```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from openvqe.ucc import ...
from openvqe.common_files.qubit_pool import QubitPool

```

{{< math >}}
$$
\gamma_{n} = \frac{ \left | \left (\mathbf x_{n} - \mathbf x_{n-1} \right )^T \left [\nabla F (\mathbf x_{n}) - \nabla F (\mathbf x_{n-1}) \right ] \right |}{\left \|\nabla F(\mathbf{x}_{n}) - \nabla F(\mathbf{x}_{n-1}) \right \|^2}
$$
{{< /math >}}


