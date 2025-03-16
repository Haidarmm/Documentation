---
title: Observables 
linkTitle: 2a-Observables  
---

This is the training session for the preliminary understanding about QLM language for mostly in Quantum Chemistry 



<!--more-->

### Class: Observable

This can be any hermitian or non-Hermitian operators
#### Stores a list of Pauli chains + their coefficients

{{% callout note %}}

{{< math >}}
$$
H\  =\  \sum_{k}^{} c_{k}P_{k}\  \text{with} \  P_{k}=I,X,Y,Z,XX,XZ,\  XYZ,...
$$
{{< /math >}}

{{% /callout %}}



### Class: Term
This class contains a Pauli chain and its coefficient






```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.core import Term, Observable

# Define a term with 4 qubits and a Pauli string
term = Term(1.5,  # coefficient
            "XXYZ",  # Pauli chain for 4 qubits
            [0, 1, 2, 3])  # qubits on which each Pauli acts

# Define an observable acting on 4 qubits with multiple terms
obs = Observable(4,  # total number of qubits
                 pauli_terms=[Term(2.5, "ZZ", [0, 1]),
                              Term(1.0, "YX", [2, 3]),
                              Term(0.8, "XYZ", [0, 2, 3])])

# Display the term and the observable
print(term)
print(obs)

```
<u>Result</u> with the Hamiltonian $$H\  =\  2.5\times Z_{0}Z_{1}+1\times Y_{2}X_{3}+0.8\times X_{0}Y_{2}Z_{3}$$

<div style="border:1px solid #ccc; padding: 10px">
Term(_coeff=TNumber(is_abstract=False, type=1, int_p=None, double_p=1.5, string_p=None, matrix_p=None, serialized_p=None, complex_p=None), op='XXYZ', qbits=[0, 1, 2, 3], _do_validity_check=True)<br/>
2.5 * (ZZ|[0, 1]) +<br/>
1.0 * (YX|[2, 3]) +<br/>
0.8 * (XYZ|[0, 2, 3])
</div>

#### Transform to matrix representation to see that the Dense matrix is exponential in number of qubits
 We can show its matrix representation for the Observable 
 ```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
print(obs.to_matrix()) # a scipy sparse array
print(obs.to_matrix(sparse=False)) # a numpy array
```
Dense matrix is exponential in number of qubits!


### Class: Fermionic System

In the context of quantum chemistry, the hamiltonian is expressed in the fermionic second-quantized form witht the '$c$' (small) for annihilation operator
'$c^{\dag }$' or creation operator

{{< math >}}
$$
    H=\sum^{}_{p,q} h_{pq}c^{\dag }_{p}c_{q}+\frac{1}{2} \sum^{}_{p,q,r,s} h_{pqrs}c^{\dag }_{p}c^{\dag }_{q}c_{r}c_{s}

$$
{{< /math >}}

where $h_{pq}$ and $h_{pqrs}$ are the one and two electron integrals. The one-electron integrals are obtained from the kinetic energy and electron-nuclei interactions while the two-electron integrals are obtained from the electron-electron interactions, defined with the help of some basis functions.

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.fermion import ElectronicStructureHamiltonian
import numpy as np
h_pq = np. array([[1., 2.],
                    [2., 0.11]])
h_pqrs = np.zeros ((2, 2, 2, 2))
h_pqrs[0, 1, 0, 1] = -8.0
ham = ElectronicStructureHamiltonian(h_pq, h_pqrs)
print (ham)

```
<u>Result</u> with the Hamiltonian $$H = c_0^\dagger c_0 + 2c_0^\dagger c_1 + 2c_1^\dagger c_0 + 4c_0^\dagger c_0 c_1^\dagger c_1
$$

In the Fock space representation, the operator can be transformed to the qubit space by three main common mapping methods: Jordan-Wigner , Parity and Bravyi-Kitaev found more details in this [reference](https://arxiv.org/pdf/1208.5986).  . It is currently unknown which encoding method is the most noise-resilient - i.e.
best for NISQ experiments. Numerical studies aiming to compare the Jordan-Wigner and Bravyi-Kitaev mappings, has found that the BK transformations was at less as efficient as the JW one, in finding the ground states of molecular systems. Most commonly, we directly use the Jordan Wigner transformation as it is the most straightforward qubit encoding; thus it is a one to one correspondence between Slater determinants and computational basis qubit states:

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
ham_spin = ham.to_spin("jordan-wigner")
print (ham_spin)

```
<div style="border:1px solid #ccc; padding: 10px">
(1.5550000000000002+0j) * I^2 + <br/>
(1+0j) * (XX|[0, 1]) + <br/>
(1+0j) * (YY|[0, 1]) + <br/>
(1+0j) * (ZZ|[0, 1]) + <br/>
(-1.5+0j) * (Z|[0])  + <br/>
(-1.055+0j) * (Z|[1])
</div>

Now that we have defined the Hamiltonian in spin representatin, we can already start playing with it. We can give a try for running exact diagonalization. Firstly we can  convert our Hamiltonian operator into a sparse matrix

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
model_matrix_sp = ham_spin.get_matrix(sparse=True)

```

Since this is just a regular scipy sparse matrix, we can just use any sparse diagonalization routine in there to find the eigenstates. 

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from scipy.sparse.linalg import eigsh
eigval, eigvec = eigsh(model_matrix_sp, k = 2)
print("eigenvalues with scipy sparse:", eigval)
E_gs = eigval[0]
```
eigenvalues with scipy sparse:  [5.11&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2.60390825]


### **Reference**

<a href="https://myqlm.github.io/" style="color:#1E90FF;">
Made by Eviden "myQLM – Quantum Python Package"
</a>


### **About the author**



<div align="center">
  <img src="/imagez/huybinh.png" alt="Author's Photo" width="150" style="border-radius: 50%; border: 2px solid #1E90FF;">
  <br>
  <strong>Huy Binh TRAN</strong>
  <br>
  <em>Master 2 Quantum Devices at Institute Paris Polytechnic, France</em>
  <br>
  <a href="https://www.linkedin.com/in/huybinhtran/" style="color:#1E90FF;">LinkedIn</a>
</div>