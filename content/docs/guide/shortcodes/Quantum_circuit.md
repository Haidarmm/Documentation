---
title:  Circuit-based quantum programming 
linkTitle: 1- Circuit-based quantum programming 
---

This is the training session for the preliminary understanding about QLM language

![image](/uploads/notebook1/output1.png)

<!--more-->

## Usage

### Creating a bell pair

{{% callout note %}}
Let's us create a Bell pair
{{% /callout %}}

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.lang import Program, H, CNOT, X, S

# Create a Program
qprog = Program()
nbqbits = 2
qbits = qprog.qalloc(nbqbits)

H(qbits[0])
CNOT(qbits[0], qbits[1])

# Export this program into a quantum circuit
circuit = qprog.to_circ()
circuit.display()

# Import a Quantum Processor Unit Factory (the default one)
from qlmaas.qpus import get_default_qpu
# from qlmaas.qpus import get_default_qpu to run on the QAPTIVA appliance
qpu = get_default_qpu()

# Create a job
job = circuit.to_job(nbshots=100)
result = qpu.submit(job)

# Iterate over the final state vector to get all final components
for sample in result:
    print("State %s amplitude %s, %s (%s)" % (sample.state, sample.amplitude, sample.probability, sample.err))
```
**Result**
```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
State |00> amplitude None, 0.48 (0.05021167315686783)
State |11> amplitude None, 0.52 (0.05021167315686783)
```



More advanced features:
- `nbshots` in job
- Measure only certain qubits
- Difference between `sample.amplitude` and `sample.probability`
- Difference between final measure and intermediate measure

### Intermediate measurements

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
qprog = Program()
nbqbits = 2
qbits = qprog.qalloc(nbqbits)
H(qbits[0])
qprog.measure(qbits[0])
CNOT(qbits)
circuit = qprog.to_circ()
circuit.display()


qpu = get_default_qpu()
job = circuit.to_job(nbshots=5, aggregate_data=False)
result = qpu.submit(job)

for sample in result:
    print(sample.state, sample.intermediate_measurements)
```

![image](/uploads/notebook1/output2.png)

**Result**
```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
|11> [IntermediateMeasurement(cbits=[True], gate_pos=1, probability=0.4999999999999999)]
|00> [IntermediateMeasurement(cbits=[False], gate_pos=1, probability=0.4999999999999999)]
|00> [IntermediateMeasurement(cbits=[False], gate_pos=1, probability=0.4999999999999999)]
|11> [IntermediateMeasurement(cbits=[True], gate_pos=1, probability=0.4999999999999999)]
|11> [IntermediateMeasurement(cbits=[True], gate_pos=1, probability=0.4999999999999999)]
```

### Useful tools for gates 

You can check all the gates avalable in the myQLM demo: [Open Available Gates Tutorial](https://mybinder.org/v2/gh/myQLM/myqlm-notebooks/HEAD?filepath=tutorials%2Flang%2Favailable_gates.ipynb). You can also create personal gates
 
#### Quantum routines

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.lang.AQASM import Program, QRoutine, H, CNOT

routine = QRoutine()
routine.apply(H, 0)
routine.apply(CNOT, 0, 1)

prog = Program()
qbits = prog.qalloc(4)
for _ in range(3):
    for bl in range(2):
        prog.apply(routine, qbits[2*bl:2*bl+2])
prog.apply(routine, qbits[0], qbits[2])
circ = prog.to_circ()
circ.display()

```

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
circ = prog.to_circ(box_routines=True)
circ.display()

```

![image](/uploads/notebook1/output3.png)

#### Using typed registers

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
## quantum adder
from qat.lang.AQASM.classarith import add
prog = Program()
reg_a = prog.qalloc(2, QInt)
reg_b = prog.qalloc(2, QInt)

X(reg_a[0])
X(reg_b[1])

# |a> = |10> ("2") and |b>=|01> ("1") 
# expect |a+b>|b> = |11>|01>
prog.apply(add(2, 2), reg_a, reg_b)
circ = prog.to_circ(inline=False)
circ.display()


qpu = get_default_qpu()
result = qpu.submit(circ.to_job())
for sample in result:
    print(sample.state, sample.amplitude)

```

![image](/uploads/notebook1/output4.png)



## Variational computations
framework:
Variational Quantum Eigensolver (VQE) is to find eigenvalues of a Hamiltonian


![image](/uploads/notebook1/stack1.png)

Our task: VQE on the following Ising model:

{{< math >}}
$$
H = \sum_{i=1}^{N} a_i X_i + \sum_{i=1}^{N} \sum_{j=1}^{i-1} J_{ij} Z_i Z_j
$$
{{< /math >}}


... with a "hardware-efficient" ansatz.

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
import numpy as np
from qat.core import Observable, Term

def ising(N):
    np.random.seed(123)  

    terms = []

    # Generate random coefficients for the transverse field term (X)
    a_coefficients = np.random.random(N)
    for i in range(N):
        term = Term(coefficient=a_coefficients[i], pauli_op="X", qbits=[i])
        terms.append(term)

    # Generate random coefficients for the interaction term (ZZ)
    J_coefficients = np.random.random((N, N))
    for i in range(N):
        for j in range(i):
            if i != j:  # avoid duplicate terms
                term = Term(coefficient=J_coefficients[i, j], pauli_op="ZZ", qbits=[i, j])
                terms.append(term)
    ising = Observable(N, pauli_terms=terms, constant_coeff=0.0)
    return ising

```
If we have the number of qubit =  4


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
nqbits = 4
model = ising(nqbits)
print(model)
```

**Result**
```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
0.6964691855978616 * (X|[0]) +
0.28613933495037946 * (X|[1]) +
0.2268514535642031 * (X|[2]) +
0.5513147690828912 * (X|[3]) +
0.48093190148436094 * (ZZ|[1, 0]) +
0.4385722446796244 * (ZZ|[2, 0]) +
0.05967789660956835 * (ZZ|[2, 1]) +
0.18249173045349998 * (ZZ|[3, 0]) +
0.17545175614749253 * (ZZ|[3, 1]) +
0.5315513738418384 * (ZZ|[3, 2])
```



### Applying for Hardware-Efficient Anazt


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.lang.AQASM import Program, QRoutine, RY, CNOT, RX, Z, H, RZ
from qat.core import Observable, Term, Circuit
from qat.lang.AQASM.gates import Gate
import matplotlib as mpl
import numpy as np
from typing import Optional, List
import warnings

def HEA_Linear(
    nqbits: int,
    #theta: List[float],
    n_cycles: int = 1,
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit: #linear entanglement
    """
    This Hardware Efficient Ansatz has the reference from "Nonia Vaquero Sabater et al. Simulating molecules 
    with variational quantum eigensolvers. 2022" -Figure 6 -Link 
    "https://uvadoc.uva.es/bitstream/handle/10324/57885/TFM-G1748.pdf?sequence=1"

    Args:
        nqbits (int): Number of qubits of the circuit.
        n_cycles (int): Number of layers.
        rotation_gates (List[Gate]): Parametrized rotation gates to include around the entangling gate. Defaults to :math:`RY`. Must
            be of arity 1.
        entangling_gate (Gate): The 2-qubit entangler. Must be of arity 2. Defaults to :math:`CNOT`.
    """

    if rotation_gates is None:
        rotation_gates = [RZ]

    n_rotations = len(rotation_gates)

    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]   
    
    ind_theta = 0
    
    for i in range(nqbits):

        for rot in rotation_gates:

            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1
    
    for k in range(n_cycles):

        for i in range(nqbits - 1):
            prog.apply(CNOT, reg[i], reg[i+1])
            
        for i in range(nqbits):
            for rot in rotation_gates:
                            
                prog.apply(rot(theta[ind_theta]), reg[i])
                ind_theta += 1

    return prog.to_circ()

```

*Display*

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
n_layers = 4
circ_Linear = HEA_Linear(nqbits, n_layers, [RX,RZ], CNOT)
circ_Linear.display()
```

![image](/uploads/notebook1/slack2.png)

### Variational Quantum Eigensolver 

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.plugins import ScipyMinimizePlugin
qpu = get_default_qpu()
optimizer_scipy = ScipyMinimizePlugin(method="BFGS", # Methods
                                      tol=1e-6,
                                      options={"maxiter": 200},
                                      x0=np.random.rand(n_layers*nqbits))
stack1 = optimizer_scipy | qpu

import numpy as np
import matplotlib.pyplot as plt

# construct a (variational) job with the variational circuit and the observable
job = circ_Linear.to_job(observable=model)
# we submit the job and print the optimized variational energy (the exact GS energy is -3)
result1 = stack1.submit(job)
print(f"Minimum VQE energy ={result1.value}")
plt.plot(eval(result1.meta_data['optimization_trace']))
plt.xlabel("VQE iterations")
plt.ylabel("energy")
plt.grid()
plt.savefig("newfigure.pdf")

```

![image](/uploads/notebook1/slack3.png)


### VQE - Unitary Coupled Cluster

This part is adapted from myQLM documentation demo

The **Variational Quantum Eigensolver** method solves the following minimization problem:
{{< math >}}
$$
E = \min_{\vec{\theta}}\; \langle \psi(\vec{\theta}) \,|\, \hat{H} \,|\, \psi(\vec{\theta}) \rangle
$$
{{< /math >}}

Here, we use a **Unitary Coupled Cluster** trial state, of the form:
{{< math >}}
$$
|\psi(\vec{\theta})\rangle = e^{\hat{T}(\vec{\theta}) - \hat{T}^\dagger(\vec{\theta})} |0\rangle
$$
{{< /math >}}

where $\hat{T}(\theta)$ is the *cluster operator*:
{{< math >}}
$$
\hat{T}(\vec{\theta}) = \hat{T}_1(\vec{\theta}) + \hat{T}_2(\vec{\theta}) + \cdots
$$
{{< /math >}}

where
{{< math >}}
$$
\hat{T}_1 = \sum_{a\in U}\sum_{i \in O} \theta_a^i\, \hat{a}_a^\dagger \hat{a}_i \qquad
\hat{T}_2 = \sum_{a>b\in U}\sum_{i>j\in O} \theta_{a, b}^{i, j}\, \hat{a}^\dagger_a \hat{a}^\dagger_b \hat{a}_i \hat{a}_j \qquad
\cdots
$$
{{< /math >}}

$O$ is the set of occupied orbitals and $U$, the set of unoccupied ones.



```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation

geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.7414))]
basis = "sto-3g"
spin = 0
charge = 0

(
    rdm1,
    orbital_energies,
    nuclear_repulsion,
    n_electrons,
    one_body_integrals,
    two_body_integrals,
    info,
) = perform_pyscf_computation(geometry=geometry, basis=basis, spin=spin, charge=charge, run_fci=True)

print(
    f" HF energy :  {info['HF']}\n",
    f"MP2 energy : {info['MP2']}\n",
    f"FCI energy : {info['FCI']}\n",
)
print(f"Number of qubits before active space selection = {rdm1.shape[0] * 2}")

nqbits = rdm1.shape[0] * 2
print("Number of qubits = ", nqbits)

```

If we make the plot amongst the HF, MP2 with FCI is the reference we obtain 



![image](/uploads/notebook1/stack4.png)

In these such cases, methods like Hartree-Fock (HF) and Møller-Plesset perturbation theory (MP2) become less accurate compared to Full Configuration Interaction (FCI).However in this case for Hydrogen this difference is barely visible, because the HF energy is totally on the same path as FCI from the beginning till the minimum energy point then the curve begins to experience the bond dissociation when the two hydrogen molecules move far each other

Following to show the molecular Hamiltonian

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.fermion.chemistry import MolecularHamiltonian, MoleculeInfo

# Define the molecular hamiltonian
mol_h = MolecularHamiltonian(one_body_integrals, two_body_integrals, nuclear_repulsion)

print(mol_h)

```

 MolecularHamiltonian(
 - constant_coeff : 0.7137539936876182
 - integrals shape
    * one_body_integrals : (2, 2)
    * two_body_integrals : (2, 2, 2, 2)
)


### Computation of cluster operators $T$ and good guess $\vec{\theta}_0$

We now construct the cluster operators (``cluster_ops``) defined in the introduction part as $\hat{T}(\vec{\theta})$, as well as a good starting parameter $\vec{\theta}$ (based on the second order Møller-Plesset perturbation theory).


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.fermion.chemistry.ucc import guess_init_params, get_hf_ket, get_cluster_ops

# Computation of the initial parameters
theta_init = guess_init_params(
    mol_h.two_body_integrals,
    n_electrons,
    orbital_energies,
)

print(f"List of initial parameters : {theta_init}")

# Define the initial Hartree-Fock state
ket_hf_init = get_hf_ket(n_electrons, nqbits=nqbits)

# Compute the cluster operators
cluster_ops = get_cluster_ops(n_electrons, nqbits=nqbits)


from qat.fermion.transforms import transform_to_jw_basis  # , transform_to_bk_basis, transform_to_parity_basis
from qat.fermion.transforms import recode_integer, get_jw_code  # , get_bk_code, get_parity_code

# Compute the ElectronicStructureHamiltonian
H = mol_h.get_electronic_hamiltonian()

# Transform the ElectronicStructureHamiltonian into a spin Hamiltonian
H_sp = transform_to_jw_basis(H)

# Express the cluster operator in spin terms
cluster_ops_sp = [transform_to_jw_basis(t_o) for t_o in cluster_ops]

# Encoding the initial state to new encoding
hf_init_sp = recode_integer(ket_hf_init, get_jw_code(H_sp.nbqbits))

print(H_sp)

```

![image](/uploads/notebook1/slack9.png)

```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
(-0.09886396933545824+0j) * I^4 +
(0.16862219158920938+0j) * (ZZ|[0, 1]) +
(0.12054482205301799+0j) * (ZZ|[0, 2]) +
(0.165867024105892+0j) * (ZZ|[1, 2]) +
(0.165867024105892+0j) * (ZZ|[0, 3]) +
(0.17119774903432972+0j) * (Z|[0]) +
(0.12054482205301799+0j) * (ZZ|[1, 3]) +
(0.17119774903432972+0j) * (Z|[1]) +
(0.04532220205287398+0j) * (XYYX|[0, 1, 2, 3]) +
(-0.04532220205287398+0j) * (XXYY|[0, 1, 2, 3]) +
(-0.04532220205287398+0j) * (YYXX|[0, 1, 2, 3]) +
(0.04532220205287398+0j) * (YXXY|[0, 1, 2, 3]) +
(0.17434844185575668+0j) * (ZZ|[2, 3]) +
(-0.22278593040418448+0j) * (Z|[2]) +
(-0.22278593040418448+0j) * (Z|[3])

```

Applying the trotterization method 

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat.lang.AQASM import Program, X
from qat.fermion.trotterisation import make_trotterisation_routine
prog = construct_ucc_ansatz(cluster_ops_sp, hf_init_sp, n_steps=1)

prog = Program()
reg = prog.qalloc(H_sp.nbqbits)

# Initialize the Hartree-Fock state into the Program
for j, char in enumerate(format(hf_init_sp, "0" + str(H_sp.nbqbits) + "b")):
    if char == "1":
        prog.apply(X, reg[j])

# Define the parameters to optimize
theta_list = [prog.new_var(float, "\\theta_{%s}" % i) for i in range(len(cluster_ops))]

# Define the parameterized Hamiltonian
cluster_op = sum([theta * T for theta, T in zip(theta_list, cluster_ops_sp)])

# Trotterize the Hamiltonian (with 1 trotter step)
qrout = make_trotterisation_routine(cluster_op, n_trotter_steps=1, final_time=1)

prog.apply(qrout, reg)
circ = prog.to_circ()

prog = construct_ucc_ansatz(cluster_ops_sp, hf_init_sp, n_steps=1)
circ = prog.to_circ()
circ.display()


```

For the molecule $H_2$, it has double qubit excitation and by using the UCC method to simulate on this molecule, I obtain the the circuit construction  as bellow  with the interprobility with Qiskit and this construction is based on the staire-case algorithms 

![image](/uploads/notebook1/stack5.png)


### Using the Gradient Based Optimizer to solve the VQE 

The graphic is show: 

![image](/uploads/notebook1/stack6.png)

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
job = circ.to_job(observable=H_sp, nbshots=0)

from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin

optimizer_scipy = ScipyMinimizePlugin(method="BFGS", tol=1e-3, options={"maxiter": 1000}, x0=theta_init)
qpu = optimizer_scipy | get_default_qpu()
result = qpu.submit(job)

print("Minimum energy =", result.value)

```

Minimum energy = -1.1372692847285149 \\

Make the plot 


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot(eval(result.meta_data["optimization_trace"]), lw=3)
plt.plot(
    [info["FCI"] for _ in enumerate(eval(result.meta_data["optimization_trace"]))],
    "--k",
    label="FCI",
)

plt.xlabel("Steps")
plt.ylabel("Energy")
plt.grid()
```

![image](/uploads/notebook1/stack7.png)

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