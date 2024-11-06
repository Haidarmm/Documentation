---
title: MyQLM-Fermion
---



<!--more-->

A submodule of myQLM-fermion is specifically devoted to quantum chemistry. It provides tools for selecting active spaces
(based on natural-orbital occupation numbers), generating cluster operators (and thus, via the aforementioned Trotterization
tools, UCC-type ansätze), and initial guesses for their variational parameters. The architecture of QLM and myQLM-fermion
allows for experts in a given field to construct their own advanced modules with the QLM building blocks.
\
The key building block of quantum chemistry computations is the Hamiltonian. On QLM, it is described by an object
ElectronicStructureHamiltonian

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
 from qat . fermion import ElectronicStructureHamiltonian
 hamiltonian = ElectronicStructureHamiltonian (h , g)

```

where h and g are the tensors ℎ_pq and ℎ_pqrs. Such an object also describes cluster operators 


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat . fermion import get_cluster_ops
cluster_ops = get_cluster_ops ( n_electrons , nqbits = nqbits )

```

creates the list containing the sets of single excitations and double excitations wnich can be readily converted to a spin (or qubit) representation using various fermion-spin transforms:

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}

 # Jordan - Wigner
 from qat . fermion . transforms import transform_to_jw_basis
 hamiltonian_jw = transform_to_jw_basis ( hamiltonian )
 cluster_ops_jw = [ transform_to_jw_basis ( t_o ) for t_o in cluster_ops ]

 # Bravyi - Kitaev
 from qat . fermion . transforms import transform_to_bk_basis
 hamiltonian_bk = transform_to_bk_basis ( hamiltonian )
 cluster_ops_bk = [ transform_to_bk_basis ( t_o ) for t_o in cluster_ops ]


```
 With these qubit operators, one can then easily contruct a imple UCCSD ansatz via trotterization  of the exponential of the parametric cluster operator defined as cluster_ops_jw


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from qat . lang . AQASM import Program , X
from qat . fermion . trotterisation import make_trotterisation_routine

prog = Program ()
reg = prog . qalloc ( nqbits )
# Create Hartree - Fock state ( assuming JW representation )

for qb in range ( n_electrons ) :
prog . apply (X , reg [ qb ])

 # Define the full cluster operator with its parameters
theta_list = [ prog . new_var (float , "\\ theta_ {%s}" % i) for i in range (len ( cluster_ops_jw ) )]
cluster_op = sum ([ theta * T for theta , T in zip( theta_list , cluster_ops_jw ) ])

# Trotterize the Hamiltonian ( with 1 trotter step )
qrout = make_trotterisation_routine ( cluster_op , n_trotter_steps =1 , final_time =1)
prog . apply ( qrout , reg )
circ = prog . to_circ ()

```

The circuit we constructed, circ, is a variational circuit that creates a variational wavefunction.  Its parameters can be
optimized to minimize the variational energy which can be done by a simple VQE loop with the UCC method 

![image](/uploads/slack4.png)

1. A simulation starts by constructing a fermionic Hamiltonian with particularly straightforward initialization as a classical mean-field state; most often as a HF product state {{< math >}}
   $$
   \ket{\psi_{HF}}
   $$
   {{< /math >}}. This is required as the reference preparation for the UCC-chemically-inspired ansatz.

2. The fermionic Hamiltonian is mapped into a qubit Hamiltonian, represented as a sum of Pauli strings:
   {{< math >}}
   $$
   H = \sum_j \alpha_j \prod_i \sigma_i^j,
   $$
   {{< /math >}}
   where {{< math >}}
   $$
   \sigma_i^j \in \{ \text{I}, X, Y, Z \}
   $$
   {{< /math >}}.

3. A quantum circuit implementing the unitary operator {{< math >}}
   $$
   U(\vec{{\bm{\theta}} })
   $$
   {{< /math >}} is applied to {{< math >}}
   $$
   \ket{\psi_{HF}}
   $$
   {{< /math >}}, mapping the initial state to a parameterized "Ansatz" state:
   {{< math >}}
   $$
   |\psi(\vec{{\bm{\theta}}}) \rangle = U(\vec{{\bm{\theta}}}) |\psi_{HF} \rangle.
   $$
   {{< /math >}}
   Thus, the trial state is prepared on a quantum computer as a quantum circuit consisting of parameterized gates.

4. One measures the expectation value of the energy:
   {{< math >}}
   $$
   \langle H \rangle = \langle \psi_{HF}(\vec{{\bm{\theta}}}_0) | H | \psi_{HF}(\vec{{\bm{\theta}}}_0) \rangle.
   $$
   {{< /math >}}
   At iteration {{< math >}}
   $$
   k
   $$
   {{< /math >}}, the energy of the Hamiltonian is computed by measuring every Hamiltonian term:
   {{< math >}}
   $$
   \langle \psi(\vec{{\bm{\theta}}_k}) | P_j | \psi(\vec{{\bm{\theta}}_k}) \rangle
   $$
   {{< /math >}}
   on a quantum computer and adding them on a classical computer.

5. The energy {{< math >}}
   $$
   E(\vec{{\bm{\theta}}_k})
   $$
   {{< /math >}} is fed into the classical algorithm that updates parameters for the next step of optimization {{< math >}}
   $$
   \vec{{\bm{\theta}}}_{k+1}
   $$
   {{< /math >}} according to the chosen optimization algorithm.





```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
# create a quantum job containing the variational circuit and the Hamiltonian
job = circ . to_job ( observable = hamiltonian_jw , nbshots =0)
# import a plugin to perform the optimization
from qat . plugins import scipyMinimizePlugin
optimizer_scipy = scipyMinimizePlugin ( method =" COBYLA ", tol =1e -3 , options ={" maxiter ": 1000} , x0 = theta_init )

# import a QPU to execute the quantum circuit
from qat . qpus import get_default_qpu

# define the quantum stack
stack = optimizer_scipy | get_default_qpu ()
# submit the job and read the result
result = stack . submit ( job )
print (" Minimum energy =", result . value )
```

### Active space selection 

In the context of chemical properAmiensties regardless of small/large molecular systems, the lowest orbitals are assumed to be frozen and fully occupied, contributing to the core energy. The highest orbitals, on the other hand, are considered non-active and empty. Mathematically, the Quantum Learning Machine (QLM) \cite{haidar2023open} defines two subspaces: active orbitals ($\mathcal{A}$) and occupied orbitals ($\mathcal{O}$). By calculating the eigenvalues of the molecule's reduced density matrix, the Natural Orbital Occupation Numbers $n_i$ for each molecular orbital $i$ are obtained. We use the QLM library, which contains a function that enables us to determine the population of the two subspaces, through upper and lower thresholds $\epsilon_1$ and $\epsilon_2$ to select the relevant orbitals. In fact, the choice of $\epsilon_1$ and $\epsilon_2$ is the choice of the active orbitals.

$$
\mathcal{A} = \{i \ | \ n_i \in [\epsilon_2,2-\epsilon_1] \} \cup \{i \ | \ n_i \geq 2 - \epsilon_1, \  2(i+1) \geq N_{elec}  \}
$$

$$
\mathcal{O} = \{i \ | \ n_i \geq 2 - \epsilon_1, \  2(i+1) < N_{elec}  \}.
$$

The QLM function can then evaluate the one-body term and the core energy.

$$
\forall p,q \in \mathcal{A}
$$

$$
h_{pq} \text{-----} h_{pq} + \sum_{i \in \mathcal{O}} 2h_{ipqi} - h_{ipiq},
$$

$$
E_{\rm core} \text{-----} E_{\rm core} + \sum_{i \in \mathcal{O}} h_{ii} + \sum_{i,j \in \mathcal{O}} 2h_{ijji} - h_{ijij}.
$$

To apply CAS on UCCSD ansatz, it is required to determine the  total number of electrons $n_{e}$  and the subspace of spin orbitals $\mathcal{O}$ in a molecule. Then we can determine the number of active electrons $n_{e_{act}} = n_{e} - 2|\mathcal{O}|$ as well as the number of electrons that are distributed over the active orbitals. Therefore $\mathcal{A}$ is divided into two sub-spaces : $\mathcal{I'}$ which contains the unoccupied active orbitals and $\mathcal{O'}$ containing the occupied active orbitals, and the anti-Hermitian operator is created:

$\forall i,j,p,q \in \mathcal{I'}^{2}\times\mathcal{O'}^{2}$

$$
 T^{(\mathrm{CAS})}_{\mathrm{UCCSD}} =  \sum_{i\in \mathcal{O^{\prime }} , p\in \mathcal{I^{\prime }}} \theta^{p}_{i} (\hat{a}^{\dagger }_{i} \hat{a}_{p} -\hat{a}^{\dagger }_{p} \hat{a}_{i}) + + \sum_{i,j\in \mathcal{O^{\prime }}, p,q\in \mathcal{I^{\prime }}} \theta^{pq}_{ij} (\hat{a}^{\dagger }_{i} \hat{a}^{\dagger }_{j} \hat{a}_{p} \hat{a}_{q} - \hat{a}^{\dagger }_{p} \hat{a}^{\dagger }_{q} \hat{a}_{i} \hat{a}_{j})
$$


![image](/uploads/sstack2.png)

*Visualization of spatial orbitals in LiH molecule uisng sto-3g basis-set: 2 HUMOs and 4 LUMOs.*


In this method we have the figure of the plot with respect to certain number of qubit with respect to the orbital energy 

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
import matplotlib.pyplot as plt

# Sample data (you can replace this with your actual data)
noons = list(reversed(noons))

# Define the threshold couples and their corresponding labels
threshold_couples = [
    (0.02, 0.004, "4 qubit / second"),
    (0.02e-6, 0.002, "6 qubit"),
    (0.02, 0.001, "10 qubit"),
    (0.02, 0.001e-1, "12 qubit"),
]

# Initialize lists to store optimization results
results = []

# Iterate through different threshold couples
for threshold_1, threshold_2, label in threshold_couples:
    mol_h_active, active_indices, occupied_indices = mol_h_new_basis.select_active_space(
        noons=noons, n_electrons=n_elec, threshold_1=threshold_1, threshold_2=threshold_2
    )

    # Rest of your code for getting theta_list and performing the optimization...

    # Store the result
    result = qpu.submit(job)
    results.append((result, label))

# Create the plot for UCCSD-VQE with multiple threshold couples
plt.figure(figsize=(10, 6))
for result, label in results:
    plt.plot(eval(result.meta_data["optimization_trace"]), label=f"UCCSD-VQE ({label})", lw=3)

plt.plot(
    [info["FCI"] for _ in range(len(eval(result.meta_data["optimization_trace"])))],
    "--k",
    label="FCI",
)
plt.legend(loc="best")
plt.xlabel("Steps")
plt.ylabel("Energy")
plt.grid()
plt.title("UCCSD-VQE Optimization with Different Thresholds")
plt.show()
```

![image](/uploads/sstack3.png)

If we make the plot with the threshold 

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
FCI_energy = info['FCI']

markers = ['o', 'x', '^', 's']
colors = ['firebrick', 'green', 'black', 'red']

plt.figure(figsize=(10, 6))
for (result, label), marker, color in zip(results, markers, colors):
    optimization_trace = eval(result.meta_data["optimization_trace"])
    errors = [abs(energy - FCI_energy) for energy in optimization_trace]
    steps = list(range(len(errors)))
    plt.errorbar(steps, errors, fmt=marker, color=color, elinewidth=1, capsize=2, label=f"{label}")

# Plot the chemical accuracy line
plt.axhline(y=1.593e-3, color='darkblue', linestyle='-', label="Chemical Accuracy")

plt.yscale('log')
plt.legend(loc="best")
plt.xlabel("Steps")
plt.ylabel("Energy Error (Log Scale)")
plt.grid()
plt.title("UCCSD-VQE Optimization Error with Different Thresholds")
plt.show()
```
![image](/uploads/sstack4.png)

$$
U\left(x;\bm{\theta}\right)=U_N\left(\bm{\theta}_N\right)U_{N-1}\left(\bm{\theta}_{N-1}\right)\cdots U_i\left(\bm{\theta}_i\right) \cdots U_1\left(\bm{\theta}_1\right)U_0\left(x\right)
$$