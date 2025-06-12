---
title: Simulating lithium-ion batteries on quantum computers
linkTitle: 9 - Lithium-ion batteries application
weight: 9
---

## Introduction
Quantum computing provides a promising approach to simulate the ground state energy of battery materials, which is essential for improving battery efficiency. In this tutorial, we simulate the ground state energy of Li$_2$FeSiO$_4$, a lithium-ion battery cathode material, using Variational Quantum Eigensolver (VQE) and its variants. The structure of Li$_2$FeSiO$_4$ is shown below.

Conventional unit cell.
![unit cell](/uploads/app9/unit_cell.png)

Crystal Structure (Monoclinic)

![Li2FeSiO4](/uploads/app9/Li2FeSiO4--crystal-toolkit.png)

## Methodology
We employ the following algorithms:

- **VQE** to estimate ground state energy by minimizing the expectation value of a Hamiltonian.
- **Contextual Subspace VQE (CS-VQE)** to reduce qubit requirements [1].
- **CS-ADAPT-VQE**, an adaptive VQE variant that selectively includes relevant excitations [2].
- **Rotoselect** for parameter and generator optimization [3].

## Setting up Li$_2$FeSiO$_4$
Using Tangelo [4], we define the molecule in its second quantized form:

```python
from tangelo.toolboxes.molecular_computation import SecondQuantizedMolecule

mol = SecondQuantizedMolecule(geometry="Li2FeSiO4", q=0, spin=0,
							  basis="sto3g", frozen_orbitals="frozen_core")
```
## Active Space Reduction and Qubit Tapering
To reduce qubits:

- **Active Space Selection:** Select orbitals close to the HOMO-LUMO levels.

```python
frozen_orbitals = get_orbitals_excluding_homo_lumo(mol, homo_minus_n=3, lumo_plus_n=3)
frozen_mol = mol.freeze_mos(frozen_orbitals, inplace=False)
```

- **Qubit Tapering:** Symmetry-based reduction without affecting Hamiltonian accuracy.

```python
from symmer.projection import QubitTapering

# Obtain hamiltonian
qu_op = fermion_to_qubit_mapping(frozen_mol.fermionic_hamiltonian,
								 mapping="JW",
                                 n_spinorbitals=frozen_mol.n_active_sos,
								 n_electrons=frozen_mol.n_active_electrons,
                                 spin=frozen_mol.spin, up_then_down=False)

# Convert qu_op to Symmer's PauliwordOp and store it in H_q
# Checkout full code for details

# Qubit Tapering
IndependentOp.symmetry_generators(H_q, commuting_override=True)
taper_hamiltonian = QubitTapering(H_q)
taper_hamiltonian.stabilizers.rotate_onto_single_qubit_paulis()
hf_array = qml.qchem.hf_state(frozen_mol.n_active_electrons, frozen_mol.n_active_sos)
taper_hamiltonian.stabilizers.update_sector(hf_array)
ham_tap = taper_hamiltonian.taper_it(ref_state=hf_array)
```
## Contextual Subspace Projection
This involves partitioning $H$ into noncontextual and contextual componenents satisfying $H = H_{\text{NC}} + H_{\text{C}}$:

```python
cs_vqe = ContextualSubspace(ham_tap,
							noncontextual_strategy='StabilizeFirst',
							noncontextual_solver='binary_relaxation',
							unitary_partitioning_method='LCU')

# Quantum Corrections
cs_vqe.update_stabilizers(n_qubits = q) # Set the desired number of qubits
H_cs = cs_vqe.project_onto_subspace()

# Hartree-Fock in contextual subspace
hf_cs = cs_vqe.project_state_onto_subspace(taper_hamiltonian.tapered_ref_state)
```

## CS-VQE

We create a hardware-efficient ansatz (HEA) using CUDA-Q [5] and prepare the Hartree-Fock state for CS-VQE:

```python
def ansatz(n_qubits, num_layers):
    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(n_qubits)
    # Prepare Hartree-Fock state
    # Define HEA with rotation and entangling layers
    for l in range(num_layers):
        for q in range(n_qubits):
            kernel.ry(thetas[l * n_qubits + q], qubits[q])
        for q in range(n_qubits - 1):
            kernel.cx(qubits[q], qubits[q + 1])\
	# Final rotation layer
	for q in range(n_qubits):
		kernel.ry(thetas[num_layers * n_qubits + q], qubits[q])
    return kernel

ccsd_energy = -3688.046308050882

# Initialize and optimize the VQE kernel
optimizer = cudaq.optimizers.NelderMead()
energy, params = cudaq.vqe(kernel, ham, optimizer,
						   parameter_count=(num_layers + 1) * n_qubits)
rel_err(ccsd_energy, energy)
```

![CS-VQE](/uploads/app9/cs_vqe.png)

## CS-ADAPT-VQE
CS-ADAPT-VQE adaptively selects excitations based on parameter gradients. We select excitations with the highest gradients and perform VQE with selected excitations:


```python
# Calulate the singles and doubles excitations
singles, doubles = excitations(electrons, n_qubits)
kernel, parameters, qubits = create_kernel(n_qubits)
basis_state(kernel, qubits, hf)
for i, wires in enumerate(doubles):
	double_excitation(kernel, parameters[i], [qubits[q] for q in wires])
init_params = [0.0] * len(doubles)
grads = parameter_shift(kernel, ham, init_params)
doubles_select = []
if len(grads):
	doubles_select = doubles[np.argmax(abs(grads))]

# Perform VQE to obtain the optimized parameters for the selected double excitations.
if len(doubles_select):
	kernel, parameters, qubits = create_kernel(n_qubits)
	basis_state(kernel, qubits, hf)
	double_excitation(
		kernel, parameters[0], [qubits[q] for q in doubles_select]
	)
	optimizer = cudaq.optimizers.NelderMead()
	optimizer.max_iterations = 1000
	optimizer.initial_parameters = np.random.uniform(size=1)
	energy, params_doubles = cudaq.vqe(
		kernel, ham, optimizer, parameter_count=1
	)

# Compute gradients for all single excitations.
kernel, parameters, qubits = create_kernel(n_qubits)
basis_state(kernel, qubits, hf)
if len(doubles_select):
	double_excitation(
		kernel, params_doubles[0], [qubits[q] for q in doubles_select]
	)
for i, wires in enumerate(singles):
	single_excitation(kernel, parameters[i], [qubits[q] for q in wires])
init_params = [0.0] * len(singles)
grads = parameter_shift(kernel, ham, init_params)

# Select the single excitation with maximum absolute gradient value
singles_select = singles[np.argmax(abs(grads))]

# Perform the final VQE optimization with all the selected excitations.
kernel, parameters, qubits = create_kernel(n_qubits)
basis_state(kernel, qubits, hf)
parameter_count = 0
if len(doubles_select):
	double_excitation(
		kernel,
		parameters[parameter_count],
		[qubits[q] for q in doubles_select],
	)
	parameter_count += 1
single_excitation(
	kernel, parameters[parameter_count], [qubits[q] for q in singles_select]
)
parameter_count += 1
optimizer = cudaq.optimizers.NelderMead()
optimizer.max_iterations = 100
optimizer.initial_parameters = np.random.uniform(size=parameter_count)
energy, params = cudaq.vqe(
	kernel, ham, optimizer, parameter_count=parameter_count
)
```

![CS-ADAPT-VQE-circuit](/uploads/app9/adapt_circuit.png)
![CS-ADAPT-VQE](/uploads/app9/adapt.png)

## CS-VQE with Rotoselect Optimization
Rotoselect dynamically selects the optimal rotation gates (X, Y, Z) for each parameter to minimize energy:

```python
def rotoselect_cycle(cost, params, generators, n_qubits, hamiltonian, hf):
    for d in range(len(params)):
        params[d], generators[d] = optimal_theta_and_gen_helper(
            d, params, generators, cost_fn, n_qubits, hamiltonian, hf
        )
    return params, generators

# Perform optimization with Rotoselect cycles:

for i in range(n_steps):
    params, generators = rotoselect_cycle(cost_fn, params, generators, n_qubits, ham, hf)
energy = cost_fn(params, generators, n_qubits, ham, hf)
```
![Rotoselect-orig-circuit](/uploads/app9/rotoselect_orig_circuit.png)
![Rotoselect-circuit](/uploads/app9/rotoselect_circuit.png)
![Rotoselect](/uploads/app9/rotoselect.png)

## Conclusion and Future Directions
This tutorial demonstrated how CS-VQE, ADAPT-VQE, and Rotoselect optimize battery material simulations using quantum computing. Each approach uniquely balances qubit requirements and accuracy, and future work could explore combining these methods with error mitigation techniques to improve results further.

## References
1. William M Kirby, Andrew Tranter, and Peter J Love. "Contextual subspace variational quantum eigensolver". Quantum 5 (2021), p. 456.
2. Jonathan Romero et al. "Strategies for quantum computing molecular energies using the unitary coupled cluster ansatz". Quantum Science and Technology 4.1 (2018), p. 014008.
3. Mateusz Ostaszewski, Edward Grant, and Marcello Benedetti. "Structure optimization for parameterized quantum circuits". Quantum 5 (2021), p. 391.
4. Valentin Senicourt et al. "Tangelo: An Open-source Python Package for End-to-end Chemistry Workflows on Quantum Computers". arXiv:2206.12424(2022). doi: 10.48550/arXiv.2206.12424. eprint: arXiv:2206.12424. url: https://arxiv.org/abs/2206.12424.
5. The CUDA Quantum development team. CUDA Quantum. url: https : //github.com/NVIDIA/cuda-quantum.


### **About the author**



<div align="center">
  <img src="/uploads/app9/gopal.jpeg" alt="Author's Photo" width="150" style="border-radius: 50%; border: 2px solid #1E90FF;">
  <br>
  <strong>Gopal Ramesh Dahale</strong>
  <br>
  <em>Master's student in Quantum Science and Engineering at EPFL</em>
  <br>
  <a href="https://www.linkedin.com/in/gopald27" style="color:#1E90FF;">LinkedIn</a>
</div>


{{< math >}}
{{< /math >}} 