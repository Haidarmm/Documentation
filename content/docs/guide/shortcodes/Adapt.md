---
title: Adapt-VQE
linkTitle: 4- Adapt-VQE
---



<!--more-->

In 2019, ADAPT-VQE was introduced in this [paper](https://www.nature.com/articles/s41467-019-10988-2) with the purpose of creating a more
accurate, compact, and problem-customized ansatz for VQE. This version will hereby be denoted fermionic-ADAPT-VQE to distinguish it against a more recent version that will be covered shortly. The idea behind the proposal is to let the molecule in study ‘choose’ its own state preparation circuit, by creating the ansatz in a strongly system-adapted manner.

![image](/uploads/stack9.png)

The ADAPT-VQE algorithm constructs the molecular system's wavefunction dynamically and can in principle avoid redundant terms. It is grown iteratively in the form of a disentangled UCC ansatz  as  given in the below equation:

$$\prod_{k=1}^{\infty} \prod_{pq} \left( e^{\theta_{pq} (k)\hat{A}_{p,q}}\prod_{rs} e^{\theta_{pqrs} (k)\hat{A}_{pq,rs}} \right) |\psi_{\mathrm{HF}} \rangle$$

which is the in the form of a long product of one-body $\hat{A}_{p,q}$ and two-body $\hat{A}_{pq,rs}$ operators generated from the pool of excitations, where each of the variational parameters $\left\{ \theta_{pq} ,\theta_{pqrs} \right\}$ 

At each step, an operator or a few operators are chosen from a pool: the operator(s) contributing to the largest energy deviations is (are) chosen and added gradually to the ansatz until the exact FCI wavefunction has been reached is associated to an operator.



## Algorithm Steps

1. **Initialize Circuit**: Start with the identity circuit $U^{(0)}(\theta) = I$, where the state is initialized to the Hartree-Fock state $|\Psi_{HF}\rangle$.

2. **Gradient Measurement**: For the current ansatz state $|\Psi^{(k-1)}\rangle$, compute the energy gradient for each operator $A_m$ in the operator pool using the formula:
   $$
   \frac{\partial E^{(k-1)}}{\partial \theta_m} = \langle \Psi(\theta_{k-1}) | [H, A_m] | \Psi(\theta_{k-1}) \rangle.
   $$

3. **Check Gradient Norm**: Evaluate the norm of the gradient vector $||g^{(k-1)}||$. If it is below the threshold $\epsilon$, the algorithm stops. If not, proceed to the next step.

4. **Select Operator with Maximum Gradient**: The operator with the largest gradient is chosen, and its corresponding variational parameter $\theta_k$ is added to the ansatz.

5. **Update Ansatz**: Perform a Variational Quantum Eigensolver (VQE) experiment to re-optimize all the parameters $\{\theta_k, \theta_{k-1}, \dots, \theta_1\}$ in the ansatz.

6. **Repeat**: Return to step 2 and repeat the process until convergence.



In the following numerical simulation, we used UCCGSD excitation for $H_2$ molecule in 6-31g basis set thus give us 8 qubits; we can run the following script to see the result ***(The result will not be shown all in here due to its long scrolled lines,you can see it yourself when running the script )***

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe
molecule_factory = MoleculeFactory()

## non active case
molecule_symbol = 'H2'
type_of_generator = 'spin_complement_gsd'
transform = 'JW'
active = False
r, geometry, charge, spin, basis = molecule_factory.get_parameters(molecule_symbol)
print(" --------------------------------------------------------------------------")
print("Running in the non active case: ")
print("                     molecule symbol: %s " %(molecule_symbol))
print("                     molecule basis: %s " %(basis))
print("                     type of generator: %s " %(type_of_generator))
print("                     transform: %s " %(transform))
print(" --------------------------------------------------------------------------")

print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Generate Hamiltonians and Properties from :")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")
hamiltonian, hamiltonian_sparse, hamiltonian_sp, hamiltonian_sp_sparse, n_elec, noons_full, orb_energies_full, info = molecule_factory.generate_hamiltonian(molecule_symbol,active=active, transform=transform)
nbqbits = len(orb_energies_full)
print(n_elec)
hf_init = molecule_factory.find_hf_init(hamiltonian, n_elec, noons_full, orb_energies_full)
reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Generate Cluster OPS from :")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")
pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
# for case of UCCSD from  library
# pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse,theta_MP2, hf_init = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator,transform=transform, active=active)

print('Pool size: ', pool_size)
print('length of the cluster OP: ', len(cluster_ops))
print('length of the cluster OPS: ', len(cluster_ops_sp))
print(hf_init_sp)
print(reference_ket)
print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Start adapt-VQE algorithm:")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")


n_max_grads = 1
optimizer = 'COBYLA'                
tolerance = 10**(-6)            
type_conver = 'norm'
threshold_needed = 1e-2
max_external_iterations = 35
fci = info['FCI']
fermionic_adapt_vqe(hamiltonian_sparse, cluster_ops_sparse, reference_ket, hamiltonian_sp,
        cluster_ops_sp, hf_init_sp, n_max_grads, fci, 
        optimizer,                
        tolerance,                
        type_conver = type_conver,
        threshold_needed = threshold_needed,
        max_external_iterations = max_external_iterations)




```

We make an analysed plot for the converged energy and the fidelity. It took 5 fermionic adapt iteration and the converged energy is -1.1516 Ha with the fidelity : 0.999 and  368 number of CNOT gates




![image](/uploads/skack1.png)




















{{< math >}}
{{< /math >}} 






