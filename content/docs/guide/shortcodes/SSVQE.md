---
title: Subspace-search variational quantum eigensolver for excited states
---

![image](/uploads/sstack9.png)


<!--more-->

In Variational Quantum Eigensolver (VQE), the real parameters $\theta$ for the ansatz states $|\psi \left( \theta  \right) \rangle $ are classically optimised with respect to the expectation value of the Hamiltonian in the equation ; it is computed using a lowdepth quantum circuit. As a result of the variational
principle, finding the global minimum of $E\left( \theta  \right) $ is equivalent to finding the ground state energy of $H$


In this section,  our goal is to extend to calculate find excited states from HEA ansatz model, using subspace-search VQE (SSVQE). The SSVQE takes two or more orthogonal states as inputs to a parametrized quantum circuit, and minimizes the expectation value of the energy in the space spanned by those states. In this work, the proposed algorithm can find the $k$-th excited state state that
works on an $n$-qubit quantum computer; SSVQE  runs as follows:

**Algorithm**


1. Construct an ansatz circuit \( U(\bm{\theta}) \) and choose input states \( \left\{ \ket{\varphi_j} \right\}_{j=0}^k \) whic

2. Minimize 
    \[
    \mathcal{L}_1(\bm{\theta}) = \sum_{j=0}^k \langle \varphi_{j} | U^\dagger(\bm{\theta}) H U(\bm{\theta}) | \varphi_{j} \rangle.
    \]
    We denote the optimal \( \bm{\theta} \) by \( \bm{\theta}^* \).


3. Construct another parametrized quantum circuit \( V(\bm{\phi}) \) that only acts on the space spanned by \( \left\{ \ket{\varphi_j} \right\}_{j=0}^k \).

4. Choose an arbitrary index \( s \in \{0, \ldots, k\} \), and maximize  
    \[
    \mathcal{L}_2(\bm{\phi}) = \langle \varphi_{s} | V^\dagger(\bm{\phi}) U^\dagger(\bm{\theta}^*) H U(\bm{\theta}^*) V(\bm{\phi}) | \varphi_{s} \rangle.
    \]
We note that, in practice, the input states \( \left\{ \ket{\varphi_j} \right\}_{j=0}^k \) will be chosen from a set of states which are easily preparable, such as the computational basis (in our work, we use the binary representation technique for generating the basis). Additionally, in step 2, we can find the subspace which includes \( \ket{E_k} \) as the highest energy state, using a carefully constructed ansatz  $U(\boldsymbol{\theta})$. The unitary \( V(\bm{\phi}) \) is responsible for searching in that subspace. By maximizing \( \mathcal{L}_2(\bm{\phi}) \), we find the \( k \)-th excited state \( \ket{E_k} \).


A particular instance of the "hardware-efficient ansatz" - the entanglement pattern with 6 CX gates and depth = 1 is shown. The parameters $\boldsymbol{\theta}$ are optimized to to minimize the cost function $\mathcal{L}_{1}$ then the ansatz structure will be updated the parameters, called $\boldsymbol{\phi }$ thus to optimize $\mathcal{L}_{2}$ 



![image](/uploads/sslack1.png)


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from numpy import binary_repr
from qat.qpus import get_default_qpu

qpu = get_default_qpu()
method = "BFGS"
model = hamiltonian_sp
vals = 15

eigenvec_input_tar = calculate_eigen_vectors(model, vals)
eigenvec_input = [eigenvec_input_tar[2],eigenvec_input_tar[8], eigenvec_input_tar[13]] 


energy_lists = {f"energy_circ_{i}": {method: []} for i in range(len(circuits_store))}
fidelity_lists = {f"fidelity_circ_{i}": {method: []} for i in range(len(circuits_store))}

def opt_funct(circuits, model, qpu, nqbits, energy_lists, fidelity_lists, weight, eigenvec_input):
    def input_funct(x):
        total_energy = 0
        for i, circ in enumerate(circuits):
            bound_circ = circ.bind_variables({k: v for k, v in zip(sorted(circ.get_variables()), x)})
            result = qpu.submit(bound_circ.to_job(observable=model))
            energy = result.value
            energy_lists[f"energy_circ_{i}"][method].append(energy)

            # Calculate fidelity
            fidelity = fun_fidelity(bound_circ, eigenvec_input[i], nqbits)
            fidelity_lists[f"fidelity_circ_{i}"][method].append(fidelity)
            #print(fidelity)

            total_energy += weight[i] * energy
        return total_energy

    def callback(x):
        for i, circ in enumerate(circuits):
            bound_circ = circ.bind_variables({k: v for k, v in zip(sorted(circ.get_variables()), x)})
            result = qpu.submit(bound_circ.to_job(observable=model))
            energy = result.value
            energy_lists[f"energy_circ_{i}"][method].append(energy)

            # Calculate fidelity
            fidelity = fun_fidelity(bound_circ, eigenvec_input[i], nqbits)
            fidelity_lists[f"fidelity_circ_{i}"][method].append(fidelity)

    return input_funct, callback


input_funct, callback = opt_funct(circuits_store, model, qpu, nqbits, energy_lists, fidelity_lists, weight, eigenvec_input)
options = {"disp": True, "maxiter": 5000, "gtol": 1e-7}
Optimizer = scipy.optimize.minimize(input_funct, x0=init_theta_list, method=method, callback=callback, options=options)

# Plot energy
plt.rcParams["font.size"] = 18
all_energy_lists = []

for i in range(len(circuits_store)):
    energy_list = energy_lists[f"energy_circ_{i}"][method]
    all_energy_lists.append(energy_list)
    plt.plot(range(len(energy_list)), energy_list, label=f"Energy for k={binary_repr(k_lst[i]).zfill(4)}")

    # Print the final energy for each k
    final_energy = energy_list[-1]
    print(f"Final energy for k={binary_repr(k_lst[i]).zfill(4)}: {final_energy}")

plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("Energy Evolution")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()




# Plot fidelity
plt.figure()
all_fidelity_lists = []

for i in range(len(circuits_store)):
    fidelity_list = fidelity_lists[f"fidelity_circ_{i}"][method]
    all_fidelity_lists.append(fidelity_list)
    plt.plot(range(len(fidelity_list)), fidelity_list, label=f"Fidelity for k={binary_repr(k_lst[i]).zfill(4)}")

    # Print the final fidelity for each k
    final_fidelity = fidelity_list[-1]
    print(f"Final fidelity for k={binary_repr(k_lst[i]).zfill(4)}: {final_fidelity}")

plt.xlabel("Iterations")
plt.ylabel("Fidelity")
plt.title("Fidelity Evolution")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=18)
plt.show()

```


We implement the SSVQE  on the  molecular Hamiltonians such as H$_2$ molecule ( with a fixed distance between two hydrogen atoms $r=0.85 $   at the STO-3G minimal basis set. We choose our own weight (W) for the minimization; We obtain accurate energies with height fidelity rate $\left[ 0,978\longrightarrow 0,998 \right] $. Meaning that the algorithms assures assures the orthogonality of the states at the input of the ansatz circuit

![image](/uploads/sslack2.png)


The energy levels of the Hamiltonian of H$_2$ in the STO-3g basis set: four qubits. The fidelity is close to one. It is the Overlap $\langle\Psi_j(\vec{\theta})
|\Psi_{k}\rangle$ between the 
computed VQE state at each optimization step $j$ and the theoretical ground (or excited)
state $|\Psi_{k}\rangle$ of Hamiltonian $H$,  where  $|\Psi_{k}\rangle$ is obtained through diagonalization.


{{< math >}}
{{< /math >}} 













