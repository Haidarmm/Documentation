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


Despite the importance of the excited states, classical computation suffers from the increasing computational cost resulting from high number of function evaluations. \\
In the nutshell, this work greatly extends the practicability of the VQE by enabling it to find the excited states efficiently for possible application of NISQ devices further.









$$
U\left(x;\bm{\theta}\right)=U_N\left(\bm{\theta}_N\right)U_{N-1}\left(\bm{\theta}_{N-1}\right)\cdots U_i\left(\bm{\theta}_i\right) \cdots U_1\left(\bm{\theta}_1\right)U_0\left(x\right)
$$


Each of these gates is unitary, and therefore must have the form


$$
U_j\left(\gamma_j\right)=\exp{\left(i\gamma_jH_j\right)}
$$

where  $H_j$ is a Hermitian operator which generates the gate and $\gamma_j$ is the gate parameter. When using CNOT staircase method,  $U_j\left(\gamma_j\right)$ are typically equivalent to $R_z(\bm{\theta})$ gates.
Since for any type of coupled cluster excitations, the circuit, in practice should be composed of multiple parameterized $R_z(\bm{\theta})$ gates, and some other non-parameterized gates such as (CNOT, Hadamard etc.). There should be then a tool  how to  make the quantum circuit function  to compute gradient ($\nabla{\bm{\theta}_i}{f}(x;\bm{\theta}_i$)) for a certain $\bm{\theta}_i$:\\
in fact any gates applied before gate $i$ into the initial state, can be expressed as:

$$
\left|\psi_{i-1}\right\rangle=U_{i-1}\left(\bm{\theta}_{i-1}\right)\cdots U_1\left(\bm{\theta}_1\right)U_0\left(x\right)\left|0\right\rangle
$$

Similarly, any gates applied after gate $i$ are combined with the observable, which is the Hamiltonian in our case

$$
\hat{H}_{i+1}=U_N^\dag(\bm{\theta}_N)\cdot
U^\dag_{i+1}(\bm{\theta}_{i+1})\hat{H}U_{i+1}(\bm{\theta}_{i+1})\cdots U_N(\bm{\theta}_N) 
$$

With this simplification, the quantum circuit function becomes

$$
f(x;\bm{\theta})=\left\langle\psi_{i-1}\left|U_i^\dag(\bm{\theta}_i)\hat{H}_{i+1}U_i(\bm{\theta}_i)\right|\psi_{i-1}\right\rangle
$$
now suppose $$\mathcal{M}_{\bm{\theta}_i}(\hat{H}_{i+1}) = U_i^\dag(\bm{\theta}_i)\hat{H}_{i+1}U_i(\bm{\theta}_i) $$ 
then its gradient takes the form 

$$
\nabla_{\bm{\theta}_i}f\left(x;\bm{\theta}\right)=\left\langle\psi_{i-1}\left|\nabla_{\bm{\theta}_i}\mathcal{M}_{\bm{\theta}_i}\left(\hat{H}_{i+1}\right)\right|\psi_{i-1}\right\rangle
$$

as is seen in the expression above, in terms of the circuit, one can leave all other gates as they are, and only  gate  
$U_i\left(\bm{\theta}_i\right)$ should be changed
  when question comes to differentiate it with respect to the parameter  
$\bm{\theta}_i$. Let us now apply the equations above into $R_z$ gate. 
Consider a quantum computer with parameterized gates of the form
$$
    R_i\left(\bm{\theta}_i\right)=\exp{\left(-i\frac{\bm{\theta}_i}{2}\widehat{Z_i}\right)}
$$
where $\widehat{Z_i}={\widehat{Z}_i}^ \dag$ is a Pauli operator. The gradient of $R_z$ is

$$
\nabla_{\bm{\theta}_i}R_i\left(\bm{\theta}_i\right)=-\frac{i}{2}\widehat{Z_i}R_i\left(\bm{\theta}_i\right)=-\frac{i}{2}R_i\left(\bm{\theta}_i\right)\widehat{Z_i}
$$

By substituting this expression  into the quantum circuit function $f(x;\bm{\theta})$, we get

$$
\nabla_{\bm{\theta}_i} f\left(x;\bm{\theta}\right) =  \frac{i}{2} \left\langle \psi_{i-1} \left| R_i\left(\bm{\theta}_i\right)^\dag \left( Z_i \widehat{H}_{i+1} - \widehat{H}_{i+1} Z_i \right) R_i\left(\bm{\theta}_i\right) \right| \psi_{i-1} \right\rangle 
$$
$$= \frac{i}{2} \left\langle \psi_{i-1} \left| R_i\left(\bm{\theta}_i\right)^\dag \left[ Z_i, \widehat{H}_{i+1} \right] R_i\left(\bm{\theta}_i\right) \right| \psi_{i-1} \right\rangle$$


where $\left[X,Y\right]=XY-YX$ is the commutator.
\\ \\
We now make use of the following mathematical identity for commutators involving Pauli operators

$$
\left[\widehat{Z_i},\hat{H}\right] = -i\left(R_i^\dag\left(\frac{\pi}{2}\right)\hat{H}R_i\left(\frac{\pi}{2}\right) - R_i^\dag\left(-\frac{\pi}{2}\right)\hat{H}R_i\left(-\frac{\pi}{2}\right)\right).
$$

Substituting this into the previous equation, we obtain the gradient expression
$$
\nabla_{\bm{\theta}_i} f\left(x;\bm{\theta}\right)= \frac{1}{2}\left\langle\psi_{i-1}\left|R_i^\dag\left(\bm{\theta}_i+\frac{\pi}{2}\right){\widehat{H}}_{i+1}R_i\left(\bm{\theta}_i+\frac{\pi}{2}\right)\right|\psi_{i-1}\right\rangle
$$

$$
-\frac{1}{2}\left\langle\psi_{i-1}\left|R_i^\dag\left(\bm{\theta}_i-\frac{\pi}{2}\right){\widehat{H}}_{i+1}R_i\left(\bm{\theta}_i-\frac{\pi}{2}\right)\right|\psi_{i-1}\right\rangle
$$
Finally, this gradient can be  rewritten in terms of quantum functions:

$$
\nabla_{\bm{\theta}_i} f\left(x;\bm{\theta}\right)=\frac{1}{2}\left[f\left(x;\bm{\theta}+\frac{\pi}{2}\right)-f\left(x;\bm{\theta}-\frac{\pi}{2}\right)\right]
$$

We recognize from the comparison of equations above,
that these unitaries represent instances of the initial gate,
and thus shift the gate’s parameter.  This leads to the
parameter shift rule for circuit gradients. Below is the code for PMRS for LiH in with CAS method reduced to 6 qubits







```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

# Assuming the necessary functions are defined: 
# get_optimization_func, get_grad_func, and the variables circ, qpu, H_sp, nqbits, psi0, theta_0, E0

# Initialize dictionaries to store results
res = {}
energy_list, fid_list = {}, {}

# Define the optimization methods to use
methods = ["CG", "BFGS", "COBYLA"]

# Perform optimization for each method
for method in methods:
    energy_list[method], fid_list[method] = [], []
    my_func = get_optimization_func(circ, qpu, H_sp, method, nqbits, psi0, energy_list, fid_list)
    my_grad = get_grad_func(circ, qpu, H_sp)
    res[method] = scipy.optimize.minimize(my_func, jac=my_grad, x0=theta_0, method=method, options={"maxiter": 50000, "disp": True})

fig, ax = plt.subplots(figsize=(11, 7))
steps = np.arange(0, 700)

col = {"CG": "orange", "COBYLA": "firebrick", "BFGS": "darkcyan"}

for method in ["CG", "BFGS", "COBYLA"]:
    energy_list[method] = np.array(energy_list[method])
    error = np.maximum(energy_list[method] - E0, 1e-6)  # Ensure no values less than 1e-16
    print(f"The error for the {method} method:", error)
    ax.plot(error, "-o", color=col[method], label=f" Subtracted energy [{method}]")
    
    if method == "COBYLA" and error.size > 0:
        ax.fill_between(steps[:len(error)], 1e-6, 1e-3, color="cadetblue", alpha=0.2, interpolate=True, label="Chemical Accuracy")



ax.set_xlabel("optimization step")
ax.set_xlim(0,128)

ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")
plt.savefig("error_H4.pdf")
  # Grid lines for both major and minor ticks

plt.show()

```





![image](/uploads/sstack6.png)

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
fig, ax = plt.subplots(figsize=(11, 7))
steps = np.arange(0, 700)

col = {"CG": "orange", "COBYLA": "firebrick", "BFGS": "darkcyan"}

for method in ["CG", "BFGS", "COBYLA"]:
    energy_list[method] = np.array(energy_list[method])
    error = np.maximum(energy_list[method] - E0, 1e-6)  # Ensure no values less than 1e-16
    print(f"The error for the {method} method:", error)
    ax.plot(error, "-o", color=col[method], label=f" Subtracted energy [{method}]")
    
    if method == "COBYLA" and error.size > 0:
        ax.fill_between(steps[:len(error)], 1e-6, 1e-3, color="cadetblue", alpha=0.2, interpolate=True, label="Chemical Accuracy")



ax.set_xlabel("optimization step")
ax.set_xlim(0,128)

ax.set_yscale('log')
ax.legend()
ax.grid(True, which="both", ls="--")
plt.savefig("error_LiH.pdf")
  # Grid lines for both major and minor ticks

plt.show()
```
![image](/uploads/sstack7.png)


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
fig, ax = plt.subplots(figsize=(11, 7))
steps = np.arange(0, 700)

col = {"CG": "orange", "COBYLA": "firebrick", "BFGS": "darkcyan"}

for method in ["CG", "BFGS", "COBYLA"]:
    fid_list[method] = np.array(fid_list[method])

    ax.plot(fid_list[method], "-o", color=col[method], label= f"fidelity w.r.t true ground state [{method}]")
    

ax.set_xlabel("optimization step")

ax.legend()
ax.grid(True, which="both", ls="--")
plt.savefig("fidelity_H4_.pdf")
  # Grid lines for both major and minor ticks

plt.show()
```



![image](/uploads/sstack8.png)























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











