---
title: Parameter-Shift Rule
linkTitle: 5- Parameter-Shift Rule
---

![image](/uploads/sstack5.png)


<!--more-->

Let us describe below how to act the parameter-shift rules on the UCC circuit ansatz. 
In the present work, we use a UCC ansatz truncated at the single and double excitation levels with a single Trotter step. Off course, one can go further to triple, quadruple and further excitation, however our goal in this section is to show how to link the parameter shift rule on a unitary coupled cluster gates in a circuit, which is the most important key point. 
Once these approximations are performed, each exponential of a fermionic excitation operator can be directly implemented as a sequence of gates.
This in practice, can be done by  using the standard CNOT staircase method (described in the previous chapter). The unitary transformation carried out by the circuit can thus be broken down into a product of local unitaries

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




The energy {{< math >}}{{< /math >}} according to the chosen optimization algorithm.


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








