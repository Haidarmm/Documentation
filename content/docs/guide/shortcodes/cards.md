---
title: OpenVQE Overall
linkTitle: 3- OpenVQE Overall
---

A Openvqe extension to create cards. Cards can be shown as links or as plain text.

## Usage

{{< cards >}}
  {{< card url="../" title="Shortcodes Table" icon="academic-cap" >}}
{{< /cards >}}


## Options

| Parameter  | Description                                                            |
|------------|------------------------------------------------------------------------|
| `molecule_symbol`     | molecule examples : H2 , H4 , H6 , LiH , H2O , CO , CO2 , NH3 etc ... |
| `type_of_generator`    |  user can apply type of generators , such as: uccsd , quccsd , uccgsd , k- upccgsd , etc ...                                       |
| `transform` | user type Jordan wigner (JW), user can also type Bravyi - Kitaev or Parity basis.                                  |
| `active`      | user can use AS (Active Space) method - True or non active space - False                                                                   |

The user specifies these parameters in a class called MoleculeFactory. This class takes those parameters as input

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from openvqe . common_files . molecule_factory import MoleculeFactory
# returns the properties of a molecule :
r , geometry , charge , spin , basis = MoleculeFactory . get_parameters ( molecule_symbol = ’H2O ’)

```



define a function named as generate_hamiltonian that generates the electronic structure Hamiltonian (hamiltonian) and
other properties such as the spin hamiltonian (for example hamiltonian_jw), number of electrons (n_els), the list contains
the number of natural orbital occupation numbers (noons_full), the list of orbital energies (orb_energies_full) where
the orbital energies are doubled due to spin degeneracy and info which is a dictionary that stores energies of some classical
methods( such as Hartree-Fock, CCSD and FCI):

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
Hamiltonian , hamiltonian_jw , n_els , noons_full , orb_energies_full , info = MoleculeFactory . generate_hamiltonian (
molecule_symbol =’H2O ’, active = False , transform =’JW ’)
```

*Briefing about the geometry and energy level of H20 you can visualize the geometry of the molecule on the ORCA application
![image](/uploads/stack8.png)

In addition to that, we define another function named as generate_cluster_ops() that takes as input the name of excitation generator user need (e.g., UCCSD, QUCCSD, UCCGSD, etc.) and internally it calls the file name generator_excitations.py
which allows generate_cluster_ops() to return as output the size of pool excitations, fermionic operators, and JW
transformed operators denoted in our code respectively as pool_size, cluster_ops and cluster_ops_jw:



```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from . generator_excitations import ( uccsd , quccsd , singlet_gsd , singlet_sd , singlet_upccgsd , spin_complement_gsd ,
spin_complement_gsd_twin )
pool_size , cluster_ops , cluster_ops_jw = MoleculeFactory . generate_cluster_ops ( molecule_symbol =’H2O ’,
type_of_generator =’ spin_complement_gsd ’, transform =’JW ’, active = False )
# in our example ’ spin_complement_gsd ’:
def generate_cluster_ops () :
    pool_size , cluster_ops , cluster_ops_jw = None , None , None
    if type_of_generator == ’ spin_complement_gsd ’:
      pool_size , cluster_ops , cluster_ops_jw = spin_complement_gsd ( n_el , n_orb ,’JW ’)
# elif for other excitations (uccsd , quccsd , singlet_upccgsd ...)
# ::::
    return pool_size , cluster_ops , cluster_ops_jw

```
  


Once these are generated, we import them as input to the UCC-family and ADAPT modules.
In the example of fermionic-ADAPT sub-module, we call the function fermionic_adapt_vqe() that takes as parameters the
fermionic cluster operators, spin Hamiltonian, maximum number of gradients to be taken per iteration, the type of optimizer,
tolerance, threshold of norm () and the maximum number of adaptive iterations:


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from openvqe . adapt . fermionic_adapt_vqe import fermionic_adapt_vqe
# choose maximum number of gradients needed (1 ,2 ,3....)
n_max_grads = 1
# choose optimizer needed ( COBYLA , BFGS , SLSQP , Nelder - Mead etc ...)
optimizer = ’COBYLA ’
tolerance = 10**( -6)
# according to a given norm value we stop the ADAPT - VQE loop
type_conver = ’norm ’
threshold_needed = 1e -2
# the maximum external number of iterations to complete the ADAPT - VQE under a given threshold_needed
max_iterations = 35
fci = info [’FCI ’]
# sparse the Hamiltonian and cluster operators using myQLM - fermion tools obtained from MoleculeFactory , which are
explicitly :
hamiltonian_sparse = hamiltonian_jw . get_matrix ( sparse = True )
cluster_ops_sparse = cluster_ops . get_matrix ( sparse = True )
# reference_ket and hf_init_sp can be obtained from class MoleculeFactory ():
reference_ket , hf_init_sp = MoleculeFactory . get_reference_ket ( hf_init , nbqbits , ’JW ’)
# when all these parameters are satisfied , then fermionic -ADAPT - VQE function is:
fermionic_adapt_vqe ( cluster_ops , hamiltonian_sparse , cluster_ops_sparse , reference_ket , h_sp , cluster_ops_jw ,
hf_init_sp , n_max_grads , fci , optimizer , tolerance , type_conver = type_conver , threshold_needed =
threshold_needed , max_external_iterations = max_iterations )

```
The function `fermionic_adapt_vqe()` shows several steps  (1) It prepares the trial state using `prepare_state()`, (2) computes the commutator between the Hamiltonian and the fermionic operator with `compute_gradient()` (or numerically via `hamiltonian_jw | cluster_ops_sp`), (3) sorts the gradients in descending order while excluding zeros using `sorted_gradient()`, and (4) checks if the norm meets a threshold to decide whether to exit or continue. If continuing, it optimizes the maximum gradient operator(s) using `ucc_action()` before appending them to the trial state. The function returns the number of classical parameters, CNOT gates, other gates, optimized energies, and energy difference from FCI. The qubit-ADAPT sub-module is similar but differs in using qubit pool generators, a distinct trial state preparation, and a different gradient calculation method, while returning the same properties as `fermionic_adapt_vqe()`.

### Demo of the fermionic adapt VQE *( Method + Algorithms )*

OpenVQE algorithms have targetted the following object  

![image](/uploads/sstack1.png)

{{% steps %}}

### Step 1

Import the librabries from the main folder

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe
molecule_factory = MoleculeFactory()


```

### Step 2 

In this step we will run the non-active case with 8 qubits

```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
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

After the fifth iteration we obtain 

```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
 --------------------------------------------------------------------------
                     Fermionic_ADAPT-VQE iteration:  5
 --------------------------------------------------------------------------
 Check gradient list chronological order
 Norm of the gradients in current iteration =   0.00009243
 Max gradient in current iteration=  -0.00006448
 Index of the Max gradient in current iteration=  29
Convergence is done
 -----------Final ansatz----------- 
 *final converged energy iteration is      -1.151688545279

```

& 

```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
({'energies': [-1.1327826008725905,
   -1.1381935787336812,
   -1.1446756115964445,
   -1.1516131561999758,
   -1.1516885452787875],
  'energies_substracted_from_FCI': [0.018905946644018012,
   0.01349496878292733,
   0.007012935920164054,
   7.539131663270027e-05,
   2.2378210395856968e-09],
  'norms': [1.1787990251924685,
   0.8635450036709927,
   0.6286051658725658,
   0.5004792404827069,
   0.044516747244794035],
  'Max_gradients': [0.5465649202698061,
   0.4109751817065878,
   0.32276851883318675,
   0.3507577344126926,
   0.02644094985356347],
  'fidelity': [0.9852300170572142,
   0.9870545052389407,
   0.9895621336258196,
   0.9937025822193148,
   0.9999286595965128],
  'CNOTs': [48, 96, 288, 336, 368],
  'Hadamard': [32, 64, 128, 160, 168],
  'RY': [0, 4, 4, 4, 4],
  'RX': [16, 32, 64, 80, 84]},
 {'indices': [38, 32, 29, 23, 2],
  'Number_operators': 5,
  'final_norm': 9.243327155226241e-05,
  'parameters': [-0.02140083663347611,
   -0.025081299644836987,
   -0.046035927664188785,
   -0.03941898799451784,
   -0.005705200709434039],
  'Number_CNOT_gates': 368,
  'Number_Hadamard_gates': 168,
  'Number_RX_gates': 84,
  'final_energy_last_iteration': -1.1516885452787875})

```

### Step 3
In this step we consider the case of active space selection with the H4 molecule


```python {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
## active case
from openvqe.common_files.molecule_factory_with_sparse import MoleculeFactory
from openvqe.adapt.fermionic_adapt_vqe import fermionic_adapt_vqe

# initializing the variables in the case of active 
molecule_symbol = 'H4'
type_of_generator = 'spin_complement_gsd' #'spin_complement_gsd_twin'
transform = 'JW'
active = True


r, geometry, charge, spin, basis = molecule_factory.get_parameters(molecule_symbol)
print(" --------------------------------------------------------------------------")
print("Running in the active case: ")
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
hamiltonian_active, hamiltonian_active_sparse, hamiltonian_sp,hamiltonian_sp_sparse,nb_active_els,active_noons,active_orb_energies,info=molecule_factory.generate_hamiltonian(molecule_symbol,active=active,transform=transform)
print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Generate Cluster OPS from :")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")
nbqbits = hamiltonian_sp.nbqbits
hf_init = molecule_factory.find_hf_init(hamiltonian_active, nb_active_els,active_noons, active_orb_energies)
reference_ket, hf_init_sp = molecule_factory.get_reference_ket(hf_init, nbqbits, transform)
pool_size,cluster_ops, cluster_ops_sp, cluster_ops_sparse = molecule_factory.generate_cluster_ops(molecule_symbol, type_of_generator=type_of_generator, transform=transform, active=active)
print("Clusters were generated...")
print('Pool size: ', pool_size)
print('length of the cluster OP: ', len(cluster_ops))
print('length of the cluster OPS: ', len(cluster_ops_sp))
print('length of the cluster OPS_sparse: ', len(cluster_ops_sp))

print(" --------------------------------------------------------------------------")
print("                                                          ")
print("                      Start adapt-VQE algorithm:")
print("                                                          ")
print(" --------------------------------------------------------------------------")
print("                                                          ")

n_max_grads = 1
optimizer = 'COBYLA'                
tolerance = 10**(-7)            
type_conver = 'norm'
threshold_needed = 1e-3
max_external_iterations = 30
fci = info['FCI']
fermionic_adapt_vqe(hamiltonian_active_sparse, cluster_ops_sparse, reference_ket, hamiltonian_sp,
        cluster_ops_sp, hf_init_sp, n_max_grads, fci, 
        optimizer,                
        tolerance,                
        type_conver = type_conver,
        threshold_needed = threshold_needed,
        max_external_iterations = max_external_iterations)
```

After the third iteration we obtain:

```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
--------------------------------------------------------------------------
                     Fermionic_ADAPT-VQE iteration:  3
 --------------------------------------------------------------------------
 Check gradient list chronological order
 Norm of the gradients in current iteration =   0.00000157
 Max gradient in current iteration=   0.00000098
 Index of the Max gradient in current iteration=  22
Convergence is done
 -----------Final ansatz----------- 
 *final converged energy iteration is      -2.150071872977


```
 &
```bash {class="my-class" id="my-codeblock" lineNos=inline tabWidth=2}
({'energies': [-2.1475588531337, -2.149998032049138, -2.150071872976795],
  'energies_substracted_from_FCI': [0.0307547797466996,
   0.028315600831261722,
   0.02824175990360489],
  'norms': [0.9780904621524203, 0.44476756978359294, 0.04777693846151295],
  'Max_gradients': [0.5600407258694875,
   0.3073244780480784,
   0.027972834656648148],
  'fidelity': [0.9796754217306467, 0.9989670200933669, 0.9999401845822582],
  'CNOTs': [48, 96, 128],
  'Hadamard': [32, 64, 72],
  'RY': [0, 4, 4],
  'RX': [16, 32, 36]},
 {'indices': [16, 22, 2],
  'Number_operators': 3,
  'final_norm': 1.56924035843115e-06,
  'parameters': [-0.06970457006634999,
   -0.015636706722331747,
   -0.005282460997328426],
  'Number_CNOT_gates': 128,
  'Number_Hadamard_gates': 72,
  'Number_RX_gates': 36,
  'final_energy_last_iteration': -2.150071872976795})
```


{{% /steps %}}


### **References**

<a href="https://arxiv.org/pdf/2206.08798" style="color:#1E90FF;">
Haidar, Mohammad, et al. "Open source variational quantum eigensolver extension of the quantum learning machine for quantum chemistry." Wiley Interdisciplinary Reviews: Computational Molecular Science 13.5 (2023): e1664.
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


