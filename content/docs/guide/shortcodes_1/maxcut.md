---
title: Maxcut by Quantum annealing
linkTitle: 10 - Maxcut by Quantum annealing
weight: 10

---

# 1. Graph definition 
``` python
import networkx as nx
from itertools import chain, combinations
import matplotlib.pyplot as plt

import numpy as np
```

``` python
edgelists = [[(0, 1), (0, 2), (1, 2)], 
             [(0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 3), (2, 4)], 
             [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 5), (3, 4), (3, 5)],
             [(0, 1), (0, 4), (0, 6), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 3), (2, 4), (3, 4), (3, 5)]]


coordinate_list = [[(0, 0), (0, 1), (1, 1)],
                   [(1, 1), (2, 0), (0, 1), (1, 2), (1, 0)],
                   [(1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2)],
                   [(1, 2), (1, 1), (2, 0), (1, 0), (2, 1), (0, 0), (0, 2)]]



G_set = [nx.from_edgelist(edgelist) for edgelist in edgelists]

list_matrice_J = []
for graphe in G_set:
    list_matrice_J.append(nx.to_numpy_array(graphe , dtype=int))

fig, axs = plt.subplots(2, 2)
for i, ax in enumerate(fig.axes):
    pos = {n: pos for n, pos in enumerate(coordinate_list[i])}  
    nx.draw_networkx(G_set[i], pos, ax=ax)
    ax.set_title("N = {}".format(len(G_set[i].nodes())))

#fig.suptitle("Graphs of Interest")
plt.tight_layout()
plt.show()
```


![images](/uploads/app10/42d8e6221f9dc8947c4033331fcef141a48e39d7.png)

# 2. Classical calculation 
``` python
from itertools import combinations

def get_max_cut(G):
    """
    This function computes the max cut of a given graph.

    Input: Graph G

    Output: max_cut of the graph and associated partition
    """
    def two_partitions(lst):
        result = set()
        
        # Generate all possible combinations for the first subset
        for i in range(1, len(lst)):
            for part1 in combinations(lst, i):
                part1 = set(part1)
                part2 = set(lst) - part1  # The other subset is what remains
                # Add the partition to the set to avoid duplicates
                parti = frozenset([frozenset(part1), frozenset(part2)])
                result.add(parti)
        
        # Convert partitions to lists 
        return [list(map(list, parti)) for parti in result]
    
    partitions = two_partitions(range(len(G.nodes())))
    new_partitions = []

    for part in partitions:
        cut = 0
        for u in part[0]:
            for v in part[1]:
                if G.has_edge(u, v):
                    cut += 1

        # Add the partition with the cut count at the beginning
        new_partitions.append([cut] + part)

    max_cut = max(partition[0] for partition in new_partitions)
    max_cut_partitions = [partition for partition in new_partitions if partition[0] == max_cut]

    return max_cut, max_cut_partitions
```

``` python
for graph in G_set:
    print(get_max_cut(graph))
```


    (2, [[2, [1, 2], [0]], [2, [1], [0, 2]], [2, [0, 1], [2]]])
    (5, [[5, [3, 4], [0, 1, 2]], [5, [0, 3, 4], [1, 2]], [5, [1, 2, 3], [0, 4]], [5, [0, 2], [1, 3, 4]]])
    (7, [[7, [1, 2, 4, 5], [0, 3]], [7, [0, 4, 5], [1, 2, 3]], [7, [1, 2, 4], [0, 3, 5]], [7, [0, 1, 5], [2, 3, 4]]])
    (9, [[9, [2, 4, 5, 6], [0, 1, 3]]])

# 3. Quantum annealing gate based

The method consists of starting from a Hamiltonian that does not
correspond to our graph but rather to a placement of the graph\'s points
independently of each other; this is what we represent by the
Hamiltonian $H_x$. Then, we construct an interaction Hamiltonian that
maps the connections between the nodes of our graph, which we call the
Hamiltonian $H_c$. The Hamiltonian $H_x$ has a well-known ground state.
This method involves initializing the system in the ground state of the
Hamiltonian $H_x$, then using the parameters $\eta$ to slowly evolve the
Hamiltonian $H_{x}$ towards the Hamiltonian $H_c$. Since the evolution
is slow, this ensures that the system remains in the ground state
throughout the procedure, and thus, in the end, the system is in the
ground state of the Hamiltonian associated with our graph $H_c$.

### The code

1.  I create the function `u_x` which is the evolution operator of the
    Hamiltonian `hx`. 
    $$
    H_x = \sum_{i}X_i
    $$ 
    $$
    U_x = e^{-iH_{x}dt}
    $$

2.  I create the function `u_c` which is the evolution operator of the
    Hamiltonian `hc`. 
    $$
    H_c = \sum_{i<j}J_{i,j}Z_{i}Z_{j}
    $$ 
    $$
    U_c = e^{-iH_{c}dt}
    $$

3.  I create the function `U_init()` which creates a circuit that
    returns the ground state of `hx`.

4.  I create the function `u_n` which applies the Suzuki-Trotter
    approximation of $U_{\eta}$. 
    $$
    U_{\eta} = \prod_{j = 1}^{\eta}exp\left(-i\left[\frac{j - 1}{\eta -1}H_{c} + \frac{\eta - j}{\eta -1}H_{x}\right]\right)
    $$ 
    With Suzuki-Trotter approximation we have: 
    $$
    U_{\eta} \approx \prod_{j = 1}^{\eta}\left[U_{c}\left(\frac{j - 1}{m(\eta -1)}\right)U_{x}\left(\frac{\eta - j}{m(\eta -1)}\right)\right]^{m}
    $$

We notice that at each iteration, as $\frac{j - 1}{m(\eta -1)}$
increases, $\frac{\eta - j}{m(\eta -1)}$ decreases, which reflects the
transition from the Hamiltonian `hx` to `hc`.

1.  The function `qaoa_circuit` simply initializes the circuit in the
    ground state of $H_{x}$ before applying `u_n`.

2.  The function `ground_state_optimizer()` in this function, I use the
    qaoa_circuit function with the evolve method to retrieve the state
    vector of our circuit after the evolution of our system:
    $$\ket{\Phi_{\eta,m}} = U_{\eta , m}U_{init}\ket{0}$$ 
    Then I
    calculate `cost`, which is the eigenvalue of this state vector:
    $$C = \bra{\Phi_{\eta,m}}H_{c}\ket{\Phi_{\eta,m}}$$ 
    then we perform
    optimization on C to make it as small as possible and thus obtain a
    good approximation of the ground state $\ket{\Phi_{\eta,m}}$. The
    optimization doesn\'t seem to work very well, probably because the
    increments of $\eta$ and `m` are integers. I could solve this
    problem for m by changing the way I construct the `u_n` circuit. I
    could first obtain the matrix corresponding to
    $$ U_{c}(\frac{j - 1}{m(\eta - 1)})U_{x}(\frac{\eta - j}{m(\eta - 1)})$$
    Then apply the power m and transform the resulting matrix into an
    observable, and finally add it to the `u_c` circuit.

3.  The function `calculate_maxcut_from_strings()` executes the final
    task. At this stage, I already have the bit string that corresponds
    to the partition of the maxcut. In fact, the \"0\" bits form one
    partition and the \"1\" bits form the other. For example, if I have
    \"0101\" with the order \"0123\", then the nodes from zero, namely
    {0, 2}, form one partition and {1, 3} form the other partition. It
    only remains to determine the number of edges that connect these
    partitions, which is the maxcut.

Unfortunately, I couldn\'t find a simple Python function to estimate the
maxc for quick comparisonut.

Also, I notice a small bug at the end of execution. When I perform the
following execution, I don\'t have any convergence. It\'s only after a
third execution that I obtain results.keep the largest one.

``` python
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler

from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2

from qiskit_ibm_runtime import EstimatorV2 as Estimator

```

## 3.1 Build $U_{x}$

``` python
def U_x(n, t):
    """
        input: n: nombre de qubit
               t: temps d'évolution
        ouput: QuantumCircuit
    """
    # Définir les matrices de Pauli et l'identité
    I = SparsePauliOp("I")
    X = SparsePauliOp("X")

    # Initialiser un opérateur total vide
    operator_total = None

    # Boucle pour créer les opérateurs et les additionner
    for i in range(n):
        A = [I] * n
        A[i] = X
        operator = A[0]
        
        for matrix in A[1:]:
            operator = operator ^ matrix
        if operator_total is None:
            operator_total = operator
        else:
            operator_total += operator

    # Construire la porte d'évolution
    evo = PauliEvolutionGate(operator_total, time=t)

    # Insérer dans un circuit
    circuit = QuantumCircuit(n)  # Le nombre de qubits doit correspondre à la longueur de A
    circuit.append(evo, range(n))

    return circuit
```

``` python
# Exemple d'utilisation de la fonction
n = 4
t = 0.2
circuit = U_x(n, t)
circuit.decompose().draw("mpl")
```

![images](/uploads/app10/07008f6d0fa11958622abb7d543a23382f45f330.png)




``` python
def U_c(n, J, t):
    # Définir les matrices de Pauli et l'identité
    """
    input: nombre de qubit
        branches du graphes
        temps d'évolution
    ouput: QuantumCircuit
    """
    I = SparsePauliOp("I")
    Z = SparsePauliOp("Z")

    # Initialiser un opérateur total vide
    operator_total = None

    # Boucle pour créer les opérateurs et les additionner
    for (i, j) in J:
        # Créer un vecteur de matrices I
        A = [I] * n
        # Remplacer les ième et jième éléments par Z
        A[i] = Z
        A[j] = Z
        # Initialiser l'opérateur avec la première matrice du vecteur
        operator = A[0]
        # Boucle pour effectuer les produits tensoriels
        for matrix in A[1:]:
            operator = operator ^ matrix
        # Additionner les opérateurs
        if operator_total is None:
            operator_total = operator
        else:
            operator_total += operator

    # Construire la porte d'évolution
    evo = PauliEvolutionGate(operator_total, time=t)

    # Insérer dans un circuit
    circuit = QuantumCircuit(n)  # Le nombre de qubits doit correspondre à la longueur de A
    circuit.append(evo, range(n))

    return circuit
```

``` python
# Exemple d'utilisation de la fonction
n = 5
adjence_matrix = edgelists[1]
t = 0.2
circuit = U_c(n, adjence_matrix, t)
circuit.decompose().draw("mpl")
```


![images](/uploads/app10/a3f9a71ff42147a5a68c072a98a04608c1b8add0.png)


## 3.3 Build $U_{\eta}$ 

``` python
def U_n(neta: int, m: int , nombre_de_qubit, graphe):
    circuit = QuantumCircuit(nombre_de_qubit)
    
    for j in range(1, neta + 1):
        t_c = (j - 1) / (m * (neta - 1))
        t_x = (neta - j) / (m * (neta - 1))
        
        
        Uc = U_c(nombre_de_qubit, graphe, t_c)
        Ux = U_x(nombre_de_qubit, t_x)
        
        circuit_de_base = QuantumCircuit(nombre_de_qubit)
        circuit_de_base = circuit_de_base.compose(Uc)
        circuit_de_base = circuit_de_base.compose(Ux)
        
        for i in range(m - 1):
            circuit_de_base = circuit_de_base.compose(Uc)
            circuit_de_base = circuit_de_base.compose(Ux)
        
        circuit = circuit.compose(circuit_de_base)
    
    return circuit
```

``` python
circuit = U_n(neta = 2 , m = 3 , nombre_de_qubit= 3 , graphe= edgelists[0])
circuit.draw
```


![images](/uploads/app10/a2369cd3ceda15300d055f3755478d5e11d1fe19.png)


# 4 Post-processing and formatting 

``` python
def remove_complementary_bits(bit_dict):
    # Dictionary to store without redundancy
    unique_bits = {}
    
    for bits, value in bit_dict.items():
        # Generate the complement
        complement = ''.join('1' if b == '0' else '0' for b in bits)
        # If the complement is in unique_bits, do not add bits
        if complement in unique_bits:
            continue
        # Add bits to unique_bits
        unique_bits[bits] = value
    
    return unique_bits

def get_bit_positions(bit_string):
    # List of positions of 0s and 1s
    pos_zeros = [i for i, bit in enumerate(bit_string) if bit == '0']
    pos_ones = [i for i, bit in enumerate(bit_string) if bit == '1']
    return pos_zeros, pos_ones

def plot_top_5_histogram(bit_dict):
    # Remove redundant and complementary strings
    unique_bits = remove_complementary_bits(bit_dict)
    
    # Sort items by values and keep the top 5 largest
    top_10 = sorted(unique_bits.items(), key=lambda item: item[1], reverse=True)[:5]
    
    # Separate bit strings and values
    bit_strings, values = zip(*top_10)
    
    # Get positions of 0s and 1s for each bit string
    bit_positions = [get_bit_positions(bits) for bits in bit_strings]
    
    # Display the positions of 0s and 1s for each bit string
    for bit_string, (pos_zeros, pos_ones) in zip(bit_strings, bit_positions):
        print(f"bits string: {bit_string}")
        print(f"Set 1: {pos_zeros}")
        print(f"Set 2: {pos_ones}\n")
    
    # Plot the histogram
    plt.figure(figsize=(6, 3))
    plt.bar(bit_strings, values, width=0.3)
    plt.xlabel('Bit strings')
    plt.ylabel('Values')
    plt.xticks(rotation=90)
    plt.show()


# Plot the histogram and display bit positions
for r in list_result:
    plot_top_5_histogram(r)
```


    bits string: 011
    Set 1: [0]
    Set 2: [1, 2]

    bits string: 110
    Set 1: [2]
    Set 2: [0, 1]

    bits string: 010
    Set 1: [0, 2]
    Set 2: [1]

    bits string: 000
    Set 1: [0, 1, 2]
    Set 2: []


![images](/uploads/app10/27efae2cfc94edd0f08ea62e66371a91f8ac3005.png)

    bits string: 00011
    Set 1: [0, 1, 2]
    Set 2: [3, 4]

    bits string: 10011
    Set 1: [1, 2]
    Set 2: [0, 3, 4]

    bits string: 10100
    Set 1: [1, 3, 4]
    Set 2: [0, 2]

    bits string: 10001
    Set 1: [1, 2, 3]
    Set 2: [0, 4]

    bits string: 10101
    Set 1: [1, 3]
    Set 2: [0, 2, 4]


![images](/uploads/app10/4a34eba6e8c42c814bda7ba6aef24d837c86cc99.png)

    bits string: 100101
    Set 1: [1, 2, 4]
    Set 2: [0, 3, 5]

    bits string: 100100
    Set 1: [1, 2, 4, 5]
    Set 2: [0, 3]

    bits string: 011100
    Set 1: [0, 4, 5]
    Set 2: [1, 2, 3]

    bits string: 110001
    Set 1: [2, 3, 4]
    Set 2: [0, 1, 5]

    bits string: 001001
    Set 1: [0, 1, 3, 4]
    Set 2: [2, 5]

![images](/uploads/app10/dfa257ec8121153ddb343cc9de11e87c2c5402c8.png)

    bits string: 1101000
    Set 1: [2, 4, 5, 6]
    Set 2: [0, 1, 3]

    bits string: 0100101
    Set 1: [0, 2, 3, 5]
    Set 2: [1, 4, 6]

    bits string: 1110010
    Set 1: [3, 4, 6]
    Set 2: [0, 1, 2, 5]

    bits string: 1010110
    Set 1: [1, 3, 6]
    Set 2: [0, 2, 4, 5]

    bits string: 1011011
    Set 1: [1, 4]
    Set 2: [0, 2, 3, 5, 6]

![images](/uploads/app10/34db132e6978526d746d393076fe3a5d46b916ba.png)




### **References**

<a href="https://www.science.org/doi/10.1126/science.1057726" style="color:#1E90FF;">
Edward Farhi, Jeffrey Goldstone,Joshua Lapan, Andrew Lundgren, Daniel Preda. "A Quantum Adiabatic Evolution Algorithm Applied to Random Instances of an NP-Complete Problem" Science. 284 (5415): 779–81.
</a>





### **About the author**



<div align="center">
  <img src="/uploads/app10/isaac.jpg" alt="Author's Photo" width="150" style="border-radius: 50%; border: 2px solid #1E90FF;">
  <br>
  <strong>Isaac Christ Donchi Kamga</strong>
  <br>
  <em>Ing , Msc in quantum computing and quantum engineering, France</em>
  <br>
  <a href="https://www.linkedin.com/in/don-isaac/" style="color:#1E90FF;">LinkedIn</a>
</div>





{{< math >}}
{{< /math >}} 