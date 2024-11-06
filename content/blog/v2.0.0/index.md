---
title: The Journal of Physical Chemistry A, 2023, Vol 127/Issue 15, 3543–3550
summary: Extension of the Trotterized Unitary Coupled Cluster to Triple Excitations
date: 2024-01-19
authors:
  - admin
tags:
  - Quantum Computing.
  - Quantum Chemistry
  - OpenVQE/myQLM-fermion
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com)'
---

The reasearch paper "Extension of the Trotterized Unitary Coupled
Cluster to Triple Excitations" is now available on The Journal of Physical Chemistry! This release grants some following highlights include:

Highlights include:

- The research paper addresses the need to extend the Trotterized Unitary Coupled Cluster Single and Double (UCCSD) ansatz to include true Triple T excitations in order to recover missing correlation effects for molecular simulations on quantum computers
- The limitations of UCCSD for larger molecules are discussed, and the addition of (true) Triple T excitations to the UCCSD approach is proposed to improve accuracy.
- The paper introduces the Trotterized UCCSDT approach and analyzes the behavior of triple excitations on a set of molecules compared to the initial UCCSD.
- The computational methodology used for the theoretical experiments and the results obtained from testing several molecules using UCCSDT-VQE and sym-UCCSDT-VQE methods are presented.
- . The significance of incorporating symmetries, such as spin and point group symmetries, to reduce the number of circuit excitations in the UCCSDT ansatz and accelerate the optimization process for tackling larger molecules is emphasized. 

- The paper provides insights into the UCCSD and Trotterized UCCSD ansatz, acknowledging their successes and limitations in representing wavefunctions for molecular simulations.

- The significance of incorporating symmetries, such as spin and point group symmetries, to reduce the number of circuit excitations in the UCCSDT ansatz and accelerate the optimization process for tackling larger molecules is emphasized.
Thank you to everyone who contributed to this release!

- Extensive numerical tests on molecules such as LiH, BeH2, and H2O using the UCCSDT-VQE and sym-UCCSDT-VQE methods are presented, demonstrating the superiority of the sym-UCCSDT approach in terms of accuracy, particularly in recovering correlation energy missed by the sym-UCCSD ansatz. 

- The paper highlights the need for further analysis to understand the limitations of the sym-UCCSDT approach at larger bond lengths and suggests potential improvements by adding higher-order excitations, stressing the potential of the Trotterized UCCSDT approach to achieve competitive results with the gold-standard CCSD(T) classical methods


## Theoretical Framework and Quantum Computing

The paper describes the theoretical formalism of the Unitary Coupled Cluster (UCC) method for electronic structure calculations, including detailed formalism of triple excitations in its simplified form after applying both spin and orbital symmetries. It highlights the challenges and opportunities associated with the use of quantum computers for solving problems in quantum chemistry, especially in simulating the full configuration interaction wavefunction of many-electron molecular systems. The Variational Quantum Eigensolver (VQE) is discussed as a promising algorithm for practical implementation on Noisy Intermediate Scaled Quantum (NISQ) devices. The paper also provides insights into the UCCSD and Trotterized UCCSD ansatz, acknowledging their successes and limitations in representing wavefunctions for molecular simulations.

## Incorporating Symmetries for Computational Efficiency

The authors emphasize the significance of incorporating symmetries, such as spin and point group symmetries, to reduce the number of circuit excitations in the UCCSDT ansatz and accelerate the optimization process for tackling larger molecules. They discuss the reduction in the number of optimization parameters due to the incorporation of symmetry constraints and the use of the Trotterization approach for breaking up the exponential of a sum into a product of individual exponentials

##  Numerical Tests and Comparative Analysis

The research paper extensively presents the authors' numerical tests on molecules such as LiH, BeH2, and H2O using the UCCSDT-VQE and sym-UCCSDT-VQE methods. It discusses the reductions in the number of optimization parameters for different molecules and the results obtained, showing that the sym-UCCSDT method improves the overall accuracy by at least two orders of magnitudes with respect to standard UCCSD. The authors compare the performance of the sym-UCCSDT method with classical methods such as CCSD, CCSD(T), and CCSDT-full and demonstrate the superiority of the sym-UCCSDT approach in terms of accuracy, particularly in recovering correlation energy missed by the sym-UCCSD ansatz

##  Limitations and Future Directions
Furthermore, the paper highlights the need for further analysis to understand the limitations of the sym-UCCSDT approach at larger bond lengths and suggests potential improvements by adding higher-order excitations. It also discusses the implications of the correlation effects on the accuracy of the sym-UCCSDT method and provides a thorough analysis of the errors and energy differences with reference to the FCI energies. The authors stress the potential of the Trotterized UCCSDT approach to achieve competitive results with the gold-standard CCSD(T) classical methods, delineating its significance for the quantum chemistry community