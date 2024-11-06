---
title: WIREs Computational Molecular ScienceVolume 13, Issue 5 e1664
summary: Open source variational quantum eigensolver extension of the quantum learning machine for quantum chemistry
date: 2023-03-15
authors:
  - admin
tags:
  - Quantum Computing.
  - Quantum Chemistry
  - OpenVQE/myQLM-fermion
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com)'
---

The reasearch paper "Open source variational quantum eigensolver extension of the quantum learning machine for quantum chemistry" is now available on Wiley Journal! This release grants some following highlights include:


- The paper introduces the OpenVQE open-source package, which extends the Atos Quantum Learning Machine (QLM) to provide advanced tools for using and developing variational quantum eigensolver (VQE) algorithms for quantum chemistry applications.
-  Present quantum processing units (QPUs) have limited qubit counts and circuit depths due to large errors, and VQE algorithms can potentially overcome such issues.
- The OpenVQE package is designed to work synergistically with the myQLM-fermion open-source module, which provides key QLM resources important for quantum chemistry developments.

- OpenVQE focuses on giving access to modules enabling the use of the unitary coupled cluster (UCC) family of methods and adaptive ansatz algorithms like ADAPT-VQE.

- The UCC family modules in OpenVQE include various features such as different types of UCC generators, including UCCSD, k-UpCCGSD, and QUCCSD.

- The adaptive VQE algorithms include fermionic-ADAPT-VQE and qubit-ADAPT-VQE, with different operator pools.

- The paper presents extensive benchmarks using the OpenVQE/myQLM-fermion package on a range of molecules from 4 to 24 qubits, demonstrating the use of active space selection, MP2 pre-screening initial guesses, and comparing the fermionic and qubit-ADAPT-VQE results.

-  The paper compares the "fixed-length" UCC methods to the ADAPT-VQE approach, showing that ADAPT-VQE can achieve higher accuracy with fewer parameters and gates, depending on the chosen convergence threshold.

- The paper emphasizes the open-source nature of the OpenVQE/myQLM-fermion packages, facilitating their use and contribution by the broader community, and provides perspectives for developing new types of UCC ansätze and/or new variational algorithms within OpenVQE.

## Introduction to OpenVQE package

This paper introduces the OpenVQE open-source package, which extends the Atos Quantum Learning Machine (QLM) to provide advanced tools for using and developing variational quantum eigensolver (VQE) algorithms for quantum chemistry applications. The paper highlights that present quantum processing units (QPUs) have limited qubit counts and circuit depths due to large errors, and that VQE algorithms can potentially overcome such issues.

## Design of the OpenVQE package

The UCC family modules in OpenVQE include various features such as different types of UCC generators (truncated to single and double excitations), including unitary coupled cluster singles and doubles (UCCSD), unitary pair CC with generalized singles and doubles product (k-UpCCGSD), and qubit unitary coupled cluster singles and doubles (QUCCSD). The adaptive VQE algorithms include fermionic-ADAPT-VQE and qubit-ADAPT-VQE, with different operator pools.

## Benchmarking using OpenVQE/myQLM-fermion package

The paper presents extensive benchmarks using the OpenVQE/myQLM-fermion package on a range of molecules from 4 to 24 qubits. It first shows the properties of the QLM simulator, including the timings for applying the UCCSD ansatz and measuring the Hamiltonian expectation value. It then demonstrates the use of active space selection and MP2 pre-screening initial guesses for different test molecules. Using the ADAPT-VQE module, it compares the fermionic and qubit-ADAPT-VQE results in terms of chemical accuracy, number of variational parameters, operators, and quantum gates. Finally, it compares the "fixed-length" UCC methods to the ADAPT-VQE approach, showing that ADAPT-VQE can achieve higher accuracy with fewer parameters and gates, depending on the chosen convergence threshold.
