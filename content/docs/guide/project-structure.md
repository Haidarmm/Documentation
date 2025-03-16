---
title: Package Structure
weight: 1
---

## Folder Structure

OpenVQE consists of **two main modules that are inside the "openvqe" folder:**:

- [<span style="color:red">UCC Family</span>](https://openvqe.github.io/OpenVQE/_build/html/ucc_family.html)
: This module includes different classes and functions to generate the fermionic cluster operators (fermionic pool) and the qubit pools, and to get the VQE optimized energies in the cases of active and non-active orbital selections.
- [<span style="color:red">adapt</span>](http://localhost:1313/docs/guide/project-structure/) includes two sub-modules:
:
  - `Fermionic-ADAPT/`: containing functions performing the fermionic-ADAPT-VQE algorithmic steps in the active and non-active space selections;
  - `Qubit-ADAPT/`: containing functions that perform the qubit-ADAPT-VQE algorithmic steps calculation in the active and non-active space orbital selections.

## SubFolder Structure


- [<span style="color:red">common_files</span>](https://openvqe.github.io/OpenVQE/_build/html/common_files.html)
: stores all the internal functions needed to be imported for executing the two modules.
- `notebooks`: allows the user to run and test the above two modules: `UCC Family/` and `adapt/`.

![image](/uploads/output.jpg)


## Updated second version package
1. OpenVQE main diagram showing the <span style="color:red">folder structure</span> of the package
   
![image](/uploads/sketch_.png)

1. OpenVQE code diagram showing the <span style="color:red"> main structure</span> of the code and their involvement when executing VQE.
  
![image](/uploads/sketch.png) 

