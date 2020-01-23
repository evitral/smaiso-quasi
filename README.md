# smaiso-quasi
The purpose of the smaiso-dft project is to study the evolution of an interface between a smectic-A liquid crystal and a disordered phase, both displaying different densities. The present files contain an implementation of the models proposed in:

(1) E. Vitral, P.H. Leo, J. Viñals, Physical Review E 100.3 (2019)

(2) E. Vitral, P.H. Leo, J. Viñals, in preparation

These are custom C++ codes based on the parallel FFTW library and the standard MPI passing interface. The present smaiso-quasi repository belong to Part C of project. Part A was the initial study of a smectic-A in contat with its isotropic phase of same density, describing a diffusive evolution of the interface without advection, whose results are found in Ref. (1) and repository in https://doi.org/10.5281/zenodo.3626204. Part B is similar to A, but introduces advection to a uniform density system. Part C contains implementations of a quasi-incompressible smectic-isotropic system, presenting a varying density field between phases. The derivation, discussion and numerical results associated to this model will be published in Ref. (2).

## C. quasi: smectic-iso, 

C1. quasi.cpp : smectic-iso, quasi-incompressible model, varying density

C2. quasi-stb.cpp : stability analysis
