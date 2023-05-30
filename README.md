

# GMRES Solving Poisson's Equation

## What's this project about?

Welcome to the project "GMRES Solving Poisson's Equation"! The project intends to solve a 3D poisson's equation by the generalized minimal residual method (GMRES). So far a sequential algorithm of GMRES has just been developed. The program structure follows the idea of the [homework](https://gitlab.tudelft.nl/dhpc/sticse-hpc/homework1) of the course WI4450 held by Jonas and Martin from TU Delft. 
This project is considered as a pioneer of the project "mpi-gmres-analysis", which tries to solve a poisson's equation with convection term through GMRES in parallel (OpenMPI). If you are interested, please jump to https://github.com/MengZhaonan1998/mpi-gmres-analysis.git

## Trying it out
Simply type "make" and you will get a GMRES algorithm solving a poisson's equation. Note that the matrix-vector multiplication is implemented matrix-free. 
