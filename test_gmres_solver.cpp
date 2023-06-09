#include "gtest_mpi.hpp"
#include "operations.hpp"
#include "gmres_solver.hpp"

#include <iostream>
#include <cmath>
#include <limits>

TEST(gmres_solver, gmres_solver)
{
/*This is a test for cg_solver which solves the linear system Ax=b
  Here we use a diagonal matrix as A and all-one vector as b, by which 
  we know the correct solution easily.*/
  const int nx=3, ny=3, nz=3;
  const int n=nx*ny*nz;
  
  stencil3d S;
  S.nx=nx; S.ny=ny; S.nz=nz;
  S.value_c = 2;
  S.value_n = 1;
  S.value_e = 1;
  S.value_s = 1;
  S.value_w = 1;
  S.value_b = 1;
  S.value_t = 1;

  double *x = new double[n]; // solution vector x
  double *b = new double[n]; // right hand side vector b
  double *r = new double[n]; // residual r=Ax-b

  init(n, x, 0.0); // solution starts with [0,0,...]
  init(n, b, 1.0); // right hand side b=[1,1,...] 

  // solve the linear system of equations using CG
  int numIter, maxIter=10;
  double resNorm, tol=std::sqrt(std::numeric_limits<double>::epsilon());

  gmres_solver(&S, n, x, b, tol, maxIter, &resNorm, &numIter, 0);

  apply_stencil3d(&S, x, r); // r = op * x
  axpby(n, 1.0, b, -1.0, r); // r = b - r

  double err=std::sqrt(dot(n, r, r))/std::sqrt(dot(n,b,b));
  EXPECT_NEAR(1.0+err, 1.0, 10*std::numeric_limits<double>::epsilon());
  
  delete [] x;
  delete [] b;
  delete [] r;

}

