#include "timer.hpp"
#include "operations.hpp"
#include "gmres_solver.hpp"

#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>

void gmres_solver(stencil3d const* op, int n, double* x, double const* b,
        double tol, int maxIter,
        double* resNorm, int* numIter,
        int verbose)
{
  if (op->nx * op->ny * op->nz != n)
  {
    throw std::runtime_error("mismatch between stencil and vector dimension passed to gmres_solver");
  }

  double* r = new double[n]; // residual vector
  double* sn = new double[maxIter]; // used in given rotation
  double* cs = new double[maxIter]; // used in given rotation
  double* e1 = new double[maxIter+1];
  double* beta = new double[maxIter+1];

  double* Q = new double[(maxIter+1) * n];                    // note that Q is stored by column major
  double* H = new double[((maxIter+1) * maxIter)/2+maxIter];  // note that H is stored by column major

  double r_norm;             // residual norm
  double b_norm;             // right hand side b norm

  // r=b-A*x
  apply_stencil3d(op, x, r); // r = op * x
  axpby(n, 1.0, b, -1.0, r); // r = b - r

  // compute the error
  r_norm = std::sqrt(dot(n,r,r));
  b_norm = std::sqrt(dot(n,b,b));
  double error = r_norm/b_norm;

  // initialize the 1D vectors
  init(maxIter, sn, 0.0); 
  init(maxIter, cs, 0.0);
  init(maxIter+1, e1, 0.0);
  init(maxIter+1, beta, 0.0);
  e1[0]=1.0;
  beta[0]=r_norm;

  // initialize Q and H;
  init((maxIter+1) * n, Q, 0.0);
  init(((maxIter+1) * maxIter)/2+maxIter, H, 0.0);
  for (int i=0;i<n;i++) Q[i]=r[i]/r_norm;  


  // start CG iteration
  int iter = -1;
  while (true)
  {
    iter++;

    if (verbose)
    {
      std::cout << std::setw(4) << iter << "\t" << std::setw(8) << std::setprecision(4) << error << std::endl;
    }

    // check for convergence or failure
    if ( (error < tol) || (iter == maxIter) )
    {
      break;
    }

    arnoldi(iter, Q, H + iter*(iter+1)/2+iter, op); 

    given_rotation(iter, H + iter*(iter+1)/2+iter, cs, sn);

    beta[iter+1] = -sn[iter]*beta[iter];
    beta[iter] = cs[iter]*beta[iter];
    error = std::abs( beta[iter+1] ) / b_norm;

  } // end of while-loop

  // backward substitution
  double* y = new double[iter];
  init(iter, y, 0.0);
  for (int i=0; i<iter; i++) 
  {
     for (int j=0; j<i; j++)
     {  
        beta[iter-1-i] -=  y[iter-1-j] * H[((iter-j+1) * (iter-j))/2 +iter-j-2-(i-j)]; 
     }
     y[iter-1-i] = beta[iter-1-i] / H[((iter-i+1) * (iter-i))/2+iter-i-2]; 
  }

  // x = x + Q*y
  for (int i=0; i<iter; i++) axpby(n, y[i], Q+i*n, 1.0, x);

  // clean up
  delete [] r;
  delete [] sn;
  delete [] cs;
  delete [] e1;
  delete [] Q;
  delete [] H;
  delete [] y;

  // return number of iterations and achieved residual
  *resNorm = beta[iter];
  *numIter = iter;

  return;
}
