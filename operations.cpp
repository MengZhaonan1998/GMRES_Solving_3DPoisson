#include "operations.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

void init(int n, double* x, double value)
{
  
  #pragma omp parallel for 
  for (int i=0; i<n; i++)
     x[i] = value;   // assign value to every entry of x
  
  return;
}

double dot(int n, double const* x, double const* y)
{
  double res=0.0;
  
  #pragma omp parallel for reduction(+:res)   // I use reduction(+:res) here due to the operation +=
  for (int i=0; i<n; i++)
     res += x[i]*y[i];
  
  return res;
  
}

void axpby(int n, double a, double const* x, double b, double* y)
{
  #pragma omp parallel for
  for (int i=0; i<n; i++)
     y[i] = (a * x[i] + b * y[i]);
  return;
}

// apply a 7-point stencil to a vector
void apply_stencil3d(stencil3d const* S,
        double const* u, double* v)
{
  int nx=S->nx, ny=S->ny, nz=S->nz;  // access the number of points in x, y, z coordinate
  double ele;                        // I use ele to store the value of v[S->index_c(i,j,k)]

  #pragma omp parallel for reduction(+:ele)
  for (int k=0; k<nz; k++)
    for (int j=0; j<ny; j++)
        for(int i=0; i<nx; i++)
	{
	  ele = S->value_c * u[S->index_c(i,j,k)] +
	        S->value_e * u[S->index_e( i-(i==(nx-1)), j, k)] * (i < nx - 1) +
		S->value_w * u[S->index_w( i+(i==0), j, k)] * (i > 0)  +
		S->value_n * u[S->index_n( i, j-(j==(ny-1)), k)] * (j < ny - 1) +
		S->value_s * u[S->index_s( i, j+(j==0), k)] * (j > 0)  +
	        S->value_t * u[S->index_t( i, j, k-(k==(nz-1)))] * (k < nz - 1) +
		S->value_b * u[S->index_b( i, j, k+(k==0))] * (k > 0);	 
	  v[S->index_c(i,j,k)] = ele;  
	}
  
  return;
}

// apply given rotation
void given_rotation(int k, double* h, double* cs, double* sn)
{
  double temp, t, cs_k, sn_k;
  for (int i=0; i<k; i++)
  {
     temp = cs[i] * h[i] + sn[i] * h[i+1];
     h[i+1] = -sn[i] * h[i] + cs[i] * h[i+1];
     h[i] = temp;
  }
  
  // update the next sin cos values for rotation
  t = std::sqrt( h[k]*h[k] + h[k+1]*h[k+1] );
  cs[k] = h[k]/t;
  sn[k] = h[k+1]/t;

  // eliminate H(i+1,i)
  h[k] = cs[k]*h[k] + sn[k]*h[k+1];
  h[k+1] = 0.0;

  return;
}

// Arnoldi function
void arnoldi(int k, double* Q, double* h, stencil3d const* op) 
{
  int n = op->nx * op->ny * op->nz;
  apply_stencil3d(op, Q+k*n, Q+(k+1)*n);

  for (int i=0; i<=k; i++)
  {
    h[i] = dot(n, Q+(k+1)*n, Q+i*n);
    axpby(n, -h[i], Q+i*n, 1.0, Q+(k+1)*n);
  }

  h[k+1] = std::sqrt(dot(n, Q+(k+1)*n, Q+(k+1)*n));

  for (int i=0; i<n; i++)
    Q[(k+1)*n+i] = Q[(k+1)*n+i] / h[k+1];

 return; 
}





















