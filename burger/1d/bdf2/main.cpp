// Copyright (C) 2010 Jeannette Spuehler.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2017.
// Modified by Ashish Bhole 2023: 
// Modified from ConvectionDiffusion demo
//
// This is an IVP tp solve linear advection equation.
// The initial condition is specified as a Gaussian function.
// It advects with constant velocity vector = {adv_x, adv_y}.
//
//     u_t + \mathbb{c} \cdot \nabla u = 0
//     u(x, 0) = exp(-alpha*( (x-x0)**2 + (y-y0)**2 ))
//     u_ex(x ,t) = exp(-alpha*( (x-adv_x*t-x0)**2 + (y-adv_y*t-y0)**2 ))

#define IO

#include "BurgerCN.h"
#include "BurgerBDF2.h"
#include "Projection.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.0001;
real Tfinal  = 1.0;

// Analytic function to specify the initial condition and exact solution.
struct AnalyticFunction : public Value< AnalyticFunction, 1 >
{
  // Constructor
  AnalyticFunction() : t(0), alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    values[0] = sin(2.0*DOLFIN_PI*x[0]);
  }
  // Current time
  double t, alpha;
};

// can be simplified for 0.0 boundary condition. kept for generality.
struct DirichletFunction : public Value<DirichletFunction, 1>
{
  void eval(real *value, const real *x) const
  {
    value[0] = 0.0;
  }
};

// identify boundaries
struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

// File to write the numerical solution
File file("solution.pvd");

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  // Parallel file writing does not work with in-built meshes.
  UnitInterval mesh(1000);

  // Set boundary conditions
  Analytic<DirichletFunction> ub(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(ub, mesh, boundary);

  // For initial conditions 
  double t = 0.0;
  AnalyticFunction Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<AnalyticFunction> ui( mesh, Gaussian);

  // Note that interpolation operation u0<<ui only works for degree=1.
  // Hence projection of the analytic function on FE space is used as
  // the initial condition.
  Projection::BilinearForm a1(mesh);
  Projection::LinearForm L1(mesh, ui);
  Matrix A1;
  Vector b1;
  a1.assemble(A1, true);
  L1.assemble(b1, true);
  Function u0(a1.trial_space());
  KrylovSolver solver(bicgstab, bjacobi);
  solver.solve(A1, u0.vector(), b1);
  u0.sync();
  
  // some quantities in form files
  Constant dt(tstep);

  // Apply Crank-Nicholson at the first time step
  BurgerCN::BilinearForm acn(mesh, u0, dt);
  Function du0(acn.trial_space());
  BurgerCN::LinearForm Lcn(mesh, u0, dt);
  Matrix Acn;
  Vector bcn;
  acn.assemble(Acn, true);
  Lcn.assemble(bcn, true);
  bc.apply(Acn, bcn, acn);
  solver.solve(Acn, du0.vector(), bcn);
  du0.sync();
  u0 += du0;
  t +=tstep;

  // Apply BDF2 later
  uint step = 1;
  BurgerBDF2::BilinearForm a(mesh, u0, dt);
  Function du(a.trial_space());
  BurgerBDF2::LinearForm L(mesh, u0, du0, dt);
  Matrix A;
  Vector b;
  a.assemble(A, true);

  // write the initial condition to the solution file
  #ifdef IO
  file << u0;
  #endif
  
  while (t < Tfinal)
  {
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, du.vector(), b);
    du.sync();
    u0 += du;
    t +=tstep;
    step += 1;
    #ifdef IO
    if (step%100 == 0) file << u0; 
    #endif
  }
  message( "Solution norm l1, l2, linf: %e %e %e", u0.vector().norm(l1), u0.vector().norm(l2), u0.vector().norm(linf) );

  dolfin_finalize();
  return 0;
}
