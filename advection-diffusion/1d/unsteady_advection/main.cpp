// Copyright (C) 2010 Jeannette Spuehler.
// Licensed under the GNU LGPL Version 2.1.
//
// Written by Ashish Bhole 2023: 
//
// This is an IVP tp solve linear advection equation.
// The initial condition is specified as a Gaussian function.
// It advects with constant speed.
//
//     u_t + \mathbb{c} \cdot \nabla u = 0
//     u(x, 0) = exp(-alpha*( (x-x0)**2 ) )
//     u_ex(x ,t) = exp(-alpha*( (x-adv_x*t-x0)**2 ))

#define IO

#include "AdvectionDiffusion.h"
#include "Projection.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.01;
real speed_x = 0.2;
real Tfinal  = 0.5;
real Np = 100;

struct ExactSolution : public Value< ExactSolution, 1 >
{
  ExactSolution() : t(0), alpha(32.0), speed_x(1.0) {}	
  void eval( real * values, const real * x ) const
  {
    real xc = speed_x*t;
    values[0] = exp(-alpha*( (x[0]-xc-0.25)*(x[0]-xc-0.25) ));
  }
  double t, alpha, speed_x;
};

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

  UnitInterval mesh(Np);

  // Set boundary conditions
  Analytic<DirichletFunction> ub(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(ub, mesh, boundary);

  // For initial conditions 
  real t = 0.0;
  ExactSolution Gaussian;
  Gaussian.alpha = 128.0;
  Gaussian.speed_x = speed_x;
  Gaussian.t = t;
  Analytic<ExactSolution> ui( mesh, Gaussian);

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

  real Nc = 0.05;
  //real h = MeshQuality(mesh).h_max; // does not seem to work for 1d meshes
  real h = 1.0 / Np;
  tstep = Nc * h / speed_x;

  Constant dt(tstep);
  Constant speed(speed_x);
  real tau_val = 0.0*h/speed_x;
  Constant tau(tau_val);

  AdvectionDiffusion::BilinearForm a(mesh, speed, tau, dt);
  Function u1(a.trial_space());
  AdvectionDiffusion::LinearForm L(mesh, u0);
  Matrix A;
  Vector b;
  a.assemble(A, true);

  uint step = 0;

  #ifdef IO
  file << u0;
  #endif

  while (t < Tfinal)
  {
    // Adjust dt to reach final time exactly
    if (t+tstep > Tfinal) dt = Tfinal - t;
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u1.vector(), b);
    u1.sync();
    u0 = u1;
    t +=tstep;
    step += 1;
    #ifdef IO
    if (step%10 == 0) file << u1; 
    #endif
  }
  // Get the exact solution at Tfinal
  Gaussian.t = Tfinal;
  Analytic<ExactSolution> uex( mesh, Gaussian);
  Projection::LinearForm L2(mesh, uex);
  Vector b2;
  L2.assemble(b2, true);
  solver.solve(A1, u0.vector(), b2);
  u0.sync();

  file << u0;

  // Compute the numerical error in u1 as : u1 = u1 - ue
  u1 -= u0;
  message( "h, Error l1, l2, linf norm: %e %e %e %e", h, u1.vector().norm(l1), u1.vector().norm(l2), u1.vector().norm(linf) );

  dolfin_finalize();
  return 0;
}
