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

#include "AdvectionDiffusion.h"
#include "Projection.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.01;
real speed_x = 0.2;
real speed_y = 0.2;
real Tfinal  = 1.0;

// Analytic function to specify the initial condition and exact solution.
struct ExactSolution : public Value< ExactSolution, 1 >
{
  // Constructor
  ExactSolution() : t(0), alpha(32.0), speed_x(1.0), speed_y(1.0) {}	
  void eval( real * values, const real * x ) const
  {
    real xc = speed_x*t;
    real yc = speed_y*t;
    values[0] = exp(-alpha*( (x[0]-xc-0.25)*(x[0]-xc-0.25) + (x[1]-yc-0.25)*(x[1]-yc-0.25) ));
  }
  // Current time
  double t, alpha, speed_x, speed_y;
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
  //UnitSquare mesh(100, 100);
  //
  // Exporting mesh for parallel file writing. Furthermore, refinement
  // looks buggy. It is better to export meshes (in bin format).
  Mesh mesh("UnitSquareMesh_32x32.bin");

  // Set boundary conditions
  Analytic<DirichletFunction> ub(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(ub, mesh, boundary);

  // For initial conditions 
  double t = 0.0;
  ExactSolution Gaussian;
  Gaussian.alpha = 128.0;
  Gaussian.speed_x = speed_x;
  Gaussian.speed_y = speed_y;
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
  
  // time step computation
  double Nc = 0.05;
  double h = MeshQuality(mesh).h_max;
  tstep = Nc * h / (sqrt(pow(speed_x,2)+pow(speed_y,2)));

  // some quantities in form files
  Constant dt(tstep);
  Constant adv_x(speed_x);
  Constant adv_y(speed_y);
  Constant tau(0.0*h/(sqrt(pow(speed_x,2)+pow(speed_y,2))));

  // See the declaration in the header file
  AdvectionDiffusion::BilinearForm a(mesh, adv_x, adv_y, tau, dt);
  Function u1(a.trial_space());
  AdvectionDiffusion::LinearForm L(mesh, u0);
  Matrix A;
  Vector b;
  a.assemble(A, true);

  uint step = 0;

  // write the initial condition to the solution file
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
