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
  //uint Np = pow(2,num_refine)*32;
  //UnitSquare mesh(Np, Np);
  //
  // Exporting mesh for parallel file writing. Furthermore, refinement
  // looks buggy. It is better to export meshes (in bin format).
  Mesh mesh("UnitSquareMesh_32x32.bin");

  // Set boundary conditions
  Analytic<DirichletFunction> u0(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  // For initial conditions 
  double t = 0.0;
  ExactSolution Gaussian;
  Gaussian.alpha = 128.0;
  Gaussian.speed_x = speed_x;
  Gaussian.speed_y = speed_y;
  Gaussian.t = t;
  Analytic<ExactSolution> ui( mesh, Gaussian);

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
  Function un(a.trial_space());

  // interpolate the initial condition to initialize the problem
  un << ui;

  AdvectionDiffusion::LinearForm L(mesh, un);
  // declare and assemble FE Matrix and vector
  Matrix A;
  Vector b;
  a.assemble(A, true);

  // specify linear solver
  KrylovSolver solver(bicgstab, bjacobi);
  uint step = 0;

  // write the initial condition to the solution file
  #ifdef IO
  file << un;
  #endif
  
  while (t < Tfinal)
  {
    // Adjust dt to reach final time exactly
    if (t+tstep > Tfinal) dt = Tfinal - t;
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u1.vector(), b);
    u1.sync();
    un = u1;
    t +=tstep;
    step += 1;
    #ifdef IO
    if (step%10 == 0) file << u1; 
    #endif
  }
  // Get the exatc solution at Tfinal
  Gaussian.t = Tfinal;
  Analytic<ExactSolution> uex( mesh, Gaussian);
  // interpolate the exact solution to FE function
  Function ue(a.trial_space());
  ue << uex;
  // Compute the numerical error in u1 as : u1 = u1 - ue
  u1 -= ue;
  message( "h, Error l1, l2, linf norm: %e %e %e %e", h, u1.vector().norm(l1), u1.vector().norm(l2), u1.vector().norm(linf) );

  dolfin_finalize();
  return 0;
}
