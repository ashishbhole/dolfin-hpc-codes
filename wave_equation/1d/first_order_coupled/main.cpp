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

#include "WaveEquation.h"
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
    real xcl = x[0]-0.5-speed_x*t;
    real xcr = x[0]-0.5+speed_x*t;
    values[0] = 0.5*exp(-alpha*( xcl*xcl )) + 0.5*exp(-alpha*( xcr*xcr ));
  }
  double t, alpha, speed_x;
};

struct ExactVelocity : public Value< ExactVelocity, 1 >
{
  ExactVelocity() : t(0), alpha(32.0), speed_x(1.0) {}
  void eval( real * values, const real * x ) const
  {
    real xcl = x[0]-0.5-speed_x*t;
    real xcr = x[0]-0.5+speed_x*t;
    values[0] = - exp(-alpha*( xcl*xcl )) * (-alpha*xcl) * speed_x \
	        + exp(-alpha*( xcr*xcr )) * (-alpha*xcr) * speed_x;
  }
  double t, alpha, speed_x;
};

struct DirichletFunction : public Value<DirichletFunction, 2>
{
  void eval(real *value, const real *x) const
  {
    value[0] = 0.0;
    value[1] = 0.0;
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
File file1("displacement.pvd");
File file2("velocity.pvd");

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

  ExactVelocity Velocity;
  Velocity.alpha = 128.0;
  Velocity.speed_x = speed_x;
  Velocity.t = t;
  Analytic<ExactVelocity> vi( mesh, Velocity);
  
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

  Projection::BilinearForm a2(mesh);
  Projection::LinearForm L2(mesh, vi);
  Matrix A2;
  Vector b2;
  a2.assemble(A2, true);
  L2.assemble(b2, true);
  Function v0(a2.trial_space());
  solver.solve(A2, v0.vector(), b2);
  v0.sync();
  
  #ifdef IO
  file1 << u0;
  file2 << v0;
  #endif
  
  real Nc = 0.05;
  //real h = MeshQuality(mesh).h_max; // does not seem to work for 1d meshes
  real h = 1.0 / Np;
  tstep = Nc * h / speed_x;

  Constant dt(tstep);
  Constant speed(speed_x);

  WaveEquation::BilinearForm a(mesh, speed, dt);
  Function w(a.trial_space());
  Function u, v;
  WaveEquation::LinearForm L(mesh, u0, v0);
  Matrix A;
  Vector b;
  a.assemble(A, true);
  KrylovSolver solver1(bicgstab, bjacobi);

  uint step = 0;

  while (t < Tfinal)
  {
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver1.solve(A, w.vector(), b);
    w.sync();
    u = SubFunction(w, 0);
    v = SubFunction(w, 1);
    u0 = u;
    v0 = v;
    #ifdef IO
    if (step%10 == 0) 
    {
      file1 << u;
      file2 << v;
    }
    #endif
    t +=tstep;
    step += 1;
  }
  // Get the exact solution at Tfinal
  Gaussian.t = Tfinal;
  Analytic<ExactSolution> uex( mesh, Gaussian);
  Projection::LinearForm L3(mesh, uex);
  Vector b3;
  L3.assemble(b3, true);
  solver.solve(A1, u0.vector(), b3);
  u0.sync();

  // Compute the numerical error in u1 as : u1 = u1 - ue
  u -= u0;
  message( "h, Error l1, l2, linf norm: %e %e %e %e", h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );

  dolfin_finalize();
  return 0;
}
