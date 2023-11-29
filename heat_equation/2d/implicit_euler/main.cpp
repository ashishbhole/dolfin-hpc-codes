// Copyright (C) 2010 Jeannette Spuehler.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ashish Bhole 2023: 
//
// This is a solver for Cauchy proble for the linear heat equation.
//
//     u_t = div( nu * grad(u)) on D
//     u(0, x, y) = f(x,y)
//
// 'Adhoc' homogeneous Dirichelt boundary conditions are specified
// assuming that simulation is not intended for long time.

#define IO

#include "HeatEquation.h"
#include "Projection.h"
#include <sstream>
#include <dolfin.h>

using namespace dolfin;

double tstep = 0.01;
double nu_val = 0.001;
double Tfinal = 2.0;

// Analytic function to specify the initial condition.
struct InitialCondition : public Value< InitialCondition, 1 >
{
  // Constructor
  InitialCondition() : alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    values[0] = exp(-alpha*( (x[0]-0.5)*(x[0]-0.5) + (x[1]-0.5)*(x[1]-0.5) ));
  }
  // Current time
  double alpha;
};

struct DirichletFunction : public Value<DirichletFunction, 1>
{
  void eval(real *value, const real *x) const
  {
    value[0] = 0.0;
  }
};

struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

File file("solution.pvd");

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  // Parallel file writing does not work with in-built meshes.
  //Mesh mesh("../../../advection-diffusion/2d/explicit-euler/mesh2D.bin");
  UnitSquare mesh(100,100);

  Analytic<DirichletFunction> ub(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(ub, mesh, boundary);

  double t = 0.0;
  InitialCondition Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<InitialCondition> ui( mesh, Gaussian);

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

  // Estimate the time step based on Peclet number
  double Pe = 0.05;
  double h = MeshQuality(mesh).h_max;
  tstep = Pe*h*h/nu_val;

  Constant dt(tstep);
  Constant nu(nu_val);

  HeatEquation::BilinearForm a(mesh, nu, dt);
  Function u(a.trial_space());
  #ifdef IO
  file << u0;
  #endif
  HeatEquation::LinearForm L(mesh, u0);
  Matrix A;
  Vector b;
  a.assemble(A, true);
  uint step = 0;
  while (t < Tfinal)
  {
    if (t+tstep > Tfinal) dt = Tfinal - t;
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    #ifdef IO
    if (step%10 == 0) file << u; 
    #endif
    u0 = u;
    t +=tstep;
    step += 1;
  }
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
