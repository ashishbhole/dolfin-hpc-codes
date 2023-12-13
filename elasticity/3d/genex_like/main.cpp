// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// This demo program solves the equations of static
// linear elasticity for a gear clamped at two of its
// ends and twisted 30 degrees.

#define IO

#include "Elasticity.h"
#include <dolfin.h>

using namespace dolfin;

real tstep =  0.25 * 1e-6; // based on the frquency of excitation
real Tfinal = 0.15 * 1e-3;

struct ConstantFunction : public Value< ConstantFunction, 1, 3 >
{
  ConstantFunction( real c_ )
    : c( c_ )
  {
  }

  void eval( real * values, const real * x ) const
  {
    values[0] = c;
    values[1] = c;
    values[2] = c;
  }

  real c;
};

// Dirichlet boundary condition for clamp at left end
struct Source : public Value< Source, 3 >
{
  Source(): t(0), freq(400000.0) {}
  void eval( real * values, const real * x ) const
  {
    if( fabs(x[2]) < DOLFIN_EPS && ( sqrt(x[0]*x[0]+x[1]*x[1]) <= (0.0036775 + fabs(DOLFIN_EPS)) ) ) 
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 1e3*sin(2.0*freq*DOLFIN_PI*t);
    }
    else
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
    }
  }
  double t, freq;  
};

struct DirichletFunction : public Value<DirichletFunction, 3>
{
  void eval(real *value, const real *x) const
  {
    value[0] = 0.0;
    value[1] = 0.0;
    value[2] = 0.0;
  }
};

struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

int main()
{
  dolfin_init();

  // Read mesh
  Mesh mesh("./disc_in_disc_3d.bin");

  Analytic<DirichletFunction> u0(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);
  
  double t = 0.0;
  // Set elasticity parameters
  real const                   density=2780.0; // 100.0; // 2780.0;
  real const                   E  = 7.31e10; // 10.0; // 7.31e10;
  real const                   nu = 0.33;
  ConstantFunction             mu_( E / ( 2 * ( 1 + nu ) ) );
  Analytic< ConstantFunction > mu( mesh, mu_ );
  ConstantFunction lambda_( E * nu / ( ( 1 + nu ) * ( 1 - 2 * nu ) ) );
  Analytic< ConstantFunction > lambda( mesh, lambda_ );

  Source src;
  Analytic<Source> f( mesh, src);
  
  Constant dt(tstep);
  Constant rho(density );

  // Set up PDE
  Elasticity::BilinearForm a( mesh, dt, rho, mu, lambda );
  Function u(a.trial_space());
  Function un(a.trial_space());
  Function u_old(a.trial_space());
  Function ff(a.trial_space());
  Elasticity::LinearForm L( mesh, un, u_old, f, dt, rho );

  // Solve PDE
  Matrix A;
  Vector b;
  a.assemble(A, true);
  KrylovSolver solver(bicgstab, bjacobi);  

  // Save solution to VTK format
  File file( "elasticity.pvd" ); 
  uint step = 0;
  while (t < Tfinal)
  {
    if (t+tstep > Tfinal) dt = Tfinal - t;
    src.t=t;
    Analytic<Source> f( mesh, src);
    Elasticity::LinearForm L(mesh, un, u_old, f, dt, rho);
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u_old = un;
    un = u;
    #ifdef IO
    if (step%100 == 0) file << u;
    #endif
    message( "no_ot_tstep, tstep, t, l2 norm: %d %e %e %e", step, tstep, t, u.vector().norm(l2) );
    t +=tstep;
    step += 1;
  }
  dolfin_finalize();
  return 0;
}
