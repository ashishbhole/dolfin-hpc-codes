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

real tstep = 0.0001; //1.0/5000.0;
real Tfinal = 5.0;

struct Null : public Value< Null, 3 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.;
    values[1] = 0.;
    values[2] = 0.;
  }
};

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
struct Harmonic : public Value< Harmonic, 3 >
{
  Harmonic(): t(0), freq(2.0) {}
  void eval( real * values, const real * x ) const
  {
    if(x[2] == 0.0 && ( sqrt(x[0]*x[0]+x[1]*x[1]) <= 0.0036775 ))
    {
      values[0] = 0.0;
      values[1] = 0.0;
      values[2] = 0.0;
      if(t<0.15) values[2] = 1e-4*sin((freq/0.15)*DOLFIN_PI*t);
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
  Mesh mesh("./square_in_disc_3d.bin");
  // Create right-hand side
  Analytic< Null > f( mesh );

  // Set up boundary condition at left end
  Harmonic signal; 
  DirichletBoundary boundary;

  double t = 0.0;
  // Set elasticity parameters
  real const                   density=100.0; // 2780.0;
  real const                   E  = 10.0; // 7.31e10;
  real const                   nu = 0.33;
  ConstantFunction             mu_( E / ( 2 * ( 1 + nu ) ) );
  Analytic< ConstantFunction > mu( mesh, mu_ );
  ConstantFunction lambda_( E * nu / ( ( 1 + nu ) * ( 1 - 2 * nu ) ) );
  Analytic< ConstantFunction > lambda( mesh, lambda_ );

  Constant dt(tstep);
  Constant rho(density );

  // Set up PDE
  Elasticity::BilinearForm a( mesh, dt, rho, mu, lambda );
  Function u(a.trial_space());
  Function un(a.trial_space());
  Function u_old(a.trial_space());
  Elasticity::LinearForm   L( mesh, un, u_old, f, dt, rho );

  // Solve PDE
  Matrix A;
  Vector b;
  a.assemble( A, true );
  L.assemble( b, true );
  //LinearSolver solver;
  KrylovSolver solver(bicgstab, bjacobi);  
  solver.solve( A, u.vector(), b );

  // Save solution to VTK format
  File file( "elasticity.pvd" ); 
  file << u;

  uint step = 0;
  while (t < Tfinal)
  {
    if (t+tstep > Tfinal) dt = Tfinal - t;
    signal.t=t;
    Analytic<Harmonic> sense_signal( mesh, signal);
    DirichletBC       bc( sense_signal, mesh, boundary );
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u_old = un;
    un = u;
    #ifdef IO
    if (step%100 == 0) file << u;
    #endif
    t +=tstep;
    step += 1;
  }
  dolfin_finalize();
  return 0;
}
