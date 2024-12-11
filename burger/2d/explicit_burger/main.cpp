// Copyright (C) 2010 Ashish Bhole.
// Licensed under the GNU LGPL Version 2.1.

#define IO

#include "Burger.h"
#include "Projection.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 1e-4;
real Tfinal  = 0.5;

// Analytic function to specify the initial condition and exact solution.
struct AnalyticFunction : public Value< AnalyticFunction, 1 >
{
  // Constructor
  AnalyticFunction() : t(0.0) {}	
  void eval( real * values, const real * x ) const
  {
    if(x[0] <= 0.5-3.0*t/5.0)
    {
      if(x[1] >= 0.5+3.0*t/20.0)
      {
        values[0] = -0.2;
      }
      else
      {
        values[0] = 0.5;
      }
    }
    else if(0.5-3.0*t/5.0 < x[0] && 0.5 - 0.25*t >= x[0])
    {
      if(x[1] >= -8.0*x[0]/7.0 + 15.0/14.0 - 15.0*t/28.0)
      {
        values[0] = -1.0;
      }
      else
      {
        values[0] = 0.5;
      }
    }
    else if(0.5 - 0.25*t < x[0] && 0.5 + 0.5*t >= x[0])
    {
      if(x[1] >= x[0]/6.0 + 5.0/12.0 - 5.0*t/24.0)
      {
        values[0] = -1.0;
      }
      else
      {
        values[0] = 0.5;
      }
    }
    else if(0.5 + 0.5*t < x[0] && 0.5 + 4.0*t/5.0 >= x[0])
    {
      if(x[1] >= x[0] - 5.0*(x[0] + t - 0.5)*(x[0] + t - 0.5)/(18.0*t))
      {
        values[0] = -1.0;
      }
      else
      {
        values[0] = (2.0*x[0]-1.0)/(2.0*t);
      }
    }
    else if(x[0] > 0.5 + 4.0*t/5.0)
    {
      if(x[1] >= 0.5-0.1*t)
      {
        values[0] = -1.0;
      }
      else
      {
        values[0] = 0.8;
      }
    }
  }
  // Current time
  double t;
};

struct InitialCondition : public Value< InitialCondition, 1 >
{
  void eval( real * values, const real * x ) const
  {
    if(x[0] <= 0.5 && x[1] < 0.5)
    {
      values[0] = 0.5;
    }
    else if(x[0] <= 0.5 && x[1] >= 0.5) 
    {
      values[0] = -0.2;
    }
    else if(x[0] > 0.5 && x[1] >= 0.5)
    {
      values[0] = -1.0;
    }
    else 
    {
      values[0] = 0.8;
    }
	    
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
  UnitSquare mesh(100, 100);

  // Set boundary conditions
  AnalyticFunction exact_sol;
  DirichletBoundary boundary;

  InitialCondition int_con;

  // For initial conditions 
  double t = 0.0;
  Analytic<InitialCondition> ui( mesh, int_con);

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
  double h = MeshQuality(mesh).h_max;

  // some quantities in form files
  Constant dt(tstep);
  Constant tau(1e-1*tstep);
  Constant tau_sc_perp(1e-6*h*h*h);
  Constant tau_sc_par(1e-8*h*h*h);

  // See the declaration in the header file
  Burger::BilinearForm a(mesh);
  Function u(a.trial_space());
  Burger::LinearForm L(mesh, u0, dt, tau, tau_sc_perp, tau_sc_par);
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
    exact_sol.t = t;
    Analytic<AnalyticFunction> ub(mesh, exact_sol);
    DirichletBC bc(ub, mesh, boundary);
    
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u0 = u;
    t +=tstep;
    step += 1;
    message( "t, l2 norm: %e %e", t, u0.vector().norm(l2) );
    
    #ifdef IO
    if (step%100 == 0) file << u0; 
    #endif
  }

  dolfin_finalize();
  return 0;
}
