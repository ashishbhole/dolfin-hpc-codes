// Copyright (C) 2010 Ashish Bhole.
// Licensed under the GNU LGPL Version 2.1.

#define IO

#include "KPP.h"
#include "Projection.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 1e-4;
real Tfinal  = 1.0;

struct InitialCondition : public Value< InitialCondition, 1 >
{
  void eval( real * values, const real * x ) const
  {
    if(sqrt(x[0]*x[0] + x[1]*x[1]) <= 1.0)
    {
      values[0] = 3.5*DOLFIN_PI;
    }
    else 
    {
      values[0] = 0.25*DOLFIN_PI;
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

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  // Parallel file writing does not work with in-built meshes.
  Mesh mesh("kpp.bin");

  // Set boundary conditions
  DirichletBoundary boundary;

  InitialCondition int_con;
  Constant          ub( 0.25*DOLFIN_PI );
  DirichletBC bc(ub, mesh, boundary);

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
  KPP::BilinearForm a(mesh);
  Function u(a.trial_space());
  KPP::LinearForm L(mesh, u0, dt, tau, tau_sc_perp, tau_sc_par);
  Matrix A;
  Vector b;
  a.assemble(A, true);

  uint step = 0;

  // File to write the numerical solution
  File file("solution.pvd", t);
  
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
    solver.solve(A, u.vector(), b);
    u.sync();
    u0 = u;
    t +=tstep;
    step += 1;
    message( "t, l2 norm: %e %e", t, u0.vector().norm(l2) );
    
    #ifdef IO
    if (step%200 == 0) file << u0; 
    #endif
  }

  dolfin_finalize();
  return 0;
}
