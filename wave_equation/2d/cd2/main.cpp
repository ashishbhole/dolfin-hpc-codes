// Copyright (C) 2010 Jeannette Spuehler.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Ashish Bhole 2023: 
// Modified from ConvectionDiffusion demo
//
// This is a linear wave equation equation.
//
//     u_tt + c^2 div( grad(u)) = f

#define IO

//#include "InitWaveEquation.h"
#include "WaveEquation.h"
#include "Projection.h"
#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.01;
real speed = 1.0;
real Tfinal = 5.0;

real rec1[3] = {3.0, 3.0, 3.0};
real u_values[1] = {0.0};

// Analytic function to specify the initial condition and exact solution.
struct InitialCondition : public Value< InitialCondition, 1 >
{
  // Constructor
  InitialCondition() : alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.0;
  }
  // Current time
  double alpha;
};

struct Source : public Value< Source, 1 >
{
  Source(): t(0), alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    if(fabs(x[0]-0.5)<0.05 && fabs(x[1]-0.5)<0.05)
    {	    
      values[0] = sin(10*t); //exp(-alpha*((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5))) * exp(-alpha*t*t); //
    }
    else
    {
      values[0] = 0.0;
    }
  }
  double t, alpha;
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
  Mesh mesh("rectangular_struct_tria.bin");

  Analytic<DirichletFunction> u0(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  double t = 0.0;
  InitialCondition Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<InitialCondition> ui( mesh, Gaussian);

  Source src;

  double Nc = 0.05;
  double h = MeshQuality(mesh).h_max;
  tstep = Nc * h / speed;
  Constant dt(tstep);
  Constant c(speed);
  WaveEquation::BilinearForm a(mesh, c, dt);
  Function u(a.trial_space());
  Function un(a.trial_space());
  Function u_old(a.trial_space());
  Function ff(a.trial_space());
  WaveEquation::LinearForm L(mesh, un, u_old, ff, dt);
  Matrix A;
  Vector b;
  a.assemble(A, true);
  KrylovSolver solver(bicgstab, bjacobi);

  uint step = 0;
  while (t < Tfinal)
  {
    if (t+tstep > Tfinal) dt = Tfinal - t;
    src.t=t;
    Analytic<Source> f( mesh, src);
    ff << f;
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u_old = un;
    un = u;
    //u.eval(u_values, rec1);
    message("u(rec1) = %g", u.vector()[20000]);   
    #ifdef IO
    if (step%10 == 0) file << u; 
    #endif
    t +=tstep;
    step += 1;
  }
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
