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
#include <fstream>

using namespace dolfin;

real tstep = 0.01;
real speed = 1.0;
real Tfinal = 100.0;
real Nc = 0.01;

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
    if( (sqrt(x[0]*x[0] + x[1]*x[1]) ) <= 3.6775 + abs(DOLFIN_EPS) )
    {	    
      values[0] = sin(10.0*DOLFIN_PI*t);
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

void project(Function &ff, Mesh &mesh, Source &src, double t,
             Matrix &Ap, Vector &bp, 
	     KrylovSolver &solver, bool flag)
{
  src.t=t;
  Analytic<Source> f( mesh, src);
  Projection::LinearForm Lp(mesh, f);
  Lp.assemble(bp, flag);
  solver.solve(Ap, ff.vector(), bp);
  ff.sync();
}

File file("solution.pvd");
std::ofstream outfile ("signals.txt");

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  real rec1[3] = { 4.0, 0.5, 0.0};
  real rec2[3] = {-4.0, 0.5, 0.0};
  real u_values1[1] = {0.0};
  real u_values2[1] = {0.0};

  // Parallel file writing does not work with in-built meshes.
  Mesh mesh("disc_in_disc.bin");

  Analytic<DirichletFunction> u0(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  double t = 0.0;
  InitialCondition Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<InitialCondition> ui( mesh, Gaussian);

  Source src;

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

  Projection::BilinearForm ap(mesh);
  Matrix Ap;
  Vector bp;
  ap.assemble(Ap, true);

  uint step = 0;
  while (t < Tfinal)
  {
    if (t+tstep > Tfinal) dt = Tfinal - t;
    project(ff, mesh, src, t, Ap, bp, solver, step==0);

    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u_old = un;
    un = u;
    #ifdef IO
    if (step%100 == 0) file << u; 
    #endif

    u.eval(u_values1, rec1);
    u.eval(u_values2, rec2);
    message("t, u1, u2 = %g %g %g", t, u_values1[0], u_values2[0]);
    outfile << t << " " << u_values1[0] << " " << u_values2[0] << "\n";

    t +=tstep;
    step += 1;
  }
  outfile.close();  
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
