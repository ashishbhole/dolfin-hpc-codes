// Copyright (C) 2010 Jeannette Spuehler.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2017.
// Modified by Ashish Bhole 2023: Modified from ConvectionDiffusion demo

#define IO

#include "AdvectionDiffusion.h"
#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.01;
real speed_x = 0.2;
real speed_y = 0.2;
real Tfinal  = 1.0;

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
  uint num_refine = 0;

  if (argc > 1)
  {
      for (uint i = 0; argv[1][i] != '\0'; i++)
      {
         if (!isdigit(argv[1][i]))
         {
           cout << "Bad character in command line argument\n";
           exit(1);
         }
      }
      num_refine = std::__cxx11::stoi(argv[1]);
  }
  else
  {
      cout << "Error: missing command line argument\n";
      exit(1);
  }
    
  dolfin_init(argc, argv);

  // Create mesh
  //Mesh mesh("mesh/UnitSquareMesh_32x32.bin");

  uint Np = pow(2,num_refine)*32;
  UnitSquare mesh(Np, Np);

  // Set boundary conditions
  Analytic<DirichletFunction> u0(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  double t = 0.0;
  // Set solution to initial condition
  ExactSolution Gaussian;
  Gaussian.alpha = 128.0;
  Gaussian.speed_x = speed_x;
  Gaussian.speed_y = speed_y;
  Gaussian.t = t;
  Analytic<ExactSolution> ui( mesh, Gaussian);

  double Nc = 0.05;
  double h = MeshQuality(mesh).h_max; //1.0/sqrt(N*N);
  tstep = Nc * h / (sqrt(pow(speed_x,2)+pow(speed_y,2)));

  Constant dt(tstep);
  Constant adv_x(speed_x);
  Constant adv_y(speed_y);
  Constant tau(0.0*h/(sqrt(pow(speed_x,2)+pow(speed_y,2))));

  AdvectionDiffusion::BilinearForm a(mesh);
  Function u(a.trial_space());
  Function u1(a.trial_space());

  //Interpolate from an expression or a coefficient or a generic function 
  u << ui;

  // Beware: arguments must be called in a sequence matched by the call in header file.
  // Need to confirm this fact from developers.
  AdvectionDiffusion::LinearForm L(mesh, u, adv_x, adv_y, tau, dt);

  Matrix A;
  Vector b;
  a.assemble(A, true);

  KrylovSolver solver(bicgstab, bjacobi);
  uint step = 0;
  #ifdef IO
    file << u;
  #endif
  
  while (t < Tfinal)
  {
    // Adjust dt to reach final time exactly
    if (t+tstep > Tfinal) dt = Tfinal - t;
    L.assemble(b, step == 0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u1 = u;
    t +=tstep;
    step += 1;
    #ifdef IO
    if (step%10 == 0) file << u; 
    #endif
  }
  Gaussian.t = Tfinal;
  Analytic<ExactSolution> uex( mesh, Gaussian);
  u1 << uex;
  u1 -= u;
  message( "h, Error l1, l2, linf norm: %e %e %e %e", h, u1.vector().norm(l1), u1.vector().norm(l2), u1.vector().norm(linf) );

  dolfin_finalize();
  return 0;
}
