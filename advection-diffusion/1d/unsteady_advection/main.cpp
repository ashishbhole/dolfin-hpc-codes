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

#include "AdvectionDiffusion.h"

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
    real xc = speed_x*t;
    values[0] = exp(-alpha*( (x[0]-xc-0.25)*(x[0]-xc-0.25) ));
  }
  double t, alpha, speed_x;
};

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

  real Nc = 0.05;
  //real h = MeshQuality(mesh).h_max; // does not seem to work for 1d meshes
  real h = 1.0 / Np;
  tstep = Nc * h / speed_x;

  Constant dt(tstep);
  Constant speed(speed_x);
  real tau_val = 0.0*h/speed_x;
  Constant tau(tau_val);

  AdvectionDiffusion::BilinearForm a(mesh, speed, tau, dt);
  Function u1(a.trial_space());
  Function u0(a.trial_space());
  FunctionInterpolation::compute(ui, u0);
  AdvectionDiffusion::LinearForm L(mesh, u0);
  Matrix A;
  Vector b;
  a.assemble(A, true);
  KrylovSolver solver(bicgstab, bjacobi);

  uint step = 0;

  #ifdef IO
  file << u0;
  #endif

  while (t < Tfinal)
  {
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u1.vector(), b);
    u1.sync();
    u0 = u1;
    t +=tstep;
    step += 1;
    #ifdef IO
    if (step%10 == 0) file << u1; 
    #endif
  }
  // Get the exact solution at Tfinal
  Gaussian.t = t;
  Analytic<ExactSolution> uex( mesh, Gaussian);
  FunctionInterpolation::compute(uex, u0);
  u0.sync();

  file << u0;

  // Compute the numerical error in u1 as : u1 = u1 - ue
  u1 -= u0;
  message( "time, h, Error l1, l2, linf norm: %e %e %e %e %e", t, h, u1.vector().norm(l1), u1.vector().norm(l2), u1.vector().norm(linf) );

  dolfin_finalize();
  return 0;
}
