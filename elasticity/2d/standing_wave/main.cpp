// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// This demo program solves the equations of static
// linear elasticity for a gear clamped at two of its
// ends and twisted 30 degrees.

#define IO

#include "Elasticity.h"
#include "Projection.h"
#include "ErrorNorm.h"

#include <dolfin.h>

using namespace dolfin;

real tstep = 0.001;
real Tfinal = 0.2;

real alpha_m_value = 0.2;
real alpha_f_value = 0.4;
real eta_m_value = 0.0;
real eta_k_value = 0.0;

// Set elasticity parameters
real E  = 1.0;
real nu = 0.3;
real rho = 1.0; 
real mu  = 0.5*E/(1.0+nu); 
real lambda = E*nu/((1.0+nu)*(1.0-2.0*nu)); 

struct ConstantFunction : public Value< ConstantFunction, 1, 2 >
{
  ConstantFunction( real c_ )
    : c( c_ )
  {
  }

  void eval( real * values, const real * x ) const
  {
    values[0] = c;
    values[1] = c;
  }

  real c;
};

// In C++, the line struct ExactSolution : public Value< ExactSolution, 2 > defines 
// a structure named 'ExactSolution' that inherits from the Value template class. 
// The 'Value' template class is a generic data structure that can hold any type of data. 
// The 2 in the template arguments indicates that the ExactSolution structure has two dimensions.
struct ExactSolution : public Value< ExactSolution, 2 >
{
  ExactSolution(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1]));
    values[1] = cos(2.0*DOLFIN_PI*t) * sin(DOLFIN_PI*x[0]) * cos(4.0*DOLFIN_PI*x[1]);
  }
  double t;
};

struct ExactVelocity : public Value< ExactVelocity, 2 >
{
  ExactVelocity(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = -2.0 * DOLFIN_PI * sin(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1]));
    values[1] = -2.0 * DOLFIN_PI * sin(2.0*DOLFIN_PI*t) * sin(DOLFIN_PI*x[0]) * cos(4.0*DOLFIN_PI*x[1]);
  }
  double t;
};

struct ExactAcc : public Value< ExactAcc, 2 >
{
  ExactAcc(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = -4.0 * DOLFIN_PI * DOLFIN_PI * cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1]));
    values[1] = -4.0 * DOLFIN_PI * DOLFIN_PI * cos(2.0*DOLFIN_PI*t) * sin(DOLFIN_PI*x[0]) * cos(4.0*DOLFIN_PI*x[1]);
  }
  double t;
};

// Dirichlet boundary condition for clamp at left end
struct Source : public Value< Source, 2 >
{
  Source(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = 13.0 * DOLFIN_PI * DOLFIN_PI * mu * cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1])) \
	      -	4.0  * DOLFIN_PI * DOLFIN_PI * rho * cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1])) \
	      + 4.0  * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * sin(4.0*DOLFIN_PI*x[1]) * cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*x[0])  \
              + 9.0  * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1]));

    values[1] = 17.0 * DOLFIN_PI * DOLFIN_PI * mu  * sin(DOLFIN_PI*x[0]) * cos(2.0*DOLFIN_PI*t) * cos(4.0*DOLFIN_PI*x[1]) \
	      - 4.0  * DOLFIN_PI * DOLFIN_PI * rho * sin(DOLFIN_PI*x[0]) * cos(2.0*DOLFIN_PI*t) * cos(4.0*DOLFIN_PI*x[1]) \
	      + 16.0 * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * cos(2.0*DOLFIN_PI*t) * sin(DOLFIN_PI*x[0]) * cos(4.0*DOLFIN_PI*x[1]) \
	      + 6.0  * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * cos(2.0*DOLFIN_PI*t) * cos(DOLFIN_PI*(3.0*x[0] + 2.0*x[1]));
  }
  double t;  
};

struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

Function project_disp(Mesh &mesh, double t)
{
  ExactSolution exact_fun;
  exact_fun.t = t;
  Analytic<ExactSolution> f(mesh, exact_fun);

  Projection::BilinearForm ap(mesh);
  Matrix Ap;
  Vector bp;
  Projection::LinearForm Lp(mesh, f);

  ap.assemble(Ap, true);
  Lp.assemble(bp, true);
  KrylovSolver solver(bicgstab, bjacobi);

  Function ff(ap.trial_space());
  solver.solve(Ap, ff.vector(), bp);
  ff.sync();
  return ff;
}

Function project_vel(Mesh &mesh, double t)
{
  ExactVelocity exact_vel;
  exact_vel.t = t;
  Analytic<ExactVelocity> f(mesh, exact_vel);

  Projection::BilinearForm ap(mesh);
  Matrix Ap;
  Vector bp;
  Projection::LinearForm Lp(mesh, f);

  ap.assemble(Ap, true);
  Lp.assemble(bp, true);
  KrylovSolver solver(bicgstab, bjacobi);

  Function ff(ap.trial_space());
  solver.solve(Ap, ff.vector(), bp);
  ff.sync();
  return ff;
}

Function project_acc(Mesh &mesh, double t)
{
  ExactAcc exact_acc;
  exact_acc.t = t;
  Analytic<ExactAcc> f(mesh, exact_acc);

  Projection::BilinearForm ap(mesh);
  Matrix Ap;
  Vector bp;
  Projection::LinearForm Lp(mesh, f);

  ap.assemble(Ap, true);
  Lp.assemble(bp, true);
  KrylovSolver solver(bicgstab, bjacobi);

  Function ff(ap.trial_space());
  solver.solve(Ap, ff.vector(), bp);
  ff.sync();
  return ff;
}

Function project_src(Mesh &mesh, double t)
{
  Source src;
  src.t = t;
  Analytic<Source> f(mesh, src);

  Projection::BilinearForm ap(mesh);
  Matrix Ap;
  Vector bp;
  Projection::LinearForm Lp(mesh, f);

  ap.assemble(Ap, true);
  Lp.assemble(bp, true);
  KrylovSolver solver(bicgstab, bjacobi);

  Function ff(ap.trial_space());
  solver.solve(Ap, ff.vector(), bp);
  ff.sync();
  return ff;
}

// Acceleration update
// a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
void update_a(Function& a, const Function& u, const Function& a0,
              const Function& v0,  const Function& u0,
              double beta, double tstep)
{
  a.vector()  = u.vector();
  a.vector() -= u0.vector();
  a.vector() *= 1.0/tstep;
  a.vector() -= v0.vector();
  a.vector() *= 1.0/((0.5-beta)*tstep);
  a.vector() -= a0.vector();
  a.vector() *= (0.5-beta)/beta;
}

// Velocity update
// v = dt * ((1-gamma)*a0 + gamma*a) + v0
void update_v(Function& v, const Function& a, const Function& a0,
              const Function& v0, double gamma, double tstep)
{
  v.vector()  = a0.vector();
  v.vector() *= (1.0-gamma)/gamma;
  v.vector() += a.vector();
  v.vector() *= tstep*gamma;
  v.vector() += v0.vector();
}

int main()
{
  dolfin_init();

  // Read mesh
  //Mesh mesh("./mesh2D.bin");
  UnitSquare mesh(50, 50);
  DirichletBoundary boundary;

  double t = 0.0;
  
  Constant dt(tstep);
  Constant rho_(rho);
  Constant mu_(mu);
  Constant lmbda(lambda);
  Constant alpha_m(alpha_m_value);
  Constant alpha_f(alpha_f_value);
  Constant eta_m(eta_m_value);
  Constant eta_k(eta_k_value);

  // Set up PDE
  Elasticity::BilinearForm a(mesh, dt, rho_, mu_, lmbda, alpha_m, alpha_f, eta_m, eta_k );
  Function u_old(a.trial_space());
  Function v_old(a.trial_space());
  Function a_old(a.trial_space());
  Function u(a.trial_space());

  // Solve PDE
  Matrix A;
  Vector b;
  a.assemble( A, true );
  KrylovSolver solver(bicgstab, bjacobi);  
  //dolfin_set( "Krylov maximum iterations", 10000 );

  u_old = project_disp(mesh, t);
  v_old = project_vel (mesh, t);
  a_old = project_acc (mesh, t);

  Function v_vec(a.trial_space());
  Function a_vec(a.trial_space());
  
  Source src;
  ExactSolution exact_fun;

  // Save solution to VTK format
  File file( "elasticity.pvd" ); 
  real const gamma = 0.5+alpha_f_value-alpha_m_value;
  real const beta  = (gamma+0.5)*(gamma+0.5)/4.0;
  uint step = 0;
  while (t < Tfinal)
  {
    src.t=t;
    Analytic<Source> f(mesh, src);
    Elasticity::LinearForm L(mesh, u_old, v_old, a_old, f, dt, rho_, mu_, lmbda, alpha_m, alpha_f, eta_m, eta_k );
    L.assemble(b, step==0);
    exact_fun.t =t;
    Analytic<ExactSolution> ub(mesh, exact_fun);
    DirichletBC bc(ub, mesh, boundary);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();

    // use update functions using vector arguments
    update_a(a_vec, u, a_old, v_old, u_old, beta, tstep);
    update_v(v_vec, a_vec, a_old, v_old, gamma, tstep);

    // Update (u_old <- u)
    v_old = v_vec;
    a_old = a_vec;
    u_old = u;
 
    #ifdef IO
    if (step%100 == 0) file << u;
    #endif

    t +=tstep;
    step += 1;

    message( "t, l2 norm of the numerical solution is %g %.15g", t, u.vector().norm(l2));
  }

  u_old = project_disp(mesh, t);
  u -= u_old;
  ErrorNorm::Functional M( mesh, u );
  Scalar l2_err;
  Assembler::assemble( l2_err, M, true );
  real value = l2_err;
  message( "l2 norm of the Error is %.15g", value);
 
  dolfin_finalize();
  return 0;
}
