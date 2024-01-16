// Written by Ashish Bhole (2024): 
//
// This is a linear wave equation equation.
//
//     u_tt - c^2 u_xx = f

#define IO

#include "WaveEquation.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real Np = 100;

real tstep = 0.001;
real speed = 0.1;
real Tfinal = 4.0;

real alpha_m_value = 0.0;
real alpha_f_value = 0.0;

struct ExactSolution : public Value< ExactSolution, 1 >
{
  ExactSolution() : t(0), alpha(32.0), speed(1.0) {}
  void eval( real * values, const real * x ) const
  {
    real xcl = x[0]-0.5-speed*t;
    real xcr = x[0]-0.5+speed*t;
    values[0] = 0.5*exp(-alpha*( xcl*xcl )) + 0.5*exp(-alpha*( xcr*xcr ));
  }
  double t, alpha, speed;
};

struct Source : public Value< Source, 1 >
{
  Source(): t(0), alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.0;
  }
  double t, alpha;
};

struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

// Acceleration update
void update_a(Function& a, const Function& u, const Function& a0,
              const Function& v0,  const Function& u0,
              double beta, double tstep)
{
  // a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
  a.vector()  = u.vector();
  a.vector() -= u0.vector();
  a.vector() *= 1.0/tstep;
  a.vector() -= v0.vector();
  a.vector() *= 1.0/((0.5-beta)*tstep);
  a.vector() -= a0.vector();
  a.vector() *= (0.5-beta)/beta;
}

// Velocity update
void update_v(Function& v, const Function& a, const Function& a0,
              const Function& v0, double gamma, double tstep)
{
  // v = dt * ((1-gamma)*a0 + gamma*a) + v0
  v.vector()  = a0.vector();
  v.vector() *= (1.0-gamma)/gamma;
  v.vector() += a.vector();
  v.vector() *= tstep*gamma;
  v.vector() += v0.vector();
}

File file("solution.pvd");

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  UnitInterval mesh(Np);

  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  real t = 0.0;
  ExactSolution Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<ExactSolution> ui( mesh, Gaussian);

  Source src;

  real Nc = 0.05;
  real h = 1.0/Np;
  tstep = Nc * h / speed;
  Constant dt(tstep);
  Constant c(speed);
  Constant alpha_m(alpha_m_value);
  Constant alpha_f(alpha_f_value);

  WaveEquation::BilinearForm a(mesh, c, dt, alpha_m, alpha_f);
  Function u(a.trial_space());
  Function u_old(a.trial_space());
  Function v_old(a.trial_space());
  Function a_old(a.trial_space());
  Function ff(a.trial_space());
  FunctionInterpolation::compute(ui, u_old);
  u_old.sync();
  WaveEquation::LinearForm L(mesh, u_old, v_old, a_old, ff, c, dt, alpha_m, alpha_f);
  Matrix A;
  Vector b;
  a.assemble(A, true);
  KrylovSolver solver(bicgstab, bjacobi);

  Function u_vec(a.trial_space());
  Function v_vec(a.trial_space());
  Function a_vec(a.trial_space());
  Function u0_vec(a.trial_space());
  Function v0_vec(a.trial_space());
  Function a0_vec(a.trial_space());

  real const gamma = 0.5+alpha_f_value-alpha_m_value;
  real const beta  = (gamma+0.5)*(gamma+0.5)/4.0;
  uint step = 0;
  while (t < Tfinal)
  {
    src.t=t;
    Analytic<Source> f( mesh, src);
    FunctionInterpolation::compute(src, ff);
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    
    // Update fields
    u_vec  = u;
    u0_vec = u_old;
    v0_vec = v_old;
    a0_vec = a_old;

    // use update functions using vector arguments
    update_a(a_vec, u, a0_vec, v0_vec, u0_vec, beta, tstep);
    update_v(v_vec, a_vec, a0_vec, v0_vec, gamma, tstep);

    // Update (u_old <- u)
    v_old = v_vec;
    a_old = a_vec;
    u_old = u;
    
    #ifdef IO
      if (step%100 == 0) file << u; 
    #endif
    t +=tstep;
    step += 1;
  }
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
