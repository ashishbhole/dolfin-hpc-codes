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

#include "WaveEquation.h"
//#include "Projection.h"

#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.01;
real speed0 = 1.0;
real speed1 = 0.1;
real Tfinal = 5.0;

real alpha_m_value = 0.0;
real alpha_f_value = 0.0;

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
  tstep = Nc * h / speed0;
  Constant dt(tstep);
  Constant alpha_m(alpha_m_value);
  Constant alpha_f(alpha_f_value);
  WaveEquation::BilinearForm a(mesh, dt, alpha_m, alpha_f);
  Function u(a.trial_space());
  Function u_old(a.trial_space());
  Function v_old(a.trial_space());
  Function a_old(a.trial_space());
  Function ff(a.trial_space());
  WaveEquation::LinearForm L(mesh, u_old, v_old, a_old, ff, dt, alpha_m, alpha_f);
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
    if (t+tstep > Tfinal) dt = Tfinal - t;
    src.t=t;
    Analytic<Source> f( mesh, src);
    ff << f;
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
    if (step%10 == 0) file << u; 
    #endif
    t +=tstep;
    step += 1;
  }
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
