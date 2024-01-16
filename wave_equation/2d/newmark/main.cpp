// Modified by Ashish Bhole (2023): 
// Modified from ConvectionDiffusion demo
//
// This is a linear wave equation equation.
//
//     u_tt + c^2 div( grad(u)) = f

#define IO

#include "WaveEquation.h"
#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real tstep = 0.01;
real speed = 1.0;
real Tfinal = 5.0;

real beta_value = 0.25;
real gamma_value= 0.5;

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

// Displacement update
void update_d(Function &u, Function& a, const Function& u0,
              const Function& v0,  const Function& a0,
              double beta_value, double tstep)
{
  // u = u0 + dt * v0 + 0.5*dt*((1-2*beta)*a0 + 2*beta*a) 
  u.vector()  = a.vector();
  u.vector() *= 2.0*beta_value/(1.0-2.0*beta_value);
  u.vector() += a0.vector();
  u.vector() *= 0.5*tstep*(1.0-2.0*beta_value);
  u.vector() *= 1.0/tstep;
  u.vector() += v0.vector();
  u.vector() *= tstep;
  u.vector() += u0.vector();
}

// Velocity update
void update_v(Function& v, const Function& a, const Function& a0,
              const Function& v0, double gamma_value, double tstep)
{
  // v = dt * ((1-gamma)*a0 + gamma*a) + v0
  v.vector()  = a0.vector();
  v.vector() *= (1.0-gamma_value)/gamma_value;
  v.vector() += a.vector();
  v.vector() *= tstep*gamma_value;
  v.vector() += v0.vector();
}

// Acceleration update
void update_a(Function& a, const Function& u, const Function& a0,
              const Function& v0,  const Function& u0,
              double beta_value, double tstep)
{
  // a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
  a.vector()  = u.vector();
  a.vector() -= u0.vector();
  a.vector() *= 1.0/tstep;
  a.vector() -= v0.vector();
  a.vector() *= 1.0/((0.5-beta_value)*tstep);
  a.vector() -= a0.vector();
  a.vector() *= (0.5-beta_value)/beta_value;
}

File file("solution.pvd");

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  real val;

  // Parallel file writing does not work with in-built meshes.
  Mesh mesh("rectangular_struct_tria.bin");

  Analytic<DirichletFunction> u0(mesh);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  double t = 0.0;
  InitialCondition Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<InitialCondition> ui( mesh, Gaussian);
  Constant c(speed);

  Source src;

  // solve for a_old
  KrylovSolver solver(bicgstab, bjacobi);

  double Nc = 0.05;
  double h = MeshQuality(mesh).h_max;
  tstep = Nc * h / speed;
  Constant dt(tstep);
  Constant beta(beta_value);
  Constant gamma(gamma_value);
  WaveEquation::BilinearForm a(mesh, c, dt, beta, gamma);
  Function u_new(a.trial_space());
  Function v_old(a.trial_space());
  Function a_old(a.trial_space());
  Function u_old(a.trial_space());
  Matrix A;
  Vector b;
  a.assemble(A, true);

  Function u_vec(a.trial_space());
  Function v_vec(a.trial_space());
  Function a_vec(a.trial_space());
  Function u0_vec(a.trial_space());
  Function v0_vec(a.trial_space());
  Function a0_vec(a.trial_space());
  
  uint step = 0;
  while (t < Tfinal)
  {
    src.t=t;
    Analytic<Source> f( mesh, src);
    WaveEquation::LinearForm L(mesh, u_old, v_old, a_old, f, c, dt, beta, gamma);
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u_new.vector(), b);
    u_new.sync();

    // Update fields
    u_vec  = u_new;
    u0_vec = u_old;
    v0_vec = v_old;
    a0_vec = a_old;

    // use update functions using vector arguments
    update_a(a_vec, u_new, a0_vec, v0_vec, u0_vec, beta_value, tstep);
    update_v(v_vec, a_vec, a0_vec, v0_vec, gamma_value, tstep);

    // Update (u_old <- u)
    a_old = a_vec;
    v_old = v_vec;
    u_old = u_new;
 
    u_new.eval(u_values, rec1);
    val = u_values[0];

    // MPI reduction is needed as the point could be in any rank.
    // Reduction is supposed to pick the real number amongst 'inf's
    if(mesh.is_distributed())
    {
      MPI::reduce< MPI::min >( &u_values[0], &val, 1, 0, MPI::DOLFIN_COMM );
    }

    message("value at rec1 = %e", val);

    #ifdef IO
    if (step%10 == 0) file << u_new; 
    #endif
    t +=tstep;
    step += 1;
  }
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u_new.vector().norm(l1), u_new.vector().norm(l2), u_new.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
