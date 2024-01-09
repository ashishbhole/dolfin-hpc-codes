// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//

#include "Piezoelectric_model.h"
#include <dolfin.h>

using namespace dolfin;

int Np = 1000;

real tstep =  2.0 * 1e-9;
real Tfinal = 2.0 * 1e-6;

real alpha_m_value = 0.0;
real alpha_f_value = 0.0;

struct Force : public Value< Force, 1 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.;
  }
};

struct Traction : public Value< Traction, 1 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = 1e10;
  }
};

struct Charge : public Value< Charge, 1 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.;
  }
};

struct Gradphi : public Value< Gradphi, 1 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.;
  }
};

struct Dummy : public Value< Dummy, 1 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = x[0]*200.0;
  }
};

struct LeftBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS && on_boundary;
  }
};

struct RightBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    // fabs((x[0]-1.0) <= DOLFIN_EPS && on_boundary; does not work
    //message("%d %f %d", on_boundary, x[Np], abs(1.0-x[Np]) < DOLFIN_EPS );    
    return on_boundary &&  abs(1.0-x[Np]) < DOLFIN_EPS; 
  }
};

struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

class Left : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return x[0] < DOLFIN_EPS && on_boundary;
  }
};

class Right : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return on_boundary &&  abs(1.0-x[Np]) < DOLFIN_EPS;
  }
};

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
  UnitInterval mesh(Np);

  real t = 0.0;

  Left left_boundary;
  Right right_boundary;
  // Create mesh function over the cell facets
  MeshValues<size_t, Cell> sub_domains(mesh);

  // Mark all facets as sub domain 2
  sub_domains = 0;
  left_boundary.mark(sub_domains, 1);
  right_boundary.mark(sub_domains, 2);

  // Define sub systems for boundary conditions
  SubSystem displacements( 0 );
  SubSystem potential( 1 );
  
  LeftBoundary lb;
  RightBoundary rb;
  DirichletBoundary db;

  // Create right-hand side
  Analytic< Force > force( mesh );
  Analytic< Traction > traction ( mesh );
  Analytic< Charge > charge( mesh );
  Analytic< Gradphi > gradphi( mesh );

  Constant          zero( 0.0 );
  Constant          Vright( 200.0 );
  Analytic< Dummy > dum( mesh );

  DirichletBC bcu (zero,   mesh, lb,  displacements);
  DirichletBC bcp (dum ,   mesh, db,  potential);
  
  Constant dt(tstep);
  Constant alpha_m(alpha_m_value);
  Constant alpha_f(alpha_f_value);

  // Set up PDE
  Piezoelectric_model::BilinearForm a( mesh, dt, alpha_m, alpha_f );
  Function w(a.trial_space());
  Function u, phi, u_old, v_old, a_old, v_vec, a_vec;
  u_old= SubFunction(w, 0);
  v_old= SubFunction(w, 0);
  a_old= SubFunction(w, 0);
  v_vec= SubFunction(w, 0);
  a_vec= SubFunction(w, 0);
		    
  //Piezoelectric_model::LinearForm L( mesh, force, traction, u_old, v_old, a_old, charge, gradphi, dt, alpha_m, alpha_f );
  Piezoelectric_model::LinearForm L( mesh, traction, u_old, v_old, a_old, dt, alpha_m, alpha_f );

  // u and phi needs to be separated here
  u   = SubFunction(w, 0);
  phi = SubFunction(w, 1);

  // Solve PDE
  Matrix A;
  Vector b;
  a.assemble( A, true );

  KrylovSolver solver(bicgstab, bjacobi);

  // Save solution to VTK format
  File file1( "displacement.pvd" );
  File file2( "potential.pvd" );

  real const gamma = 0.5+alpha_f_value-alpha_m_value;
  real const beta  = (gamma+0.5)*(gamma+0.5)/4.0;
  uint step = 0;
  while (t < Tfinal)
  {
    L.assemble(b, step==0);
    bcu.apply(A, b, a);
    bcp.apply(A, b, a);

    solver.solve( A, w.vector()  , b );
    w.sync();

    u = SubFunction(w, 0);
    phi = SubFunction(w, 1);

    // use update functions using vector arguments
    update_a(a_vec, u, a_old, v_old, u_old, beta, tstep);
    update_v(v_vec, a_vec, a_old, v_old, gamma, tstep);

    // Update (u_old <- u)
    v_old = v_vec;
    a_old = a_vec;
    u_old = u;
 
    if (step%100 == 0) 
    { 
      file1 << u;
      file2 << phi;
    }

    t +=tstep;
    step += 1;

    message( "t, l2 norm of the numerical solution is %0.15g %.15g", t, w.vector().norm(l2));
  }
  dolfin_finalize();

  return 0;
}
