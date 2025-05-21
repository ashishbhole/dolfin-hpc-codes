// Copyright (C) 2024 Ashish Bhole.
// Licensed under the GNU LGPL Version 2.1.
#define IO

#include "Euler.h"
#include "Projection.h"
#include <sstream>
#include <dolfin.h>

using namespace dolfin;

real GAMMA = 1.4;
real tstep = 1e-7;
real Tfinal  = 2.0;

real rho_inf = 1.4;
real p_inf = 101325.0;

real Mach = 1.5; // 2.52;
real c = sqrt(GAMMA * p_inf / rho_inf);

uint save_iter = 100;

struct CylinderWall : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return (sqrt((x[0]-0.8)*(x[0]-0.8)+(x[1]-1.0)*(x[1]-1.0)) <= (0.2+DOLFIN_EPS)) && on_boundary;
  }
};

struct TunnelWall : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return (abs(x[1]-0.0) < DOLFIN_EPS || abs(x[1]-2.0) < DOLFIN_EPS ) && on_boundary;
  }
};

struct Inflow : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return (abs(x[0]) < DOLFIN_EPS) && on_boundary;
  }
};

struct Outflow : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return (abs(x[0]-3.0) < DOLFIN_EPS) && on_boundary;
  }
};

struct InflowBC: public Value< InflowBC, 4 >
{
  void eval( real * values, const real * x ) const
  {
      values[0] = rho_inf;
      values[1] = rho_inf * c * Mach;
      values[2] = 0.0;
      values[3] = p_inf/ (GAMMA-1.0) + 0.5*abs(values[1]*values[1]+values[2]*values[2])/rho_inf;
  }
};

struct InitialCondition: public Value< InitialCondition, 4 >
{
  void eval( real * values, const real * x ) const
  {
      values[0] = rho_inf;
      values[1] = rho_inf * c * Mach; // 0.0;
      values[2] = 0.0;
      values[3] = p_inf/ (GAMMA-1.0) + 0.5*abs(values[1]*values[1]+values[2]*values[2])/rho_inf;
  }
};

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);
  Mesh mesh("cylinder.bin");

  SubSystem density(0);
  SubSystem momentum(1);
  SubSystem energy(2);

  // Create periodic boundary condition
  Inflow inflow;
  Outflow outflow;
  CylinderWall cylwall;
  TunnelWall   tunwall;
  Analytic<InflowBC> wi(mesh);
  Analytic<InitialCondition> wic(mesh);

  MeshValues<size_t, Facet> ext_boundaries(mesh);
  ext_boundaries = 0;
  cylwall.mark(ext_boundaries, 0);
  tunwall.mark(ext_boundaries, 0);
  inflow.mark(ext_boundaries, 1);
  outflow.mark(ext_boundaries, 2);

  // For initial conditions 
  double t = 0.0;

  DirichletBC bc_inflow( wi, mesh, inflow );
  DirichletBC bc_outflow( wi, mesh, outflow );
  

  // time step computation
  double Nc = 0.05;
  double h = MeshQuality(mesh).h_max;
  //tstep = Nc * h / (u0.vector().norm(l2) + c);

  // some quantities in form files
  Constant dt(tstep);
  Constant tau_vms_rho(0.0*tstep);
  Constant tau_vms_m  (0.0*tstep);
  Constant tau_vms_E  (0.0*tstep);
  Constant tau_sc_rho (1e-8);
  Constant tau_sc_m   (1e-8);
  Constant tau_sc_E   (1e-8);
  Constant tau_anis_sc_rho (1e-6);
  Constant tau_anis_sc_m   (1e-6);
  Constant tau_anis_sc_E   (1e-6);

  Matrix A, A1;
  Vector b, b1;
  KrylovSolver solver( gmres, bjacobi );

  Projection::BilinearForm a1(mesh);
  Function w0(a1.trial_space());
  a1.assemble(A1, true);
  Projection::LinearForm L1(mesh, wi);
  L1.assemble(b1, true);
  solver.solve(A1, w0.vector(), b1);
  w0.sync();

  // See the declaration in the header file
  Euler::BilinearForm a(mesh);
  Function w1(a.trial_space());

  Function rho0, m0, E0, u0;
  rho0 = SubFunction(w0, 0);
  m0   = SubFunction(w0, 1);
  E0   = SubFunction(w0, 2);

  File file("sol.pvd", t);

  LabelList<Function> output;
  Label<Function> rho(rho0, "Density");
  Label<Function> m(m0, "Momentum");
  Label<Function> E(E0, "Energy");
 
  output.push_back(rho);
  output.push_back(m);
  output.push_back(E);

  file << output;

  Euler::LinearForm L(mesh, rho0, m0, E0, dt, tau_vms_rho, tau_vms_m, tau_vms_E, tau_sc_rho, tau_sc_m, tau_sc_E, tau_anis_sc_rho, tau_anis_sc_m, tau_anis_sc_E );
  a.assemble(A, true);

  uint step = 0;

  while (t < Tfinal)
  {
    // Adjust dt to reach final time exactly
    if (t+tstep > Tfinal) dt = Tfinal - t;
    L.assemble(b, step==0);
    bc_inflow.apply( A, b, a );
    bc_outflow.apply( A, b, a );

    solver.solve(A, w1.vector(), b);
    w1.sync();
    w0 = w1;

    rho0 = SubFunction(w0, 0);
    m0   = SubFunction(w0, 1);
    E0   = SubFunction(w0, 2);

    message( "iter, t: %e %d", t, step );
    
    t +=tstep;
    step += 1;

    #ifdef IO
    if (step%save_iter == 0) 
    {
      file << output;
    }
    #endif

  }
  dolfin_finalize();
  return 0;
}
