// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.

#define IO

#include "Elasticity.h"
#include "Projection.h"
#include "ErrorNorm.h"
#include "LumpedMassMatrix.h"

#include <dolfin.h>

using namespace dolfin;

real Tfinal = 1.0;

// Set elasticity parameters
real E  = 1.0;
real nu = 0.3;
real rho = 1.0; 
real mu  = 0.5*E/(1.0+nu); 
real lambda = E*nu/((1.0+nu)*(1.0-2.0*nu)); 

struct UnitFunction : public Value< UnitFunction, 2 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = 1.0;
    values[1] = 1.0;
  }
};

struct ExactSolution : public Value< ExactSolution, 2 >
{
  ExactSolution(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = cos(DOLFIN_PI*(5.0*x[0] + 4.0*x[1] + 3.0*t));
    values[1] = cos(2.0*DOLFIN_PI*x[0]) * sin(DOLFIN_PI*(3.0*x[1] + 4.0*t));
  }
  double t;
};

// Dirichlet boundary condition for clamp at left end
struct Source : public Value< Source, 2 >
{
  Source(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = 41.0 * DOLFIN_PI * DOLFIN_PI * mu * cos(DOLFIN_PI*(5.0*x[0] + 4.0*x[1] + 3.0*t)) \
	      -	9.0  * DOLFIN_PI * DOLFIN_PI * rho * cos(DOLFIN_PI*(5.0*x[0] + 4.0*x[1] + 3.0*t)) \
	      + 6.0  * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * sin(2.0*DOLFIN_PI*x[0]) * cos( DOLFIN_PI*(3.0*x[1] + 4.0*t) )  \
              + 25.0  * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * cos(DOLFIN_PI*(5.0*x[0] + 4.0*x[1] + 3.0*t));

    values[1] = 13.0 * DOLFIN_PI * DOLFIN_PI * mu  * sin( DOLFIN_PI*(3.0*x[1] + 4.0*t) ) * cos(2.0*DOLFIN_PI*x[0]) \
	      - 16.0 * DOLFIN_PI * DOLFIN_PI * rho * sin( DOLFIN_PI*(3.0*x[1] + 4.0*t) ) * cos(2.0*DOLFIN_PI*x[0]) \
	      + 9.0  * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * sin( DOLFIN_PI*(3.0*x[1] + 4.0*t) ) * cos(2.0*DOLFIN_PI*x[0]) \
	      + 20.0 * DOLFIN_PI * DOLFIN_PI * (lambda + mu) * cos(DOLFIN_PI*(5.0*x[0] + 4.0*x[1] + 3.0*t));
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

real compute_inradius(Mesh & m)
{
  real h_max = 0.0;
  real h_min = 1.0e12;

  for ( CellIterator c( m ); !c.end(); ++c )
  {
    real h  = c->inradius();
    h_max = std::max( h_max, h );
    h_min = std::min( h_min, h );
  }

  real h_min_val = h_min;
  real h_max_val = h_max;
  if ( m.is_distributed() )
  {
    MPI::reduce< MPI::min >( &h_min, &h_min_val, 1, 0, MPI::DOLFIN_COMM );   
    MPI::reduce< MPI::max >( &h_max, &h_max_val, 1, 0, MPI::DOLFIN_COMM );
  }

  return h_max_val;
}

int main()
{
  dolfin_init();

  UnitSquare mesh(200, 200);

  elasticity_finite_element_0 FE;
  uint m = FE.degree();

  real h = compute_inradius(mesh);
  real tstep = 0.2 * h / m / E;

  DirichletBoundary boundary;
  Source src;
  ExactSolution exact_fun;

  double t = 0.0;
  
  Constant dt(tstep);
  Constant rho_(rho);
  Constant mu_(mu);
  Constant lmbda(lambda);
  PETScVector b;
  PETScVector ML;

  UnitFunction  One;
  Analytic< UnitFunction > one( mesh, One );
  LumpedMassMatrix::LinearForm FML(mesh, one, rho_);
  FML.assemble(ML, true);
  
  Function u_0, u_p;

  u_0 = project_disp(mesh,   t);
  u_p = project_disp(mesh, -tstep);

  Elasticity::BilinearForm a(mesh, rho_);
  Matrix A;
  a.assemble(A, true);
  Function u(a.trial_space());

  // Save solution to VTK format
  File file( "elasticity.pvd", t );
  file << u_0;
  uint step = 0;
  while (t < Tfinal)
  {
    src.t=t;
    Analytic<Source> f(mesh, src);
    Elasticity::LinearForm L(mesh, u_0, f, mu_, lmbda, dt );
    L.assemble(b, step==0);
    u.vector() = b;

    // vec.pointwise(v, pw_div); means v = v/vec;
    u.vector().pointwise(ML, pw_div);
    u.vector() += u_0.vector();
    u.vector() += u_0.vector();
    u.vector() -= u_p.vector();
    exact_fun.t =t+tstep;
    Analytic<ExactSolution> ub(mesh, exact_fun);
    DirichletBC bc(ub, mesh, boundary);    
    bc.apply(A, u.vector(), a);
    u.sync();

    // Update (u_old <- u)
    u_p.vector() = u_0.vector();
    u_0.vector() = u.vector();
 
    #ifdef IO
    if (step%100 == 0) file << u_0;
    #endif

    t +=tstep;
    step += 1;

    message( "t, l2 norm of the numerical solution is %g %.15g", t, u_0.vector().norm(l2));
  }

  u_p = project_disp(mesh, t);
  u_0 -= u_p;
  ErrorNorm::Functional M( mesh, u_0 );
  Scalar l2_err;
  Assembler::assemble( l2_err, M, true );
  real value = l2_err;
  message( "h, l2 norm of the Error is %e %e", h, value);
 
  dolfin_finalize();
  return 0;
}
