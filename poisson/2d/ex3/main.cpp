// Written by Ashish Bhole (2023): 
// This demo uses dolfin-hpc 0.9.5.
// Poisson's equation with mixed elements
// From https://fenicsproject.org/olddocs/dolfin/1.5.0/python/demo/documented/mixed-poisson-dual/python/documentation.html
//
#include "Poisson.h" // produced by ffc -l dolfin_hpc Poisson.ufl
#include <dolfin.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
struct Source : public Value< Source >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 10.0 * exp( - ( pow(x[0]-0.5,2) + pow(x[1]-0.5,2) ) / 0.02);
  }
};

struct DirichletFunction : public Value< DirichletFunction >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 0.0;
  }
};

struct Flux : public Value< Flux >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = sin(5.0*x[0]);
  }
};


struct DirichletBoundary : public SubDomain
{ 
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary && (x[0] < DOLFIN_EPS || x[0] > (1.0 - DOLFIN_EPS) );
  }
};

//-----------------------------------------------------------------------------

int main(int argc, char **argv)
{

  dolfin_init(argc, argv);
  UnitSquare mesh(100, 100); 

  SubSystem flux(0);
  SubSystem displacements(1);
  
  Analytic< Source >            f( mesh );
  Analytic<DirichletFunction> dbf( mesh );
  Analytic<Flux>                g( mesh );
  
  DirichletBoundary db;

  Poisson::BilinearForm a( mesh );
  Function w( a.trial_space() );
  Function u, sig;
  sig = SubFunction(w, 0);
  u   = SubFunction(w, 1);
  Poisson::LinearForm L( mesh, f, g);

  DirichletBC bc( dbf, mesh, db, displacements);

  Matrix A;
  Vector b;
  a.assemble( A, true );
  L.assemble( b, true );
  bc.apply( A, b, a );

  KrylovSolver solver( bicgstab, bjacobi );
  solver.solve( A, w.vector(), b );

  sig = SubFunction(w, 0);
  File( "flux.pvd" ) << sig;
  u   = SubFunction(w, 1);
  File( "poisson.pvd" ) << u;
  dolfin_finalize();

  return 0;
}
