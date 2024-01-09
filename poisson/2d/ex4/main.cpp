// Written by Ashish Bhole (2023): 
// This demo uses dolfin-hpc 0.9.5.
// Poisson's equation with multiple elements
// From https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/subdomains-poisson/python/documentation.html
//
#include "Poisson.h" // produced by ffc -l dolfin_hpc Poisson.ufl
#include <dolfin.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
struct Source : public Value< Source >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 1.0;
  }
};

struct LeftFlux : public Value< LeftFlux >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = -10.0*exp(-(x[1]-0.5)*(x[1]-0.5));
  }
};

struct RightFlux : public Value< RightFlux >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 1.0;
  }
};

struct TopValue : public Value< TopValue >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 5.0;
  }
};

struct BottomValue : public Value< BottomValue >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 0.0;
  }
};


class Left : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS && on_boundary;
  }
};

class Bottom : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return x[1] < DOLFIN_EPS && on_boundary;
  }
};

class Right : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return x[0] > 1.0 - DOLFIN_EPS && on_boundary;
  }
};

class Top : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return x[1] > 1.0 - DOLFIN_EPS && on_boundary;
  }
};

class Obstacle : public SubDomain
{
  bool inside(const real* x) const //, bool on_boundary) const
  {
    return (x[0] >= 0.5 && x[0] <= 0.7) && (x[1] >= 0.2 && x[1] <= 1.0);
  }
};

//-----------------------------------------------------------------------------

int main(int argc, char **argv)
{

  dolfin_init(argc, argv);
  UnitSquare mesh(100, 100); 

  // Create mesh function over the cell facets
  MeshValues<size_t, Cell> sub_domains(mesh);
  sub_domains = 0;
  Obstacle obstacle;
  obstacle.mark(sub_domains, 1);

  MeshValues<size_t, Facet> boundaries(mesh);
  boundaries = 10;
  Left left;
  left.mark(boundaries, 0);
  Right right;
  right.mark(boundaries,  1);
  Bottom bottom;
  bottom.mark(boundaries, 2);
  Top top;
  top.mark(boundaries, 3);

  Analytic< Source >      f( mesh );
  Analytic<TopValue>      topval( mesh );
  Analytic<BottomValue>   bottomval( mesh );
  Analytic<LeftFlux>      g_L( mesh );
  Constant g_R(1.0);

  Constant a0(1.0);
  Constant a1(0.01);
  
  Poisson::BilinearForm a( mesh , a0, a1 );
  Function u( a.trial_space() );
  Poisson::LinearForm L( mesh, f, g_L, g_R );

  DirichletBC bc_top    ( topval   , mesh, top   );
  DirichletBC bc_bottom ( bottomval, mesh, bottom);

  Matrix A;
  Vector b;
  a.assemble( A, true );
  L.assemble( b, true );
  bc_top.apply( A, b, a );
  bc_bottom.apply( A, b, a );

  KrylovSolver solver( bicgstab, bjacobi );
  solver.solve( A, u.vector(), b );

  File( "poisson.pvd" ) << u;
  dolfin_finalize();

  return 0;
}
