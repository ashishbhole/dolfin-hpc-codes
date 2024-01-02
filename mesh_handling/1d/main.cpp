// Copyright (C) 2006-2007 Johan Jansson and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
#include <dolfin.h>

using namespace dolfin;

real tstep =  2.0 * 1e-9;
real Tfinal = 2.0 * 1e-6;

struct Voltage0 : public Value< Voltage0, 1 >
{
  void eval( real * values, const real * x ) const
  {
      values[0] = 0.0;
  }
};

struct Voltage1 : public Value< Voltage1, 1 >
{
  void eval( real * values, const real * x ) const
  {
      values[0] = 200.0;
  }
};

struct DirichletBoundary1 : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return x[0] < DOLFIN_EPS && on_boundary;
  }
};

struct DirichletBoundary2 : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    // fabs((x[0]-1.0) <= DOLFIN_EPS && on_boundary; does not work
    return x[0] > (1.0 - DOLFIN_EPS) && on_boundary;
  }
};

int main()
{
  dolfin_init();

  // Read mesh
  UnitInterval mesh(100);

  //MeshTopology::disp();

  DirichletBoundary1 leftboundary;
  DirichletBoundary2 rightboundary;

  Analytic< Voltage0 > Vleft( mesh );
  Analytic< Voltage1 > Vright( mesh );

  DirichletBC bcp0(Vleft,  mesh, leftboundary);
  DirichletBC bcp1(Vright, mesh, rightboundary);

  //message(mesh.hash());
  //mesh.disp();
  //mesh.check();

  for ( VertexIterator v( mesh ); !v.end(); ++v )
  {
     message("%i %f", v->index(), v->x()[0]);	
  }

  for ( CellIterator c( mesh ); !c.end(); ++c )
  {
     message("%i %f", c->index(), c->inradius());
  }
  
  // Save solution to VTK format
  //File file( "solution.pvd" );

  dolfin_finalize();

  return 0;
}
