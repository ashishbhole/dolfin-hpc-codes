// Adapted from the implementation by:
// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-02-07
// Last changed: 2007-08-20
//
// Copyright (C) 2023 Ashish Bhole: 
// This demo uses dolfin-hpc 0.9.5.
// Poisson.ufl contains form for Poisson's equation.
// Projection.ufl contails form for FE projection.
// This demo program solves Poisson's equation
//
//     - div grad u(x, y) = f(x, y)
//
//  For the exact solution: 
//
//  uex = sin(2*pi*x) cos(2*pi*y)
//
// on the unit square the source f given by:
//
//     f(x, y) = 2(2*pi)^2 u(x,y)
//
 
#include "Poisson.h" // produced by ffc -l dolfin_hpc Poisson.ufl
#include "Projection.h" // produced by ffc -l dolfin_hpc Projection.ufl
#include <dolfin.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
struct ExactSolution : public Value< ExactSolution, 1 >
{
  void eval( real * values, const real * x ) const
  {
    values[0] = sin(2.0 * DOLFIN_PI * x[0]) * cos(2.0 * DOLFIN_PI * x[1]);
  }
};

struct Source : public Value< Source >
{
  void eval( real * value, const real * x ) const
  {
    value[0] = 8.0 * DOLFIN_PI * DOLFIN_PI * sin(2.0 * DOLFIN_PI * x[0]) * cos(2.0 * DOLFIN_PI * x[1]);
  }
};

struct DirichletBoundary : public SubDomain
{ 
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

//-----------------------------------------------------------------------------

int main(int argc, char **argv)
{

  // variable to specify how many times a mesh to be refined.	
  uint num_refine = 0;

  // command line input for num_refine
  if (argc > 1)
  {
      // Make sure that argument string contains nothing but digits
      for (uint i = 0; argv[1][i] != '\0'; i++)
      {
         if (!isdigit(argv[1][i]))
         {
           cout << "Bad character in command line argument\n";
           exit(1);
         }
      }
      // how can we get rid of std::__cxx11::
      num_refine = std::__cxx11::stoi(argv[1]);
  }
  else
  {
      cout << "Error: missing command line argument\n";
      exit(1);
  }

  // initialize dolfin-hpc
  dolfin_init(argc, argv);
  
  // Create/import mesh
  UnitSquare mesh(4, 4); 
  //Mesh mesh("./unit_square.bin");

  // refine the mesh 'num_refine' times 
  for(uint i = 0; i < num_refine; i++) mesh.refine();

  // get a characteristic length of the mesh
  double h = MeshQuality(mesh).h_max;
  
  // Create coefficients
  Analytic< Source > f( mesh );
  Analytic<ExactSolution> uex( mesh );
  
  // Define weak form
  Poisson::BilinearForm a( mesh );
  Poisson::LinearForm   L( mesh, f);

  // define boundary conditions
  DirichletBoundary boundary;
  DirichletBC       bc( uex, mesh, boundary );

  // Declare Matrix and vector for discrete problem
  Matrix A;
  Vector b;

  // assemble system
  a.assemble( A, true );
  L.assemble( b, true );

  // apply boundary conditions
  bc.apply( A, b, a );
  Function u ( a.trial_space() );

  // solver specification
  KrylovSolver solver( bicgstab, bjacobi );
  solver.solve( A, u.vector(), b );

  // write numerical solution to vtu file
  File( "poisson.pvd" ) << u;

  // Project exact solution of FE space to measure error estimates
  // This is to compute numerical error.
  Matrix A1;
  Vector b1;
  Projection::BilinearForm a1( mesh );
  Projection::LinearForm   L1( mesh, uex);
  a1.assemble( A1, true );
  L1.assemble( b1, true );
  Function ue ( a1.trial_space() );
  solver.solve( A1, ue.vector(), b1 );

  // Error ue = ue - e
  ue -= u;

  // print nomrs of the numerical error
  message( "h, l1, l2, inf norm:");
  message( "%e %e %e %e", h, ue.vector().norm(l1), ue.vector().norm(l2), ue.vector().norm(linf) );

  // finalize dolfin environment
  dolfin_finalize();

  return 0;
}
