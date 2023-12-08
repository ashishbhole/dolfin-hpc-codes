// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Testing evaluation at arbitrary points

#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

//------------------------------------------------------------------------------
struct F : public Value<F>
{
  void eval(real* values, const real* x) const
  {
    values[0] = sin(2.0 * DOLFIN_PI * x[0]) * cos(2.0 * DOLFIN_PI * x[1]) * cos(2.0 * DOLFIN_PI * x[2]) ; 
  }
};
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
int main()
{
  dolfin_init();

  // Create mesh and a point in the mesh
  UnitCube mesh(8, 8, 8);
  mesh.refine();

  real x[3] = {0.3, 0.3, 0.3};
  real f_values[1] = {0.0};
  real g_values[1] = {0.0};

  // A user-defined function
  Analytic<F> f(mesh);

  // Project to a discrete function
  Projection::BilinearForm a(mesh);
  Projection::LinearForm L(mesh, f);

  // solve PDE
  Matrix A;
  Vector b;
  a.assemble(A, true);
  L.assemble(b, true);

  Function g(a.trial_space());
  KrylovSolver solver(bicgstab, bjacobi);

  solver.solve(A, g.vector(), b);
  g.sync();

  File( "function.pvd" ) << g;
  
  // Evaluate user-defined function f
  f.eval(f_values, x);
  message("f(x) = %g", f_values[0]);

  // Evaluate discrete function g (projection of f)  
  g.eval(g_values, x);
  message("g(x) = %g", g_values[0]);

  dolfin_finalize();

  return 0;
}
