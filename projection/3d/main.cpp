// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Testing evaluation at arbitrary points

#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

struct Fun : public Value< Fun, 1 >
{
  Fun(): t(0) {}
  void eval( real * values, const real * x ) const
  {
    values[0] = exp(t) * sin(2.0 * DOLFIN_PI * x[0]) * cos(2.0 * DOLFIN_PI * x[1]) * cos(2.0 * DOLFIN_PI * x[2]) ; 
  }
  double t;
};

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
int main()
{
  dolfin_init();

  // Create mesh and a point in the mesh
  //UnitCube mesh(50, 50, 50);
  Mesh mesh("disc_with_transducers.bin");
  real t = 0.0;
  real Tfinal = 10.0;
  real tstep = 0.1;

  //real x[3] = {0.1,0.3,0.5};
  real x[3] = {0.0, 0.0, -0.001035-0.00053};
  real f_values[1] = {0.0};
  real g_values[1] = {0.0};

  Fun function;

  // Project to a discrete function
  Projection::BilinearForm a(mesh);
  Function g(a.trial_space());

  // solve PDE
  Matrix A;
  Vector b;
  a.assemble(A, true);

  KrylovSolver solver(bicgstab, bjacobi);

  File( "function.pvd" ) << g;

  uint step = 0;
  while (t < Tfinal)
  {
    function.t = t;
    Analytic<Fun> f(mesh, function);
    Projection::LinearForm L(mesh, f);
    L.assemble(b, step == 0);
    solver.solve(A, g.vector(), b);
    g.sync();

    // Evaluate user-defined function f
    f.eval(f_values, x);
    // Evaluate discrete function g (projection of f)  
    g.eval(g_values, x);

    // g.eval throws inf for no proc > 2 with bin mesh
    message("time, f(x), g(x) = %g %g %g", t, f_values[0],  g_values[0]);
    t +=tstep;
    step += 1;
  }
  dolfin_finalize();
  return 0;
}
