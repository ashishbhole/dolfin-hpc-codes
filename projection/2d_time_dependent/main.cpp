// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Testing evaluation at arbitrary points

#include <dolfin.h>
#include "Projection.h"

using namespace dolfin;

//------------------------------------------------------------------------------
// User defined function with t as a parameter
struct F : public Value<F>
{
  F () : t(0.0) {}	
  void eval(real* values, const real* x) const
  {
    values[0] = exp(t*t) * sin(2.0 * DOLFIN_PI * x[0]) * cos(2.0 * DOLFIN_PI * x[1]);
  }
  double t;
};
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
int main()
{
  dolfin_init();

  // Create mesh and a point in the mesh
  UnitSquare mesh(10, 10);

  // location at which functions to be evaluated
  real x[3] = {0.8, 0.8, 0.0};
  real f_values[1] = {0.0};
  real g_values[1] = {0.0};

  // A user-defined function
  F fun;

  // Project to a discrete function
  Projection::BilinearForm a(mesh);

  // solve PDE
  Matrix A;
  Vector b;
  a.assemble(A, true);

  // Numerical solution
  Function g(a.trial_space());
  KrylovSolver solver(bicgstab, bjacobi);

  double tstep = 0.1;
  double Tfinal = 2.0;
  double t = 0.0;
  uint step = 0;
  File file("function.pvd");  
  while (t < Tfinal)
  {
    if (t+tstep > Tfinal) tstep = Tfinal - t;
    // set time for user defined function
    fun.t = t;
    // get the user defined function on the mesh
    Analytic<F> f(mesh, fun);
    // define/update the linear form for f(x, t)
    Projection::LinearForm L(mesh, f);
    L.assemble(b, step==0);
    solver.solve(A, g.vector(), b);
    g.sync();
    file << g;
    // Evaluate user-defined function f and the discrete function g (projection of f)
    f.eval(f_values, x);
    g.eval(g_values, x);
    message("t, f(x), g(x) = %g %g %g", t, f_values[0], g_values[0]);

    t +=tstep;
    step += 1;    
  }
  dolfin_finalize();
  return 0;
}
