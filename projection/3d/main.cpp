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
    values[0] = exp(t) * cos(2.0 * DOLFIN_PI * x[0]) * cos(2.0 * DOLFIN_PI * x[1]) * cos(2.0 * DOLFIN_PI * x[2]) ; 
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
  real x[3] = {0.0, 0.0, 0.0}; // -0.001035-0.00053};
  real f_values[1] = {0.0};
  real u_values[1] = {0.0};
  real uint_values[1] = {0.0};

  Fun function;
  real val, val_int;

  // Project to a discrete function
  Projection::BilinearForm a(mesh);
  Function u(a.trial_space());
  Function u_int(a.trial_space());

  // solve PDE
  Matrix A;
  Vector b;
  a.assemble(A, true);

  KrylovSolver solver(bicgstab, bjacobi);

  File( "function.pvd" ) << u;

  uint step = 0;
  while (t < Tfinal)
  {
    function.t = t;
    Analytic<Fun> f(mesh, function);
    Projection::LinearForm L(mesh, f);
    L.assemble(b, step == 0);
    solver.solve(A, u.vector(), b);
    u.sync();

    FunctionInterpolation::compute(f, u_int);
    
    f.eval(f_values, x);
    u.eval(u_values, x);
    u_int.eval(uint_values, x);

    val = u_values[0];
    val_int = uint_values[0];

    // MPI reduction is needed as the point could be in any rank.
    // Reduction is supposed to pick the real number amongst 'inf's
    if(mesh.is_distributed())
    {
      MPI::reduce< MPI::min >( &u_values[0], &val, 1, 0, MPI::DOLFIN_COMM );
      MPI::reduce< MPI::min >( &uint_values[0], &val_int, 1, 0, MPI::DOLFIN_COMM );
    }

    message("time, f(x), fint(x), g(x) = %e %e %e %e", t, f_values[0], val_int, val);

    t +=tstep;
    step += 1;
  }
  dolfin_finalize();
  return 0;
}
