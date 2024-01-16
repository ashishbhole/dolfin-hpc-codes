// Written by Ashish Bhole (2023) 
//
// This is a linear wave equation equation.
//
//     u_tt + c^2 div( grad(u)) = f

#define IO

#include "WaveEquation.h"
#include <sstream>
#include <dolfin.h>
#include <fstream>

using namespace dolfin;

real tstep = 0.01;
real speed = 1.0;
real Tfinal = 100.0;
real Nc = 0.01;

// Analytic function to specify the initial condition and exact solution.
struct InitialCondition : public Value< InitialCondition, 1 >
{
  InitialCondition() : alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    values[0] = 0.0;
  }
  double alpha;
};

struct Source : public Value< Source, 1 >
{
  Source(): t(0), alpha(32.0) {}	
  void eval( real * values, const real * x ) const
  {
    if( (sqrt(x[0]*x[0] + x[1]*x[1]) ) <= 3.6775 + abs(DOLFIN_EPS) )
    {	    
      values[0] = sin(10.0*DOLFIN_PI*t);
    }
    else
    {
      values[0] = 0.0;
    }
  }
  double t, alpha;
};

struct DirichletBoundary : public SubDomain
{
  bool inside(const real* x, bool on_boundary) const
  {
    return on_boundary;
  }
};

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
File file("solution.pvd");
std::ofstream outfile ("signals.txt");

int main(int argc, char **argv)
{ 
  dolfin_init(argc, argv);

  real rec1[3] = { 4.0, 0.5, 0.0};
  real rec2[3] = {-4.0, 0.5, 0.0};
  real u_values1[1] = {0.0};
  real u_values2[1] = {0.0};

  real val1, val2;

  // Parallel file writing does not work with in-built meshes.
  Mesh mesh("disc_in_disc.bin");
  real t = 0.0;

  waveequation_finite_element_0 FE;
  uint m = FE.degree();
  real h = pow(compute_inradius(mesh), m);
  
  Constant u0(0.0);
  DirichletBoundary boundary;
  DirichletBC bc(u0, mesh, boundary);

  Source src;
  InitialCondition Gaussian;
  Gaussian.alpha = 128.0;
  Analytic<InitialCondition> ui( mesh, Gaussian);

  tstep = Nc * h / speed;
  Constant dt(tstep);
  Constant c(speed);
  WaveEquation::BilinearForm a(mesh, c, dt);
  Function u(a.trial_space());
  Function un(a.trial_space());
  Function u_old(a.trial_space());
  FunctionInterpolation::compute(ui, u_old);
  Matrix A;
  Vector b;
  a.assemble(A, true);
  KrylovSolver solver(bicgstab, bjacobi);

  uint step = 0;
  while (t < Tfinal)
  {
    src.t = t;
    Analytic<Source> ff( mesh, src);
    WaveEquation::LinearForm L(mesh, un, u_old, ff, dt);
    L.assemble(b, step==0);
    bc.apply(A, b, a);
    solver.solve(A, u.vector(), b);
    u.sync();
    u_old = un;
    un = u;
    #ifdef IO
      if (step%100 == 0) file << u; 
    #endif

    u.eval(u_values1, rec1);
    u.eval(u_values2, rec2);

    val1 = u_values1[0];
    val2 = u_values2[0];

    if(mesh.is_distributed())
    {
      MPI::reduce< MPI::min >( &u_values1[0], &val1, 1, 0, MPI::DOLFIN_COMM );
      MPI::reduce< MPI::min >( &u_values2[0], &val2, 1, 0, MPI::DOLFIN_COMM );
    }
    outfile << t << " " << val1 << " " << val2 << "\n";

    t +=tstep;
    step += 1;
  }
  outfile.close();  
  message( "t, h, l1, l2, linf norm: %e %e %e %e", t, h, u.vector().norm(l1), u.vector().norm(l2), u.vector().norm(linf) );
  dolfin_finalize();
  return 0;
}
