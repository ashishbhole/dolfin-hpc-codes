// Copyright (C) 2010 Jeannette Spuehler.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Niclas Jansson 2017.
// Modified by Ashish Bhole 2023: 
// Modified from ConvectionDiffusion demo
//
// This is an IVP to solve linear advection equation.
// The initial condition is specified as a Gaussian function.
// It advects with constant velocity vector = {adv_x, adv_y}.
//
//     u_t + \mathbb{c} \cdot \nabla u = 0
//     u(x, 0) = exp(-alpha*( (x-x0)**2 + (y-y0)**2 ))
//     u_ex(x ,t) = exp(-alpha*( (x-adv_x*t-x0)**2 + (y-adv_y*t-y0)**2 ))

#define IO

#include "NavierStokes3D.h"
#include "NavierStokesContinuity3D.h"
#include "Gradient.h"
#include "Gradient_components.h"
#include "Residual.h"

#include <cmath> // For sin, cos and pi
#include <sstream> // For printing
#include <lapack.h> // For the Schur decomposition needed for the triple decomposition
#include <dolfin.h> // For everything else

using namespace dolfin;

real bmarg = 1.0e-3 + DOLFIN_EPS;

real Tfinal = 20.0;
real ubar = 1.0; // Free stream velocity
real ubar_max = 1.0;

real vortex_diameter = 1.0;

real viscosity = 0.0; // Could just remove the term in the form file instead

/*real xmin = 0.0; 
real xmax = 10.0;
real ymin = 0.0;
real ymax = 1.0;
real zmin = 0.0;
real zmax = 1.0;*/
real xmin = 0.0;
real xmax = 3.0;
real ymin = -2.0;
real ymax = 2.0;
real zmin = -2.0;
real zmax = 2.0;

// Sub domain for vortex sides
class VortexBoundary : public SubDomain
{ 
public:
  bool inside(const real* p, bool on_boundary) const
  {
    return on_boundary && (p[0] < xmin + bmarg || p[0] > xmax - bmarg); // &&
//           p[1] > ymin + bmarg && p[1] < ymax - bmarg &&
//           p[2] > zmin + bmarg && p[2] < zmax - bmarg;
  }
};

// Subdomain for setting initial condition
class VortexDomain : public SubDomain
{
public:
  bool inside(const real* p, bool on_boundary) const
  {
    return(std::abs(p[1]) <= vortex_diameter/2 && std::abs(p[2]) <= vortex_diameter);
  }
};

// Subdomain for walls
class WallBoundary : public SubDomain
{ 
public:
  bool inside(const real* p, bool on_boundary) const
  {
    return on_boundary && ((p[0] < (xmax - bmarg) && p[0] > (xmin + bmarg)) || // any boundary except inflow and outflow walls...
           ((p[0] > xmax - bmarg || p[0] < xmin + bmarg) && p[1] > ymax - bmarg || p[1] < ymin + bmarg || p[2] > zmax - bmarg || p[2] < zmin + bmarg)); // ...or corners on inflow and outflow walls
  }
};

// Subdomain for all walls. Not sure if this works
class AllWalls : public SubDomain
{
public:
  bool inside(const real* p, bool on_boundary) const
  {
    return on_boundary;
  }
};

// Subdomain for the horizontal slip boundaries
class XY_Plane : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && (p[2] > zmax - bmarg || p[2] < zmin + bmarg);
    }
};

// Subdomain for the vertical slip boundaries
class XZ_Plane : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && (p[1] > ymax - bmarg || p[1] < ymin + bmarg);
    }
};

// Subdomain for the vertical slip boundaries
class YZ_Plane : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && (p[0] > xmax - bmarg || p[0] < xmin + bmarg);
    }
};

// Boundary condition for inducing Taylor-Green vortices. Can be use on internal points too as initial condition
struct BC_Momentum : public Value<BC_Momentum, 3>
{
  void eval(real* value, const real* x) const
  {
    value[1] = std::sin((x[2] - vortex_diameter/2) * M_PI / vortex_diameter) * std::cos((x[1]) * M_PI / vortex_diameter);
    value[2] = -std::cos((x[2] - vortex_diameter/2) * M_PI / vortex_diameter) * std::sin((x[1]) * M_PI / vortex_diameter);
  }
};

// Boundary condition for continuity equation 
struct BC_Continuity : public Value<BC_Continuity, 1> // Is 2 the space dimension or the pressure dofs? Dofs I think
{
  void eval(real* value, const real* x) const
  {
    value[0] = 0.0;
  }
};

struct ZeroVelocity : public Value<ZeroVelocity, 3>
{
  void eval(real *value, const real *x) const
  {
    value[0] = 0.0;
    value[1] = 0.0;
    value[2] = 0.0;
  }
};

void ComputeStabilization(Mesh& mesh, Function& w, real nu, real k,
    Function& d1, Function& d2, Form& form)
//             Vector& d1vector, Vector& d2vector, Form& form)
{
  // Compute least-squares stabilizing terms: 
  //
  // if  h/nu > 1 or nu < 10^-10
  //   d1 = C1 * ( 0.5 / sqrt( 1/k^2 + |U|^2/h^2 ) )   
  //   d2 = C2 * h 
  // else 
  //   d1 = C1 * h^2  
  //   d2 = C2 * h^2  

  // FIXME: we should use k iff CFL < 1
  //  real kk = 0.25*hmin/ubar; 
  real kk = k;

  real C1 = 4.0; //4.0
  real C2 = 2.0;

  Cell c(mesh, 0);
  size_t local_dim = c.num_entities(0);
  size_t topological_dim = mesh.topology().dim();
  real *d1_block = new real[mesh.num_cells()];
  real *d2_block = new real[mesh.num_cells()];
  size_t *rows2 = new size_t[mesh.num_cells()];
  real *w_block = new real[topological_dim * local_dim * mesh.num_cells()];

  size_t *indices = new size_t[topological_dim * local_dim * mesh.num_cells()];

  size_t *ip = &indices[0];

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    UFCCell ufc_cell(*cell);

    (form.dofmaps())[0]->tabulate_dofs(ip, ufc_cell, *cell);

    ip += topological_dim * local_dim;
  }

  w.get_block(w_block);

  real *wp = &w_block[0];
  uint ci = 0;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    real normw = 0.0;

    for(uint i =0; i < local_dim ; i++)
    {
      // Which one should be used?
      normw += sqr(*(wp++));
      // normw += sqr( (*(wp++) + *(wp++) + *(wp++) + *(wp++)) / local_dim);
    }

    normw = sqrt(normw) / local_dim; // Only divide by local_dim if not done in the for loop above

    real h = (*cell).diameter();

    uint cid = 0;
    if( MPI::global_size() > 1)
    {
      cid = mesh.distdata()[mesh.topology().dim()].get_global((*cell).index());
    }
    else
    {
      cid = (*cell).index();
    }

    if((nu < 1.0e-10) || ((h/nu) > 1.0) )
//    if(false)
    {
      d1_block[ci] = C1 * (0.5 / sqrt( 1.0/sqr(kk) + sqr(normw/h)));
      d2_block[ci] = C2 * normw * h;
    }
    else
    {
      d1_block[ci] = C1 * h; //d1_block[ci] = C1 * sqr(h);
      d2_block[ci] = C2 * normw * h; //d2_block[ci] = C2 * sqr(normw) * sqr(h);
    }
    rows2[ci++] = cid;
  }

  d1.set_block(d1_block);
  d2.set_block(d2_block);

  delete[] d1_block;
  delete[] d2_block;
  delete[] rows2;
  delete[] w_block;
  delete[] indices;
}

real sign(real x)
{
  if(x > 0) return 1;
  return -1;
}

void ComputeMeanResidual(Mesh& mesh, Function& vmean, Function& v) //, Form& form)
{
  Cell cell_tmp(mesh, 0);
  size_t nsd = v.dim(0); // Hopefully gives me 3 for u and 1 for p
  uint local_dim = cell_tmp.num_entities(0);
  real *v_block = new real[nsd * local_dim * mesh.num_cells()];
  real *vmean_block = new real[mesh.num_cells()];
  v.get_block(v_block);

  uint mi = 0;
  real cellmean = 0.0;
  real cellnorm = 0.0;
  uint ri = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    cellnorm = 0.0;
    for (uint i = 0; i < nsd; i++)
    {
      cellmean = 0.0;
      for (VertexIterator n(*c); !n.end(); ++n)
      {
        cellmean += v_block[ri++];
      }
      cellmean /= local_dim;
      cellnorm += (cellmean * cellmean);
    }
    cellnorm = std::sqrt(cellnorm);
    vmean_block[mi++] = cellnorm;
  }
  vmean.add_block(vmean_block);
  vmean.sync();

  delete[] v_block;
  delete[] vmean_block;
}

// ChatGPT code. Leave this for now, try with unicorn code instead
// Refine the n% of cells that have the highest residual values
void ComputeRefinementMarkers(Mesh& mesh, Function& residuals, MeshValues<bool, Cell>& cell_markers, const real& percentage)
{
  std::vector<real> residual_vector(residuals.vector().local_size());  // Create a vector of the same size
//  residuals.vector()->get(residual_vector.data());                // Copy values into residuals
  residuals.vector().get(residual_vector.data());

  // Step 1: Gather residuals across all MPI processes
  std::vector<double> global_residuals(mesh.num_global_cells());
  dolfin::MPI::gather(residual_vector.data(), residual_vector.size(), global_residuals.data(), mesh.num_global_cells(), 0);

  // Step 1: Find the nth percentile
  double threshold = 0.0;
//  if (dolfin::MPI::rank() == 0)
//  {
//    size_t top_index = static_cast<size_t>((1.0 - percentage) * global_residuals.size()); // Index for the nth percentile
//    std::nth_element(global_residuals.begin(), global_residuals.begin() + top_index, global_residuals.end());
//    threshold = global_residuals[top_index];  // Residual value at the nth percentile
//  }

  // Step 3: Broadcast threshold to all ranks
  MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//  for(CellIterator c(mesh); !c.end(); ++c)
//  {
//    size_t global_index = mesh.distdata()[3].get_global(c->index());
//    size_t cell_index = c->index();
//    cell_markers(*c) = (residual_vector[cell_index] > threshold);
//  }
  File cell_marker_file("refinement_marker.pvd");
  cell_marker_file << cell_markers;
}

// Mesh refinement copied from unicorn
void MeshRefinement(Mesh& mesh, Function& residuals)
{
  // Define percentage of cells to refine. Could set the number somewhere else
//  dolfin_set("adapt_percentage", 0.1);
  dolfin_add<real>("adapt_percentage", 0.1);
  real percentage = dolfin_get<real>("adapt_percentage");
//  real percentage = 0.1;
  dolfin_add("adapt_algorithm", "rivara");

  // Mark cells for refinement
  MeshValues<bool, Cell> cell_marker(mesh);
  ComputeRefinementMarkers(mesh, residuals, cell_marker, percentage); // Not sure what this is, but assume it fills cell_marker

//  File cell_marker_file("refinement_marker.pvd");
//  cell_marker_file << cell_marker;

  // Refine cells
  AdaptiveRefinement::refine(mesh, cell_marker);
}

// Triple Decomposition helper functions

// Eigenvalue selection function
int tripleSelectFunction(const double* wr, const double* wi)
{
  // Select eigenvalues with imaginary part equal to 0. These will be placed
  // top left in the Schur matrix
  return (std::abs(*wi) <= 1e-10);
}

// Triple decomposition
void tripleDecomposition(real*& grad_u, real* shear, real* strain, real* rotation)
{
  int n = 3; // Order of the matrix A, here always 3

  // LAPACK function parameters
  char jobvs = 'N';                 // Compute Schur vectors ('V') or not ('N')
  char sort = 'S';                  // Sort eigenvalues ('S') or not ('N')
  std::vector<double> wr(n), wi(n); // Real and imaginary parts of eigenvalues
  int sdim;                         // Number of selected eigenvalues (ignored in this example)
  std::vector<double> vs(n * n);    // Schur vectors (ignored in this example)
  std::vector<double> work(8 * n);  // Workspace
  int lwork = 8 * n;                // Size of workspace
  std::vector<int> bwork(n);        // Boolean workspace
  int info;                         // Status indicator

  // Call LAPACK's dgees function for eigenvalue decomposition. Arguments:
  // JOBVS: compute Schur vectors ("V") or not ("N")
  // SORT: sort eigenvalues ("S") or not ("N")
  // SELECT: logical (?), how to sort the eigenvalues
  // N: order of the matrix A, here always 3
  // A: [in and out] the matrix A, replaced by T by dgees_
  // LDA: the other dimension of a, here always 3
  // SDIM: [out] number of eigenvalues for which SELECT is true
  // WR: [out] real part of eigenvalues
  // WI: [out] imaginary part of data
  // VS: [out] the U matrix
  // LDVS: leading dimension of Q, here always 3 (but typically unused)
  // WORK: [out] not sure what this is
  // LWORK: dimension of WORK, not sure what it is
  // BWORK: [out] logical (?), not sure what it is
  // INFO: [out] not sure what this is
  dgees_(&jobvs, &sort, tripleSelectFunction, &n, grad_u, &n, &sdim,
    wr.data(), wi.data(), vs.data(), &n, work.data(), &lwork, bwork.data(), &info);

  if(false)
//  if(dolfin::MPI::rank() == 64 && grad_u[0] > bmarg || grad_u[1] > bmarg || grad_u[2] > bmarg || grad_u[3] > bmarg || grad_u[4] > bmarg || grad_u[5] > bmarg || grad_u[6] > bmarg || grad_u[7] > bmarg || grad_u[8] > bmarg)
  {
    std::cout << grad_u[0] << ", " << grad_u[1] << ", " << grad_u[2] << std::endl
              << grad_u[3] << ", " << grad_u[4] << ", " << grad_u[5] << std::endl
              << grad_u[6] << ", " << grad_u[7] << ", " << grad_u[8] << std::endl << std::endl;
  }

  // Compute sh, el, rr
  *shear = std::sqrt(std::pow(grad_u[3], 2) + std::pow(grad_u[6], 2) + std::pow(grad_u[5] + grad_u[7], 2));
  *strain = std::sqrt(std::pow(grad_u[0], 2) + std::pow(grad_u[4], 2) + std::pow(grad_u[8], 2));
  *rotation = std::sqrt(2 * std::pow(std::min(std::abs(grad_u[5]), std::abs(grad_u[7])), 2));
}

// Inverse cell volume, needed for gradient for triple decomposition
void ComputeVolInv(Mesh& mesh, Function& vol_inv)
{
  vol_inv.vector().init(mesh.num_cells());

  real* icvarr = new real[vol_inv.vector().local_size()];

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    icvarr[cell->index()] = 1.0 / (cell->volume());
  }
  vol_inv.vector().set(icvarr);
  vol_inv.vector().apply();

  delete[] icvarr;
}

// Triple decomposition
void computeTripleDecomposition(Mesh& mesh, Gradient::BilinearForm& aGrad, Gradient::LinearForm& LGrad, Function& u, Function& triple_shear, Function& triple_strain, Function& triple_rotation)
{
//  Vector gradU;
  Function gradU(aGrad.trial_space());
  LGrad.assemble(gradU.vector(), false);
  gradU.vector().apply();
  gradU.sync();

  uint d = mesh.topology().dim();
  Cell c(mesh, 0);
  uint local_dim = c.num_entities(0);
  size_t *idx  = new size_t[d * local_dim];
  real *gradU_block = new real[d * d];
  size_t global_index;
  real shear, strain, rotation;
  // Triple decomposition is computed on cell level
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  { 
    UFCCell ufc_cell(*cell);
    (aGrad.dofmaps())[0]->tabulate_dofs(idx, ufc_cell, *cell);
    gradU.vector().get(gradU_block, d*d, idx);
    global_index = mesh.distdata()[d].get_global(cell->index());
    // Compute the triple decomposition
    tripleDecomposition(gradU_block, &shear, &strain, &rotation);
    triple_shear.vector().set(&shear, 1, &global_index);
    triple_strain.vector().set(&strain, 1, &global_index);
    triple_rotation.vector().set(&rotation, 1, &global_index);
  }
  triple_shear.vector().apply();
  triple_shear.sync();
  triple_strain.vector().apply();
  triple_strain.sync();
  triple_rotation.vector().apply();
  triple_rotation.sync();
  delete[] idx;
  delete[] gradU_block;

//  if(dolfin::MPI::rank() == 64)
//    for(int i = 0; i < gradU.vector().size(); i++)
//      std::cout << gradU.vector()[i];
  File file("gradient.pvd");
  file << gradU;
}

int main(int argc, char* argv[])
{
  // Initialize
  dolfin_init(argc, argv);

  // Print normal types for debugging purposes
  dolfin_set("NodeNormal dump types", true);

  // Read mesh
  Mesh mesh("box_mesh.bin");

  // Print the mesh to new file, needed for dolfin-post
  File meshfile("meshfile.bin");
  meshfile << mesh;

  // Set time step (proportional to the minimum cell diameter) 
  real hmin = 1.0e6;
  real hminlocal = 1.0e6;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if ((*cell).diameter() < hminlocal) hminlocal = (*cell).diameter();
  }
  MPI_Barrier(dolfin::MPI::DOLFIN_COMM);
  MPI_Allreduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, dolfin::MPI::DOLFIN_COMM);
  real tstep = 0.15*hmin/ubar; // CFL constant 0.15 taken from unicorn icns 3D cylinder
  message("hmin = %g", hmin);
  message("time step size: %g", tstep);


  // Set boundary conditions
  Analytic<ZeroVelocity> zero_velocity(mesh);
  Analytic<BC_Continuity> zero_1d(mesh);
  Analytic<BC_Momentum> boundary_vortex(mesh);
  
  // Define boundaries
  VortexBoundary vortex_boundary; // Only boundary
  VortexDomain vortex_domain; // Internal points too
  WallBoundary wall_boundary;
  AllWalls all_walls_boundary;
  XY_Plane xy_plane;
  XZ_Plane xz_plane;
  YZ_Plane yz_plane;

  // SlipBC sometimes crashes when running on too many cores. Instead, use SubSystem to set u=0 in specific directions
  SubSystem subsystem_x(0);
  SubSystem subsystem_y(1);
  SubSystem subsystem_z(2);

  // Pressure condition: p=0 on all walls
  DirichletBC wall_con_bc(zero_1d, mesh, wall_boundary);
  // Momentum condition: u=0 in the normal direction
  DirichletBC xy_slip_bc(zero_1d, mesh, xy_plane, subsystem_z);
  DirichletBC xz_slip_bc(zero_1d, mesh, xz_plane, subsystem_y);
  DirichletBC yz_slip_bc(zero_1d, mesh, yz_plane, subsystem_x);


  // Put the boundary conditions in vectors
  std::vector<BoundaryCondition*> bc_con;
  bc_con.push_back(&wall_con_bc);
  std::vector<BoundaryCondition*> bc_mom;
  bc_mom.push_back(&xy_slip_bc);
  bc_mom.push_back(&xz_slip_bc);
  bc_mom.push_back(&yz_slip_bc);

  // Initial velocity for whole domain. Defined as a DirichletBC, but applied to internal points too
  DirichletBC vortex_initial(boundary_vortex, mesh, vortex_domain);


  // Set up functions. Not sure what should be Constant or Function
  Constant dt(tstep);
  Constant nu(viscosity);
  Function d1(mesh);
  Function d2(mesh);
  Function up(mesh);

  // Create forms
  NavierStokes3D::BilinearForm a_mom(mesh, up, nu, d1, d2, dt);
  NavierStokesContinuity3D::BilinearForm a_con(mesh, d1);
  Function u(a_mom.trial_space());
  Function u0(a_mom.trial_space());
  Function p(a_con.trial_space());
  Function p0(a_con.trial_space());
  NavierStokes3D::LinearForm L_mom(mesh, up, u0, p, nu, d1, d2, dt);
  NavierStokesContinuity3D::LinearForm L_con(mesh, u, u0); //, d1);
  PETScMatrix A_mom, A_con;
  Vector b_mom, b_con;

  // Initialize vectors for the time step residuals of
  // the momentum and continuity equations
  size_t n = mesh.num_vertices();
  if(MPI::global_size() > 1) n -= mesh.distdata()[0].num_ghost();
  Vector residual_mom(mesh.topology().dim()*n);
  Vector residual_con(n);

  // Initialize algebraic solvers   
  KrylovSolver solver_con(gmres, amg);
  KrylovSolver solver_mom(gmres, amg);

  // Sync ghosts of everything. Not sure if this is needed here, or why
  u.sync();  // velocity
  u0.sync(); // velocity from previous time step 
  up.sync(); // velocity linearized convection. Basically just velocity from the previous iteration
//  um.sync(); // cell mean velocity
  p.sync();  // pressure
  p0.sync(); // pressure
  d1.sync(); // stabilization coefficient
  d2.sync(); // stabilization coefficient

  // Compute stabilization parameters d1 and d2
  ComputeStabilization(mesh, u0, viscosity, tstep, d1, d2, L_mom);


  // Time stepping and iteration parameters
  double t = 0.0;
  uint step = 0;
  real residual, residual2;
  real residual_c, residual_m;
  real rtol = 1.0e-2;//1.0e-2;
  real rtol2 = 1.0e-3;
  int iteration;
  int max_iteration = 10; //50;

  // Initialize residual function, to be used for adaptive mesh refinement
  Function residual_function(mesh); // Should I tie this to the trial function of Residual, or just assemble LRes into residual_function.vector()?
  Function residual_cell(mesh); // Residual on cell level

  Residual::LinearForm LRes(mesh, u, u0, p, dt);

  residual_function.init(LRes.create_coefficient_space("U")); // Get it as linear 3D vector first, then average on cell level
  residual_cell.init(LRes.create_coefficient_space("k"));

  // Initialize Triple Decomposition functions
  Function triple_shear(mesh);
  Function triple_strain(mesh);
  Function triple_rotation(mesh);

  Function vol_inv(mesh);

  // Assembling L=inner(grad(u),v) outputs grad(u) directly, no need to solve anything. LGrad also only needs to be created once
  Gradient::BilinearForm aGrad(mesh);
  Gradient::LinearForm LGrad(mesh, u, vol_inv);

  // Initialize functions with the appropriate FE space
  vol_inv.init(LGrad.create_coefficient_space("icv"));
  triple_shear.init(LGrad.create_coefficient_space("icv"));
  triple_strain.init(LGrad.create_coefficient_space("icv"));
  triple_rotation.init(LGrad.create_coefficient_space("icv"));

  triple_shear = 0;

  // Compute inverse of mesh cell volumes, needed to compute the triple decomposition
  ComputeVolInv(mesh, vol_inv);

  // Output file. Should ideally be bin, but that doesn't work with pw constant cell values, i.e. triple components
  File solutionfile("solution.pvd");
  LabelList<Function> output;
  Label<Function> u_output(u, "Velocity");
  Label<Function> p_output(p, "Pressure");
  Label<Function> sh_output(triple_shear, "Shear");
  Label<Function> el_output(triple_strain, "Strain");
  Label<Function> rr_output(triple_rotation, "Rotation");
  Label<Function> res_output(residual_cell, "Residual");
  Label<Function> icv_output(vol_inv, "Inverse Volume");
  output.push_back(u_output);
  output.push_back(p_output);
  output.push_back(sh_output);
  output.push_back(el_output);
  output.push_back(rr_output);
  output.push_back(res_output);
  output.push_back(icv_output);

  // Debugging files
//  File residual_file("residual.pvd");
//  residual_file << output;

  // write the initial condition to the solution file
  #ifdef IO
  solutionfile << output;
  #endif

  int runs = 5;

  // Time stepping!
  for(int i = 0; i < runs; i++)
  {
  while(t < Tfinal)
  {
    dt = (t+tstep > Tfinal ? Tfinal-t : dt);

    // Initialize residuals
    residual = 2*rtol;
    residual2 = 2*rtol2;
    iteration = 0;

    // Fix-point iteration for non-linear problem 
    while((residual > rtol && iteration < max_iteration) ||
          (residual2 > rtol2 && iteration < max_iteration) ||
          iteration < 1)
    {
      // Set linearized velocity to current velocity
      up = u;
      p0 = p; // p0 is pressure from last iteration? I thought from last time step
      // NSESolver syncs here, but it's already done for u, isn't that enough?
      up.sync();
      p0.sync();

      ComputeStabilization(mesh, u0, viscosity, tstep, d1, d2, L_mom); //TODO: This should not be commented out

      a_con.assemble(A_con, step == 0);
      L_con.assemble(b_con, step == 0);

      for(uint i = 0; i < bc_con.size(); i++)
      {
        bc_con[i]->apply(A_con, b_con, a_con);
      }

      // Solve the continuity equation
      solver_con.solve(A_con, p.vector(), b_con);
      p.sync(); // Ashish syncs after solving, NSESolver does not. I think it makes sense

      // Assemble momentum vector
      a_mom.assemble(A_mom, step == 0);
      L_mom.assemble(b_mom, step == 0);

      if(step == 0)
      {
        // Set initial condition. This is easier than initial conditions the normal way
        vortex_initial.apply(A_mom, u.vector(), a_mom);
        break;
      }
      else
      {
        for(uint i = 0; i < bc_mom.size(); i++)
        {
          bc_mom[i]->apply(A_mom, b_mom, a_mom);
        }
      }

      // Solve the momentum equation
      solver_mom.solve(A_mom, u.vector(), b_mom);
      u.sync();

      // Residual2 moved here
      residual_mom = 0;
      A_mom.mult(u.vector(), residual_mom);
      residual_mom -= b_mom;
      A_con.mult(p.vector(), residual_con);
      residual_con -= b_con;
  
      residual2 = sqrt(sqr(residual_mom.norm()) + sqr(residual_con.norm()));

      // Compute more residuals
      residual = 0;

      residual_con = p.vector();
      residual_con -= p0.vector();
      residual_c = 0;
      if(p.vector().norm(linf) > 1.0e-8)
      {
        residual_c = residual_con.norm(l2) / p.vector().norm(l2);
        residual += residual_c;
      }

      residual_mom = u.vector();
      residual_mom -= up.vector(); // Comparing u after solve before murtazo shift with the same from last time step
      residual_m = 0;
      if(u.vector().norm(linf) > 1.0e-8)
      {
        residual_m = residual_mom.norm(l2) / u.vector().norm(l2);
        residual += residual_m;
      }

      message("residual_mom: %g", residual_mom.norm(l2));
      message("residual2: %e", residual2);
      message("residual: %g", residual);
      iteration++;

    } // Fix-point iteration for non-linear problem closed

    // Compute triple decomposition
//    computeTripleDecomposition(mesh, aGrad, LGrad, u, triple_shear, triple_strain, triple_rotation);

    // Gradient testing
    Gradient_components::BilinearForm aGrad_x(mesh);
    Gradient_components::LinearForm LGrad_x(mesh, *(u.decompose()[0]), vol_inv);
    Gradient_components::LinearForm LGrad_y(mesh, *(u.decompose()[1]), vol_inv);
    Gradient_components::LinearForm LGrad_z(mesh, *(u.decompose()[2]), vol_inv);
    Function gradU_x(aGrad_x.trial_space());
    Function gradU_y(aGrad_x.trial_space());
    Function gradU_z(aGrad_x.trial_space());
    LGrad_x.assemble(gradU_x.vector(), false);
    LGrad_y.assemble(gradU_y.vector(), false);
    LGrad_z.assemble(gradU_z.vector(), false);
    File grad_x_file("gradient_x.pvd");
    File grad_y_file("gradient_y.pvd");
    File grad_z_file("gradient_z.pvd");
    grad_x_file << gradU_x;
    grad_y_file << gradU_y;
    grad_z_file << gradU_z;


    // Compute residual
    LRes.assemble(residual_function.vector(), false); // false means reassemble, which we always want?
    ComputeMeanResidual(mesh, residual_cell, residual_function);

    u0 = u;
    t += tstep;
    step++;
    #ifdef IO
    real prints_per_sec = 10; // How many times to print per simulated second
    if(step < 10 || std::floor(prints_per_sec*t) > std::floor(prints_per_sec*(t-tstep))) // Print the 100 first timesteps, then some times per simulated second
    {
      solutionfile << output;
    }
    #endif
    message("------------------------------------ Step %d finished ------------------------------------", step);
  }
  // Refine mesh, then reset simulation time and solution
  MeshRefinement(mesh, residual_cell);
  t = 0.0;
  step = 0;
  // Do Functions need to be reinitiated?
  u.zero();
  u0.zero();
  p.zero();
  }

  dolfin_finalize();
  return 0;
}

