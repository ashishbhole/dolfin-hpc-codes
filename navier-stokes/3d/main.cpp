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

#include "NavierStokes3D_force.h"
#include "NavierStokesContinuity3D.h"
#include "Gradient.h"
#include "Residual.h"
#include "Project_DG0_to_CG1.h"

#include <sstream>
#include <lapack.h>
#include <dolfin.h>

using namespace dolfin;

real bmarg = 1.0e-3 + DOLFIN_EPS;

real Tfinal = 0.05*100;
real ubar = 1.0; // Free stream velocity
real ubar_max = 1.0;

// Re = ubar * radius*2 / viscosity
real viscosity = 0.000001; //0.0001;


real xmin = 0.0; 
real xmax = 2.1;
real ymin = 0.0;
real ymax = 1.4;
real zmin = 0.0;
real zmax = 0.4;
/*real xmin = 0.0;
  real xmax = 4.0;
  real ymin = 0.0;
  real ymax = 1.0;
  real zmin = 0.0;
  real zmax = 1.0;*/
real xcenter = 0.5;
real ycenter = 0.7;
real radius = 0.05;


// Sub domain for outflow
class OutflowBoundary : public SubDomain
{ 
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && (p[0] > (xmax - bmarg));
      //    return (p[0] == xmax && p[1] == ymax && p[2] == zmax);
      //           && p[1] > (ymin + bmarg) && p[1] < (ymax - bmarg) &&
      //           p[2] > (zmin + bmarg) && p[2] < (zmax - bmarg));
    }
};

// Sub domain for inflow
class InflowBoundary : public SubDomain
{ 
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && p[0] < xmin + bmarg; // &&
                                                 //           p[1] > ymin + bmarg && p[1] < ymax - bmarg &&
                                                 //           p[2] > zmin + bmarg && p[2] < zmax - bmarg;
    }
};

// Hacky subdomain for setting initial condition on the entire domain
class WholeDomain : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return true;
    }
};

// Sub domain for slip. Doesn't specify any boundary, just trusting on_boundary argument?
class SlipBoundary : public SubDomain
{ 
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && ((p[0] < (xmax - bmarg) && p[0] > (xmin + bmarg)) || // any boundary except inflow and outflow walls...
          ((p[0] > xmax - bmarg || p[0] < xmin + bmarg) && p[1] > ymax - bmarg || p[1] < ymin + bmarg || p[2] > zmax - bmarg || p[2] < zmin + bmarg)); // ...or corners on inflow and outflow walls
    }
};

// Subdomain for the horizontal slip boundaries
class HorizontalSlipBoundary : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && (p[1] > ymax - bmarg || p[1] < ymin + bmarg);
    }
};

// Subdomain for the vertical slip boundaries
class VerticalSlipBoundary : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary && (p[2] > zmax - bmarg || p[2] < zmin + bmarg);
    }
};

// Subdomain for the cylinder boundary
class CylinderSlipBoundary : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
//      if(((p[0] - xcenter)*(p[0] - xcenter) + (p[1] - ycenter)*(p[1] - ycenter)) - bmarg < (radius * radius) && p[2] < zmax - bmarg && p[2] > zmin + bmarg)
      //    return //on_boundary &&
      if(((p[0] - xcenter)*(p[0] - xcenter) + (p[1] - ycenter)*(p[1] - ycenter)) - bmarg < (radius * radius)) // All of the cylinder, including on the walls
//           (p[0] < xmax - bmarg && p[0] > xmin + bmarg) && (p[1] < ymax - bmarg && p[1] > ymin + bmarg) && (p[2] < zmax - bmarg && p[2] > zmin + bmarg);
      {
        //      std::cout << p[0] << ", " << p[1] << ", " <<p[2] << ", " << std::endl;
        return true;
      }
    }
};

// Sub domain for Channel side walls
class SideWallBoundary : public SubDomain
{
  public:
    bool inside(const real* p, bool on_boundary) const
    {
      return on_boundary &&
        (p[0] > xmin + bmarg && p[0] < xmax - bmarg) &&
        ((p[1] < ymin + bmarg || p[1] > ymax - bmarg) ||
         (p[2] < zmin + bmarg || p[2] > zmax - bmarg));
    }
};

// Sub domain for Periodic boundary condition
/*struct PeriodicBoundary : public PeriodicSubDomain
  {
  bool inside(const real * p, bool on_boundary) const
  {
  return on_boundary && (p[2] > zmax - bmarg || p[2] < zmin + bmarg);
  }

  void map( const real * xH, real * xG ) const
  {
  xG[0] = xH[0];
  xG[1] = xH[1];
  xG[2] = xH[2] - (zmax - zmin);
  }
  };

  class PeriodicBC1 : public PeriodicBC
  {
  public:
  PeriodicBC1( Mesh & mesh, PeriodicSubDomain const & sub_domain )
  : PeriodicBC( mesh, sub_domain )
  {
  }

  inline void sync( Time const & t )
  {
  }
  };*/

/*
// Boundary condition for momentum equation. It's not called Function in 0.9, what should it be?
class BC_Momentum : public Function
{
public:
BC_Momentum(Mesh& mesh) : Function(mesh) {}
void eval(real* values, const real* x) const
{
if(x[0] < xmin + bmarg) {
values[0] = 1.0;
values[1] = 0.0;
}
else {
values[0] = 0.0;
values[1] = 0.0;
}

}
};*/

// Why would the boundary have to be considered as in the function above? This should be enough?
struct BC_Momentum : public Value<BC_Momentum, 3>
{
  void eval(real* value, const real* x) const
  {
    if(x[0] < xmin + bmarg)
    {
      value[0] = ubar;
      value[1] = 0.0;
      value[2] = 0.0;
    }
    else
    {
      value[0] = 0.0;
      value[1] = 0.0;
      value[2] = 0.0;
    }
  }
};

struct BC_Poiseuille : public Value<BC_Poiseuille, 3>
{
  void eval(real* value, const real* x) const
  {
    value[0] = (1 - x[1]) * x[1] * (1 - x[2]) * x[2] * 8;
    value[1] = 0.0;
    value[2] = 0.0;
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

struct InitialVelocity : public Value<InitialVelocity, 3>
{
  void eval(real *value, const real *x) const
  {
    value[0] = 0.0;
    value[1] = 0.0;
    value[2] = 0.0;
  }
};

class InitialVelocityFunction : public Function
{
  public:
    InitialVelocityFunction(FiniteElementSpace const & space) : Function(space) {}

    void eval(real *value, const real *x) const
    {
      value[0] = (1 - x[1]) * x[1] * (1 - x[2]) * x[2] * 8;
      value[1] = 0.0;
      value[2] = 0.0;
    }
};

struct BC_DirichletSlip : public Value<BC_DirichletSlip, 3>
{
  void eval(real *value, const real *x) const
  {
    /*    if(x[1] > ymax - bmarg || x[1] < ymin + bmarg)
          {
          value[1] = 0;
          }
          if(x[2] > zmax - bmarg || x[1] < zmin + bmarg)
          {
          value[2] = 0;
          }*/
  }
};

struct InitialPressure : public Value<InitialPressure, 1>
{
  void eval(real *value, const real *x) const
  {
    value[0]= 0.0;
  }
};

// Attempt to implement Murtazo's slip by overloading GenericFunction
//class BC_Murtazo_Slip;  // Forward declaration
class BC_Murtazo_Slip : public Analytic<Zero<>> // Zero as dummy template argument that implements unused eval()
{
  public:
    // Constructor to initialize reference to velocities and normals
    explicit BC_Murtazo_Slip(Mesh & mesh, NodeNormal & normals, const Function * up = nullptr)
      : Analytic<Zero<>>(mesh), normals_(normals), up_(up) {}

    // Override the evaluate function
    auto evaluate(real* values, const real* coordinates, const ufc::cell& cell) const -> void override
    {
      // Get u, up and normal for the Vertex defined by coordinates
      //      real* u_local;
      real up_local[3];
      real normal_local[3];

      //      u_->evaluate(u_local, coordinates, cell);
      up_->evaluate(up_local, coordinates, cell);
      normals_.basis()[0].evaluate(normal_local, coordinates, cell);

      // Dot product up*n
      real upn = up_local[0] * normal_local[0] + up_local[1] * normal_local[1] + up_local[2] * normal_local[2];

      for(int i = 0; i < 3; i++)
      {
        values[i] = up_local[i] - upn * normal_local[i];
      }
    }

    void set_velocities(const Function * up)
    {
      //      u_ = u;
      up_ = up;
    }

    // Necessary functions
    auto rank() const -> size_t override { return 1; }
    auto dim( size_t i ) const -> size_t override {return 3; }

  private:
    //    const Function * u_;
    const Function * up_;
    NodeNormal & normals_;
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

  //  UFC ufc(form.form(), mesh, form.dofMaps());
  //  UFCCell ufc_cell;

  Cell c(mesh, 0);
  size_t local_dim = c.num_entities(0);
  size_t topological_dim = mesh.topology().dim();
  real *d1_block = new real[mesh.num_cells()];
  real *d2_block = new real[mesh.num_cells()];
  size_t *rows2 = new size_t[mesh.num_cells()];
  real *w_block = new real[topological_dim * local_dim * mesh.num_cells()];

  //  if(!indices)
  //  {
  size_t *indices = new size_t[topological_dim * local_dim * mesh.num_cells()];

  size_t *ip = &indices[0];

  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    //      ufc.update(*cell, mesh.distdata());
    //      ufc_cell.update(*cell);
    UFCCell ufc_cell(*cell);

    (form.dofmaps())[0]->tabulate_dofs(ip, ufc_cell, *cell);

    ip += topological_dim * local_dim;
  }
  //  }

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

void ComputeMean(Mesh& mesh, Function& vmean, Function& v) //, Form& form)
{
  Cell cell_tmp(mesh, 0);
  //  uint nsd = mesh.topology().dim();
  size_t nsd = v.dim(0); // Hopefully gives me 3 for u and 1 for p
  uint local_dim = cell_tmp.num_entities(0);
  //  UFC ufc(form.form(), mesh, form.dofMaps());
  //  UFCCell ufc_cell;
  real *v_block = new real[nsd * local_dim * mesh.num_cells()];
  real *vmean_block = new real[nsd*mesh.num_cells()];
  /*
  //  if(!c_indices)
  //  {
  size_t *c_indices = new size_t[nsd * mesh.num_cells()];

  size_t *cip = &c_indices[0];
  for(CellIterator c(mesh); !c.end(); ++c)
  {
  //      ufc.update(*c, mesh.distdata());
  ufc_cell.update(*c);
  //      (form.dofMaps())[2].tabulate_dofs(cip, ufc.cell, c->index());
  (form.dofmaps())[2]->tabulate_dofs(cip, ufc_cell, *c);

  cip += nsd;
  }
  //  }
   */
  //  size_t *indices = new size_t[nsd * local_dim * mesh.num_cells()]; // TODO: indices is a member variable in NSESolver, this ok instead? Probably not ok, it should be the indices of the vertices we want to take the data from. So it needs to be the one from ComputeStabilization
  //  v.vector().get(v_block, nsd * local_dim * mesh.num_cells(), indices);
  v.get_block(v_block);

  uint mi = 0;
  real cellmean = 0.0;
  uint ri = 0;
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (uint i = 0; i < nsd; i++)
    {
      cellmean = 0.0;
      for (VertexIterator n(*c); !n.end(); ++n)
      {
        cellmean += v_block[ri++];
      }
      cellmean /= local_dim;
      vmean_block[mi++] = cellmean;
    }
  }
  //  vmean.vector().set(vmean_block,nsd * mesh.num_cells(), c_indices);
  //  vmean.vector().apply();
  // TODO: Why no sync?
  vmean.set_block(vmean_block);
  vmean.sync();

  delete[] v_block;
  delete[] vmean_block;
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
  // Get local number of cells
  int local_num_cells = mesh.num_cells();

  // Copy local residuals to array, needed for Gatherv
  real* local_residuals = new real[local_num_cells];
  residuals.vector().get(local_residuals);

  // Step 1: Gather residuals across all MPI processes
  // Initialize global residual vector, only to be used on rank 0
  std::vector<real> global_residuals;
  if (dolfin::MPI::rank() == 0)
    global_residuals.resize(mesh.num_global_cells());

  // Gather local number of cells from all ranks into recv_counts on rank 0, and compute
  // their displacements in global_residuals
  int *recv_counts = new int[dolfin::MPI::global_size()];
  int *displs = new int[dolfin::MPI::global_size()];
  dolfin::MPI::gather(&local_num_cells, 1, recv_counts, 1, 0, dolfin::MPI::DOLFIN_COMM);
  displs[0] = 0;
  for(int i = 1; i < dolfin::MPI::global_size(); i++)
  {
    displs[i] = displs[i-1] + recv_counts[i-1];
  }

  // Gather local residuals from all ranks into global_residuals on rank 0
  MPI_Gatherv(local_residuals, local_num_cells, MPI_DOUBLE, global_residuals.data(), recv_counts, displs, MPI_DOUBLE, 0, dolfin::MPI::DOLFIN_COMM);
  MPI_Barrier(dolfin::MPI::DOLFIN_COMM);
  message("Successfully gathered all residuals into global vector on rank 0");

  // Step 2: Find the threshold for the requested percentile
  double threshold = 0.0;
  if (dolfin::MPI::rank() == 0)
  {
    size_t top_index = static_cast<size_t>((1.0 - percentage) * global_residuals.size()); // Index for the nth percentile
    std::nth_element(global_residuals.begin(), global_residuals.begin() + top_index, global_residuals.end());
    threshold = global_residuals[top_index];  // Residual value at the nth percentile
  }

  // Step 3: Broadcast threshold to all ranks
  MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  message("Residual threshold %g computed for the %.2f percentile", threshold, 100*percentage);

  // Step 4: Mark all cells with residual higher than the threshold value
  for(CellIterator c(mesh); !c.end(); ++c)
  {
    //size_t global_index = mesh.distdata()[3].get_global(c->index());
    size_t cell_index = c->index();
    cell_markers(*c) = (local_residuals[cell_index] > threshold);
  }

  // TODO: Printing refinement markers and residuals to file, can remove or condition this later
  File cell_marker_file("refinement_marker.pvd");
  cell_marker_file << cell_markers;
  File residual_file("residuals.pvd");
  residual_file << residuals;

  delete[] local_residuals;
  delete[] recv_counts;
  delete[] displs;
}

// Mesh refinement copied from unicorn
void MeshRefinement(Mesh& mesh, Function& residuals)
{
  // Define percentage of cells to refine. Could set the number somewhere else
  //  dolfin_set("adapt_percentage", 0.1);
  dolfin_add<real>("adapt_percentage", 0.3);
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

// Set hard-coded normals for vertical and horizontal boundaries
void SetNormal(Mesh mesh, NodeNormal& node_normal, bool horizontal)
{
  Cell cell_tmp(mesh, 0);
  size_t nsd = mesh.topology().dim();
  size_t local_dim = cell_tmp.num_entities(0);
  real *original_block = new real[nsd * local_dim * mesh.num_cells()];
  real *block = new real[nsd * local_dim * mesh.num_cells()];
  node_normal.basis()[0].get_block(original_block);
  size_t block_index = 0;
  real direction = 0.0;
  for(CellIterator c(mesh); !c.end(); ++c)
  {
    for(uint i = 0; i < nsd; i++)
    {
      if     (i == 0) direction = 0;           // x: set to 0
      else if(i == 1) direction = horizontal;  // y: set to 1 if horizontal surface, 0 if vertical
      else            direction = !horizontal; // z: set to 0 if horizontal surface, 1 if vertical
      for(VertexIterator v(*c); !v.end(); ++v)
      {
        block[block_index] = direction*sign(original_block[block_index]);
        block_index++;
      }
    }
  }
  node_normal.basis()[0].set_block(block);

  delete[] original_block;
  delete[] block;
}

// Make all normals on the cylinder surface point toward the center of the cylinder
void SetCylinderNormal(Mesh& mesh, NodeNormal& node_normal)
{
  Cell cell_tmp(mesh, 0);
  size_t nsd = mesh.topology().dim();
  size_t local_dim = cell_tmp.num_entities(0);
  real *block_normal = new real[nsd * local_dim * mesh.num_cells()];
  real *block_tangent_xy = new real[nsd * local_dim * mesh.num_cells()];
  real *block_tangent_yz = new real[nsd * local_dim * mesh.num_cells()];
  size_t block_index = 0;
  real dx = 0.0;
  real dy = 0.0;
  real magnitude = 0.0;
  for(CellIterator c(mesh); !c.end(); ++c)
  {
    // First set all normals of this cell's vertices to 0
    for(uint i = 0; i < nsd*local_dim; i++)
    {
      block_normal[block_index + i] = 0.0;
      block_tangent_xy[block_index + i] = 0.0;
      block_tangent_yz[block_index + i] = 0.0;
    }

    //    for(FacetIterator f(*c); !f.end(); ++f)
    //    {
    //      if(((f->midpoint()[0] - xcenter)*(f->midpoint()[0] - xcenter) + (f->midpoint()[1] - ycenter)*(f->midpoint()[1] - ycenter)) - bmarg < (radius * radius))
    //      {

    // Then loop through all the vertices of this cell
    for(VertexIterator v(*c); !v.end(); ++v)
    {
      // If the vertex is on the cylinder surface, then compute its normal
      if(((v->x()[0] - xcenter)*(v->x()[0] - xcenter) + (v->x()[1] - ycenter)*(v->x()[1] - ycenter)) - bmarg < (radius * radius))
      {
        // Compute the vector components toward the center
        dx = xcenter - v->x()[0];
        dy = ycenter - v->x()[1];

        // The vector magnitude, to normalize
        magnitude = std::sqrt(dx * dx + dy * dy);

        // Avoid division by zero
        if (magnitude > 0.0)
        {
          dx /= magnitude;
          dy /= magnitude;
          block_normal[block_index + c->index(*v)] = dx; // normal x component
          block_normal[block_index + local_dim + c->index(*v)] = dy; // normal y component
          block_tangent_xy[block_index + c->index(*v)] = -dy; // xy tangent x component
          block_tangent_xy[block_index + local_dim + c->index(*v)] = dx; // xy tangent y component
          block_tangent_yz[block_index + 2 * local_dim + c->index(*v)] = 1.0; // yz tangent z component. The other two vectors
                                                                              // are in the xy plane, so this is always 1.0.
        }
      }
    }
    //      }
    //    }
    block_index += (nsd*local_dim);
  }
  node_normal.basis()[0].set_block(block_normal);
  node_normal.basis()[1].set_block(block_tangent_xy);
  node_normal.basis()[2].set_block(block_tangent_yz);

  MPI_Barrier(MPI::DOLFIN_COMM);
  node_normal.basis()[0].sync();
  node_normal.basis()[1].sync();
  node_normal.basis()[2].sync();

  delete[] block_normal;
  delete[] block_tangent_xy;
  delete[] block_tangent_yz;
}

// From ChatGPT
#include <cmath>
#include <limits>
// Function to compute the largest eigenvalue using the Power Iteration Method
double power_iteration(const PETScMatrix& A, int max_iters = 1000, double tol = 1e-8)
{
  Vector x(A.size(1));
  //    x.randomize(); // Start with a random vector
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    double random_value = static_cast<double>(std::rand()) / RAND_MAX; // Random value in [0, 1]
    x.setitem(i, random_value); // Directly set value at index i
  }
  x.apply(); // Apply the changes to the vector
  x *= 1.0 / x.norm(l2); // Normalize

  Vector Ax(A.size(1));
  double lambda = 0.0, lambda_prev = std::numeric_limits<double>::max();

  for (int i = 0; i < max_iters; ++i)
  {
    A.mult(x, Ax); // Compute Ax
    lambda = x.inner(Ax); // Rayleigh quotient
    Ax *= 1.0 / Ax.norm(l2); // Normalize Ax
    x = Ax; // Update x

    // Check convergence
    if (std::abs(lambda - lambda_prev) < tol)
      break;

    lambda_prev = lambda;
  }

  return lambda;
}
// Function to compute the smallest eigenvalue using the Inverse Power Iteration Method
double inverse_power_iteration(const PETScMatrix& A, int max_iters = 1000, double tol = 1e-8)
{
  // Create solver for (A - shift * I)x = b (shift is set to 0 for simplicity)
  LUSolver solver;
  Vector x(A.size(1));
  //    x.randomize(); // Start with a random vector
  for (std::size_t i = 0; i < x.size(); ++i)
  {
    double random_value = static_cast<double>(std::rand()) / RAND_MAX; // Random value in [0, 1]
    x.setitem(i, random_value); // Directly set value at index i
  }
  x.apply(); // Apply the changes to the vector
  x *= 1.0 / x.norm(l2); // Normalize

  Vector Ax(A.size(1));
  double lambda = 0.0, lambda_prev = std::numeric_limits<double>::max();

  for (int i = 0; i < max_iters; ++i)
  {
    solver.solve(A, Ax, x); // Solve A*x = b (in-place)
    lambda = x.inner(Ax); // Rayleigh quotient
    Ax *= 1.0 / Ax.norm(l2); // Normalize Ax
    x = Ax; // Update x

    // Check convergence
    if (std::abs(lambda - lambda_prev) < tol)
      break;

    lambda_prev = lambda;
  }

  return 1.0 / lambda; // Return the smallest eigenvalue
}
// Function to compute the condition number of the matrix
void compute_condition_number(const PETScMatrix& A)
{
  int max_iters = 100;
  double lambda_max = power_iteration(A, max_iters);
  double lambda_min = inverse_power_iteration(A, max_iters);

  double condition_number = lambda_max / lambda_min;

  std::cout << "Largest eigenvalue (lambda_max): " << lambda_max << std::endl;
  std::cout << "Smallest eigenvalue (lambda_min): " << lambda_min << std::endl;
  std::cout << "Condition number: " << condition_number << std::endl;
}
/*#include <Eigen/Dense>
  void compute_condition_number(Matrix& A)
  {
// Convert DOLFIN matrix to Eigen matrix
Eigen::MatrixXd eigen_matrix = Eigen::MatrixXd::Zero(A.size(0), A.size(1));
for (std::size_t i = 0; i < A.size(0); ++i)
{
for (std::size_t j = 0; j < A.size(1); ++j)
{
eigen_matrix(i, j) = A(i, j);
}
}

// Compute SVD
Eigen::JacobiSVD<Eigen::MatrixXd> svd(eigen_matrix);
double sigma_max = svd.singularValues().maxCoeff();
double sigma_min = svd.singularValues().minCoeff();

// Condition number
double condition_number = sigma_max / sigma_min;

message("Condition number: %g", condition_number);
}*/
/*void compute_condition_number(Matrix& A)
  {
// Create eigenvalue solver
SLEPcEigenSolver solver(A);
solver.parameters["problem_type"] = "gen_hermitian"; // For symmetric matrices
solver.solve();

// Extract eigenvalues
double lambda_max = solver.get_eigenvalue(0).real(); // Largest eigenvalue
double lambda_min = solver.get_eigenvalue(solver.get_number_converged() - 1).real(); // Smallest eigenvalue

// Compute condition number
double condition_number = lambda_max / lambda_min;

message("Condition number: %g", condition_number);
}*/

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
  Vector gradU;
  LGrad.assemble(gradU, false);
  gradU.apply();

  uint d = mesh.topology().dim();
  Cell c(mesh, 0);
  uint local_dim = c.num_entities(0);
  size_t *idx  = new size_t[d * local_dim];
  real *gradU_block = new real[d * local_dim];
  size_t global_index;
  real shear, strain, rotation;
  // Triple decomposition is computed on cell level
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  { 
    UFCCell ufc_cell(*cell);
    (aGrad.dofmaps())[0]->tabulate_dofs(idx, ufc_cell, *cell);
    gradU.get(gradU_block, d*d, idx);
    if( MPI::global_size() > 1)
    {
      global_index = mesh.distdata()[d].get_global(cell->index());
    }
    else
    {
      global_index = cell->index();
    }
    // Compute the triple decomposition
    tripleDecomposition(gradU_block, &shear, &strain, &rotation);
    triple_shear.vector().set(&shear, 1, &global_index);
    triple_strain.vector().set(&strain, 1, &global_index);
    triple_rotation.vector().set(&rotation, 1, &global_index);
  }

  // Apply and sync ghosts
  triple_shear.vector().apply();
  triple_shear.sync();
  triple_strain.vector().apply();
  triple_strain.sync();
  triple_rotation.vector().apply();
  triple_rotation.sync();

  delete[] idx;
  delete[] gradU_block;
}

void project_DG0_to_CG1(Mesh mesh, Function DG0_function, Function CG1_function)
{
  KrylovSolver solver;

  Project_DG0_to_CG1::BilinearForm a(mesh);
  Project_DG0_to_CG1::LinearForm L(mesh, DG0_function);

  PETScMatrix A;
  Vector b;

  a.assemble(A, true);
  L.assemble(b, true);
  
  solver.solve(A, CG1_function.vector(), b);
}

// Initialize everything that depends on the mesh. Call this at the start, and after adaptive mesh refinement
/*void initialize(Mesh mesh)
{
  message("(Re-)initializing");

  set_time_step(mesh);

  set_boundary_conditions(mesh);
}*/

int main(int argc, char* argv[])
{
  dolfin_init(argc, argv);
  Mesh mesh("cylinder_3d_bmk.bin"); // Original coarse cylinder mesh
  
  // Print the mesh to new file, needed for dolfin-post
  File meshfile("meshfile.bin");
  meshfile << mesh;

  // Set time step (proportional to the minimum cell diameter) 
  // Get minimum cell diameter
  real hmin = 1.0e6;
  real hminlocal = 1.0e6;
  for (CellIterator cell(mesh); !cell.end(); ++cell)
  {
    if ((*cell).diameter() < hminlocal) hminlocal = (*cell).diameter();
  }
  MPI_Barrier(dolfin::MPI::DOLFIN_COMM);
  MPI_Allreduce(&hminlocal, &hmin, 1, MPI_DOUBLE, MPI_MIN, dolfin::MPI::DOLFIN_COMM);
  real tstep = 0.15*hmin/ubar; // 0.15 taken from unicorn icns 3D cylinder
  message("hmin = %g", hmin);
  message("time step size: %g", tstep);

  // Set boundary conditions
  Analytic<BC_Momentum> momentum_inflow(mesh);
  Analytic<BC_Poiseuille> poiseuille_inflow(mesh);
  Analytic<BC_Continuity> continuity_outflow(mesh);
  Analytic<BC_DirichletSlip> dirichlet_slip(mesh);
  Analytic<InitialVelocity> zero_velocity(mesh);
  InflowBoundary inflow_boundary;
  WholeDomain whole_domain;
  SlipBoundary slip_boundary; // TODO: remove
  HorizontalSlipBoundary horizontal_slip_boundary;
  VerticalSlipBoundary vertical_slip_boundary;
  CylinderSlipBoundary cylinder_slip_boundary;

  OutflowBoundary outflow_boundary;

  // NodeNormal doesn't handle corners well, so we create three instances of it, and manually manipulate the horizontal and vertical normals
  NodeNormal horizontal_node_normal(mesh, NodeNormal::facet, DOLFIN_PI/3); // pi/1.9 in unicorn/icns-newton. But it doesn't affect normals, only tangents
  NodeNormal vertical_node_normal(mesh, NodeNormal::facet, DOLFIN_PI/3);
  NodeNormal cylinder_node_normal(mesh, NodeNormal::facet, DOLFIN_PI/3);
  NodeNormal node_normal(mesh, NodeNormal::facet, DOLFIN_PI/3); // 1.2*pi doesn't work if applied everywhere
  NodeNormal sub_node_normal(mesh, slip_boundary, NodeNormal::facet, DOLFIN_PI/3);

  // Murtazo Slip
  BC_Murtazo_Slip murtazo_slip(mesh, sub_node_normal);
  BC_Murtazo_Slip murtazo_slip_horizontal(mesh, horizontal_node_normal);
  BC_Murtazo_Slip murtazo_slip_vertical(mesh, vertical_node_normal);
  BC_Murtazo_Slip murtazo_slip_cylinder(mesh, cylinder_node_normal);

  SubSystem subsystem_y(1);
  SubSystem subsystem_z(2);

  DirichletBC inflow_mom_bc(momentum_inflow, mesh, inflow_boundary);
  DirichletBC inflow_mom_bc_poiseuille(poiseuille_inflow, mesh, inflow_boundary);
  DirichletBC outflow_mom_bc_poiseuille(poiseuille_inflow, mesh, outflow_boundary); // Poiseuille analytic solution on the outflow
  DirichletBC poiseuille_initial(poiseuille_inflow, mesh, whole_domain);
  //  SlipBC horizontal_slip_bc(mesh, horizontal_slip_boundary, horizontal_node_normal); // This slip should be used
  //  SlipBC vertical_slip_bc(mesh, vertical_slip_boundary, vertical_node_normal); // This slip should be used
  SlipBC cylinder_slip_bc(mesh, cylinder_slip_boundary, cylinder_node_normal); // This slip should be used
  SlipBC slip_bc(mesh, slip_boundary, node_normal); // Testing without defining NodeNormals
  SlipBC sub_slip_bc(mesh, slip_boundary, sub_node_normal); // Testing NodeNormal with SubDomain
                                                            //TODO: Testing subsystems
  DirichletBC horizontal_slip_bc(continuity_outflow, mesh, horizontal_slip_boundary, subsystem_y);
  DirichletBC vertical_slip_bc(continuity_outflow, mesh, vertical_slip_boundary, subsystem_z);
  //  DirichletBC slip_bc(dirichlet_slip, mesh, slip_boundary); // TODO: testing dirichlet slip
  DirichletBC no_slip_bc(zero_velocity, mesh, slip_boundary); // no-slip everywhere
  DirichletBC no_slip_horizontal(zero_velocity, mesh, horizontal_slip_boundary); // no-slip everywhere
  DirichletBC no_slip_vertical(zero_velocity, mesh, vertical_slip_boundary); // no-slip everywhere
  DirichletBC cylinder_no_slip(zero_velocity, mesh, cylinder_slip_boundary);
  DirichletBC outflow_con_bc(continuity_outflow, mesh, outflow_boundary);
  DirichletBC inflow_con_bc(continuity_outflow, mesh, inflow_boundary); //TODO: p=0 on inflow, test this next
  DirichletBC outflow_mom_bc(poiseuille_inflow, mesh, outflow_boundary); // TODO: testing u=poiseuille on the outflow boundary
                                                                         //  PeriodicBC1 periodic_mom_bc(mesh, periodic_boundary);
  DirichletBC murtazo_slip_bc(murtazo_slip, mesh, slip_boundary); // Murtazo slip everywhere at once
  DirichletBC murtazo_slip_bc_horizontal(murtazo_slip_horizontal, mesh, horizontal_slip_boundary); // Murtazo slip on horizontal boundary
  DirichletBC murtazo_slip_bc_vertical(murtazo_slip_vertical, mesh, vertical_slip_boundary); // Murtazo slip on vertical boundary
  DirichletBC murtazo_slip_bc_cylinder(murtazo_slip_cylinder, mesh, cylinder_slip_boundary); // Murtazo slip on cylinder boundary

  std::vector<BoundaryCondition*> bc_con;
  bc_con.push_back(&outflow_con_bc);
  //  bc_con.push_back(&inflow_con_bc);
  std::vector<BoundaryCondition*> bc_mom;
  //  bc_mom.push_back(&cylinder_no_slip);
  //  bc_mom.push_back(&inflow_mom_bc_poiseuille); // For Poiseuille flow test
  //  bc_mom.push_back(&cylinder_slip_bc);
  bc_mom.push_back(&horizontal_slip_bc);
  bc_mom.push_back(&vertical_slip_bc);
//  bc_mom.push_back(&no_slip_horizontal);
//  bc_mom.push_back(&no_slip_vertical);
  //  bc_mom.push_back(&cylinder_no_slip);
  //  bc_mom.push_back(&cylinder_slip_bc);
  //  bc_mom.push_back(&slip_bc);
  //  bc_mom.push_back(&sub_slip_bc);
  //  bc_mom.push_back(&periodic_mom_bc);
  bc_mom.push_back(&inflow_mom_bc); //TODO: This should be used
                                    //  bc_mom.push_back(&sub_slip_bc); //TODO: This has potential, try without on_boundary. Tried, it fails because looping through cells, and the cell is considered internal.
                                    //  bc_mom.push_back(&slip_bc);
                                    //  bc_mom.push_back(&no_slip_bc);
                                    //  bc_mom.push_back(&outflow_mom_bc); // TODO: remove this, it's only for testing

                                    // Separate Murtazo BC vector
  std::vector<BoundaryCondition*> bc_murtazo;
  //  bc_murtazo.push_back(&murtazo_slip_bc);
  //  bc_murtazo.push_back(&murtazo_slip_bc_horizontal);
  //  bc_murtazo.push_back(&murtazo_slip_bc_vertical);
//  bc_murtazo.push_back(&murtazo_slip_bc_cylinder);

  // Set up functions
  Constant dt(tstep);
  Constant nu(viscosity);
  Constant beta(0.0);
  //  Analytic<InitialPressure> d1(mesh); // Stabilization parameter
  //  Analytic<InitialPressure> d2(mesh); // Stabilization parameter
  Function d1(mesh);
  Function d2(mesh);
  //  Analytic<InitialVelocity> u_initial(mesh); // Velocity in previous time step
  //  Analytic<InitialVelocity> up(mesh); // Linearized velocity. Not used in the forms, but needed for computing residuals
  //  Analytic<InitialVelocity> um(mesh); // Cell mean velocity
  //  Function um(mesh);
  Function up(mesh);
  //  Analytic<InitialPressure> p0(mesh); // Is this needed or used? How else to set initial pressure?

  // Create forms
  NavierStokes3D_force::BilinearForm a_mom(mesh, up, nu, d1, d2, dt, beta, cylinder_node_normal.basis()[0]);
  NavierStokesContinuity3D::BilinearForm a_con(mesh, d1);
  Function u(a_mom.trial_space());
  //  InitialVelocityFunction u_initial(a_mom.trial_space());
  //  u = u_initial;
  Function u0(a_mom.trial_space());
  //  u0 = 1;
  //  Function up(a_mom.trial_space()); // Needed for bilinear form
  //  Function um(a_mom.trial_space());
  Function p(a_con.trial_space());
  Function p0(a_con.trial_space());
  //  NavierStokes3Dmin::LinearForm L_mom(mesh, up, u0, p, nu, dt);// d1, d2, dt);
  //  NavierStokes3Dmin::LinearForm L_mom(mesh, up, u0);
  //  NavierStokes3Dmin::LinearForm L_mom(mesh);
  NavierStokes3D_force::LinearForm L_mom(mesh, up, u0, p, nu, d1, d2, dt);
  NavierStokesContinuity3D::LinearForm L_con(mesh, u, u0); //, d1);
  PETScMatrix A_mom, A_con;
  Vector b_mom, b_con;
  //  a_mom.assemble(A_mom, true);
  //  a_con.assemble(A_con, true);

  //  d1.init(a_mom.trial_space());
  //  d2.init(a_mom.trial_space());

  // Initialize vectors for the time step residuals of
  // the momentum and continuity equations
  size_t n = mesh.num_vertices();
  if(MPI::global_size() > 1) n -= mesh.distdata()[0].num_ghost();
  Vector residual_mom(mesh.topology().dim()*n);
  Vector residual_con(n);

  // Initialize algebraic solvers   
  KrylovSolver solver_con(gmres, amg);
  KrylovSolver solver_mom(gmres, amg);
  //  LUSolver solver_con;
  //  LUSolver solver_mom;

  // Sync ghosts of everything. Not sure if this is needed, or why
  u.sync();  // velocity
  u0.sync(); // velocity from previous time step 
  up.sync(); // velocity linearized convection. Basically just velocity from the previous iteration
             //  um.sync(); // cell mean velocity
  p.sync();  // pressure
  p0.sync(); // pressure
  d1.sync(); // stabilization coefficient
  d2.sync(); // stabilization coefficient

  //  Assembler assembler(mesh); // Maybe not needed, this could be a dolfin 0.8 thing?

  // Compute stabilization parameters d1 and d2
  ComputeStabilization(mesh, u0, viscosity, tstep, d1, d2, L_mom);

  // Compute normal vectors for the surface nodes, needed for the slip boundary condition
  horizontal_node_normal.init(a_mom.trial_space());
  vertical_node_normal.init(a_mom.trial_space());
  cylinder_node_normal.init(a_mom.trial_space());
  node_normal.init(a_mom.trial_space());
  sub_node_normal.init(a_mom.trial_space());
  horizontal_node_normal.compute();
  vertical_node_normal.compute();
  cylinder_node_normal.compute();
  node_normal.compute();
  sub_node_normal.compute();

  // TODO: Testing manipulating node_normal
  /*  node_normal.basis()[0].decompose()[0]->zero();
      node_normal.basis()[0].decompose()[1]->zero();
      node_normal.basis()[0].decompose()[2]->zero();
   *(node_normal.basis()[0].decompose()[0]) = 0;
   *(node_normal.basis()[0].decompose()[1]) = 0;
   *(node_normal.basis()[0].decompose()[2]) = 0;
   node_normal.basis()[0].decompose()[0]->disp();
   node_normal.basis()[0].decompose()[1]->disp();
   node_normal.basis()[0].decompose()[2]->disp(); */
  //  horizontal_node_normal.basis()[0].vector().disp();
  SetNormal(mesh, horizontal_node_normal, true);
  SetNormal(mesh, vertical_node_normal, false);
  SetCylinderNormal(mesh, cylinder_node_normal);

  // Set Murtazo's slip BC to use u and up
  murtazo_slip.set_velocities(&u); //&up
  murtazo_slip_horizontal.set_velocities(&up);
  murtazo_slip_vertical.set_velocities(&up);
  murtazo_slip_cylinder.set_velocities(&up);

  // Time stepping and iteration parameters
  double t = 0.0;
  uint step = 0;
  real residual, residual2, last_residual;
  real residual_c, residual_m;
  real rtol = 1.0e-2;//1.0e-2;
  real rtol2 = 1.0e-3;
  int iteration;
  int max_iteration = 10; //50;

  // Initialize residual function, to be used for adaptive mesh refinement
  Function residual_function(mesh); // Should I tie this to the trial function of Residual, or just assemble LRes into residual_function.vector()?
                                    //  Function um(mesh);
                                    //  Function u0m(mesh);
                                    //  Function pm(mesh);
  Function residual_cell(mesh); // Residual on cell level

  //  Residual::BilinearForm aRes(mesh); // Unused I think, so I can probably remove this
  Residual::LinearForm LRes(mesh, u, u0, p, dt);

  residual_function.init(LRes.create_coefficient_space("U")); // Get it as linear 3D vector first, then average on cell level
                                                              //  um.init(LRes.create_coefficient_space("Um"));
                                                              //  u0m.init(LRes.create_coefficient_space("Um"));
                                                              //  pm.init(LRes.create_coefficient_space("P"));
  residual_cell.init(LRes.create_coefficient_space("k"));

  // Initialize piecewise constant triple decomposition functions
  Function triple_shear(mesh);
  Function triple_strain(mesh);
  Function triple_rotation(mesh);
  // Initialize piecewise linear triple decomposition functions, needed to
  // print to binary file format
  Function shear_linear(mesh);
  Function strain_linear(mesh);
  Function rotation_linear(mesh);

  Function vol_inv(mesh);

  // Assembling L=inner(grad(u),v) outputs grad(u) directly, no need to solve anything. LGrad also only needs to be created once
  Gradient::BilinearForm aGrad(mesh);
  Gradient::LinearForm LGrad(mesh, u, vol_inv);

  // Initialize functions with the appropriate FE space
  vol_inv.init(LGrad.create_coefficient_space("icv"));
  triple_shear.init(LGrad.create_coefficient_space("icv"));
  triple_strain.init(LGrad.create_coefficient_space("icv"));
  triple_rotation.init(LGrad.create_coefficient_space("icv"));
  shear_linear.init(LGrad.create_coefficient_space("u"));
  strain_linear.init(LGrad.create_coefficient_space("u"));
  rotation_linear.init(LGrad.create_coefficient_space("u"));

  // Compute inverse of mesh cell volumes, needed to compute the triple decomposition
  ComputeVolInv(mesh, vol_inv);

  // Output files
  File solutionfile("solution.bin");
  //  std::vector<std::pair<Function*, std::string> > output;
  LabelList<Function> output;
  //  std::pair<Function*, std::string> u_output(&u, "Velocity");
  //  std::pair<Function*, std::string> p_output(&p, "Pressure");
  //  std::pair<Function*, std::string> n_output(&sub_node_normal.basis()[0], "Normals");
  Label<Function> u_output(u, "Velocity");
  Label<Function> p_output(p, "Pressure");
  Label<Function> n_output(sub_node_normal.basis()[0], "Normals");
  Label<Function> sh_output(shear_linear, "Shear");
  Label<Function> el_output(strain_linear, "Strain");
  Label<Function> rr_output(rotation_linear, "Rotation");
//  Label<Function> res_output(residual_cell, "Residual");
  output.push_back(u_output);
  output.push_back(p_output);
  output.push_back(n_output);
  output.push_back(sh_output);
  output.push_back(el_output);
  output.push_back(rr_output);
//  output.push_back(res_output);

  // Debugging files
  File residual_file("residual.pvd");

  // write the initial condition to the solution file
#ifdef IO
  //  u_file << u0; //u0;
  //  p_file << sub_node_normal.basis()[0]; //horizontal_node_normal.basis()[0]; //p0;
  solutionfile << output;
#endif

  // HeartSolver/Dolfin 0.8 assembling
  //  Assembler assembler_con;
  //  Assembler assembler_mom;

  int runs = 5;

  // Time stepping!
  for(int i = 0; i < runs; i++)
  {
    while(t < Tfinal)
    {
      dt = (t+tstep > Tfinal ? Tfinal-t : dt);

      // Ramp up inflow condition
      ubar = std::min(ubar_max, ubar_max*(step+1)/10);

      // NSESolver sets u0 = u here, but it makes more sense to do it at the end of the time step

      // Initialize residuals
      residual = 2*rtol;
      residual2 = 2*rtol2;
      last_residual = 3*rtol;
      iteration = 0;

      // HeartSolver/Dolfin 0.8 assembling
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

        // Try resetting A and b. This was needed because A and b didn't reset, but then I realized the reset
        // boolean should be false to force reset
        //      if(step > 0)
        if(false)
        {
          b_con.zero();
          A_con.zero();
          b_mom.zero();
          A_mom.zero();
        }

        message("norm(u): %g", u.vector().norm(l2));
        message("norm(up): %g", up.vector().norm(l2));
        message("norm(u0): %g", u0.vector().norm(l2));


        ComputeStabilization(mesh, u0, viscosity, tstep, d1, d2, L_mom); //TODO: This should not be commented out

        // Compute cell mean velocity
        //      ComputeMean(mesh, um, up, a_mom);

        // HeartSolver/Dolfin 0.8 assembling
        //      assembler_con.assemble(A_con, a_con, false);
        //      assembler_con.assemble(b_con, L_con, false);
        // Assemble is not a class anymore, trying this instead
        //      Assembler::assemble(A_con, a_con, true);
        //      Assembler::assemble(b_con, L_con, true);
        // The dolfin 0.9 way to assemble I think
        // Trying to add old A_con
        if(false)//(step > 0)
        {
          PETScMatrix A_con_old(A_con.size(0), A_con.size(0), false);
          A_con_old.dup(A_con);
          a_con.assemble(A_con, true);
          A_con += A_con_old;
        }
        else
        {
          a_con.assemble(A_con, step == 0);
        }
        // Trying to add old A_con to new
        L_con.assemble(b_con, step == 0);

        A_con.mult(p.vector(), residual_con);
        residual_con -= b_con;
        //      residual_con += p.vector(); //TODO: remove, just checking if A*u-b=u
        message("A_con*p-b_con=%g before applying boundary conditions", residual_con.norm(l2));      

        std::pair<size_t, size_t> diagonal_index;
        // TODO: adding identity matrix to A_con as well
        for(uint i = 0; i < bc_con.size(); i++)
        {
          //        if(step < 15)
          bc_con[i]->apply(A_con, b_con, a_con);
        }

        // Solve the continuity equation
        solver_con.solve(A_con, p.vector(), b_con);
        p.sync(); // Ashish syncs after solving, NSESolver does not. I think it makes sense

        //      message("norm(A) before assemble: %g", A_mom.norm());

        // HeartSolver/Dolfin 0.8 assembling
        //      assembler_mom.assemble(A_mom, a_mom, false);
        //      assembler_mom.assemble(b_mom, L_mom, false);
        // Assemble is not a class anymore, trying this instead
        //      Assembler::assemble(A_mom, a_mom, true);
        //      Assembler::assemble(b_mom, L_mom, true);
        // Assemble momentum vector
        a_mom.assemble(A_mom, step == 0);
        L_mom.assemble(b_mom, step == 0);
        //      A_mom/=tstep;
        //      A_mom.apply();
        // Setting A_mom += I
        //      std::pair<size_t, size_t> diagonal_index;
        //      for(size_t i = 0; i < A_mom.size(0); i++)
        //      {
        //        diagonal_index = {i, i};
        //        message("%g",A_mom(i,i));
        //        A_mom.setitem(diagonal_index, A_mom(i,i) + 1.0);
        //        A_mom.apply();
        //      }//TODO: adding an identity matrix gives a correct solution, but is it just because A_mom is small?
        message("norm(A) before BC: %g", A_mom.norm("frobenius"));
        A_mom.mult(u.vector(), residual_mom);
        residual_mom -= b_mom;
        message("A*u-b=%g before applying boundary conditions", residual_mom.norm(l2));
        residual_mom += u.vector(); //TODO: remove, just checking if A*u-b=u
        message("norm(b): %g", b_mom.norm(l2));
        message("A*u-b+u=%g before applying boundary conditions", residual_mom.norm(l2));
        //      message("norm(A) before applying boundary conditions: %g", A_mom.norm());
        Vector u_minus_b(mesh.topology().dim()*n);
        u_minus_b.zero();
        u_minus_b += u.vector();
        u_minus_b -= b_mom;
        message("u-b=%g", u_minus_b.norm(l2));
        //      message("before BC:");
        //      compute_condition_number(A_mom);
        //      message("norm(b) before applying boundary conditions: %g", b_mom.norm(l2));
        //      if(step == 0)

        //      if(step <= 20)
        /*      if(false)
                {
                cylinder_no_slip.apply(A_mom, b_mom, a_mom);
                }
                else
                {
                cylinder_slip_bc.apply(A_mom, b_mom, a_mom);
                }*/

        if(false)
        {
          poiseuille_initial.apply(A_mom, b_mom, a_mom);
        }
        else
        {
          for(uint i = 0; i < bc_mom.size(); i++)
          {
            bc_mom[i]->apply(A_mom, b_mom, a_mom);
          }
        }
        //      if(step < 5) outflow_mom_bc_poiseuille.apply(A_mom, b_mom, a_mom); // Analytic solution on the outflow for the first steps

        A_mom.mult(u.vector(), residual_mom);
        residual_mom -= b_mom;
        //      residual_mom += u.vector(); //TODO: remove, just checking if A*u-b=u
        //      Vector u_minus_b(mesh.topology().dim()*n);
        u_minus_b.zero();
        u_minus_b += u.vector();
        u_minus_b -= b_mom;
        message("norm(A) after BC: %g", A_mom.norm("frobenius"));
        //      message("norm(b): %g", b_mom.norm(l2));
        message("A*u-b=%g", residual_mom.norm(l2));
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

        // Trying Murtazo after residual2. Au-b will never converge if doing this before calculating it
        murtazo_slip_bc.apply(A_mom, u.vector(), a_mom); // Does this work? Applying with u instead of b should be enough?
        u.sync();

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

        // Tired of dealing with residuals getting messed up because of the Murtazo addition, but should do better than this
        //      if(abs(last_residual - residual) < rtol)
        //        break;
        //      else
        //        last_residual = residual;
      } // Fix-point iteration for non-linear problem closed

      // Compute residual
      message("residual function vector size: %d", residual_function.vector().size());
      LRes.assemble(residual_function.vector(), false); // false means reassemble, which we always want?
      ComputeMeanResidual(mesh, residual_cell, residual_function);

      // Set p0=p here too? Is it from last iteration or last time step?
      u0 = u;
      //    u0m = um;
      t += tstep;
      step++;
#ifdef IO
      //    if(std::floor(t) > std::floor(t-tstep)) // Only print once per simulated second
      //    if(true) // Print every timestep
      if(step < 100 || std::floor(10*t) > std::floor(10*(t-tstep))) // Print the 100 first timesteps, then ten times per simulated second
      {
        // Compute triple decomposition
        computeTripleDecomposition(mesh, aGrad, LGrad, u, triple_shear, triple_strain, triple_rotation);

        project_DG0_to_CG1(mesh, triple_shear, shear_linear);
        project_DG0_to_CG1(mesh, triple_strain, strain_linear);
        project_DG0_to_CG1(mesh, triple_rotation, rotation_linear);

        solutionfile << output;
      }
#endif
      message("------------------------------------ Step %d finished ------------------------------------", step);
    }

    if(true)
    {
      // Refine mesh, then reset simulation time and solution
      MeshRefinement(mesh, residual_cell);
      t = 0.0;
      step = 0;
      // Do Functions need to be reinitiated?
      u.zero();
      u0.zero();
      p.zero();
    }
  }
  //  unicorn_solve(mesh, chkp, w_limit, s_time, iter, 0, &smooth, &solve);

  dolfin_finalize();
  return 0;
}

