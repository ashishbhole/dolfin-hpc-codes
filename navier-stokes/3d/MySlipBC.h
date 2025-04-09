// Copyright (C) 2024 Joel Kronborg
// Licensed under the GNU LGPL Version 2.1.

#ifndef __DOLFIN_SLIPBC_H
#define __DOLFIN_SLIPBC_H

#include <dolfin/fem/BoundaryCondition.h>
#include <dolfin/fem/NodeNormal.h>
#include <dolfin/la/Matrix.h>
#include <dolfin/la/Vector.h>

namespace dolfin
{

class DofMap;
class Form;
class Function;
class Mesh;
//class ScratchSpace;
//class SubDomain;

class MySlipBC : public BoundaryCondition
{
public:
  /// Create boundary condition for sub domain given normals
  MySlipBC(Mesh & mesh, SubDomain const & sub_domain, NodeNormal & normals, Coefficient const& up);

  /// Destructor
  ~MySlipBC() override;

  /// Apply boundary condition to linear system
  void apply(GenericMatrix &      A,
             GenericVector &      b,
             BilinearForm const & form ) override;

private:
  NodeNormal* node_normal;
  Matrix* As;

  // array of coefficient-subdomain pairs
  std::vector< std::pair< Coefficient &, SubDomain const & > > conditions;
};

MySlipBC::MySlipBC(Mesh & mesh, SubDomain const & sub_domain, NodeNormal & normals, Coefficient const& up)
  : BoundaryCondition("SlipBC", mesh, sub_domain),
  mesh(mesh),
  node_normal(&normals),
//  node_normal_local( true ),
  conditions( { { up, sub_domain } } ),
  As( nullptr )
//  As_local( true ),
{
  // Do nothing
}

void MySlipBC::apply(GenericMatrix& A, GenericVector& b, BilinearForm const& form, Coefficient const& up)
{
  // u = up - (up*n)n
  //
  // I.e. for every component j:
  //
  // u[j] = up[j] - (up*n)n[j]
  
  // Initial checks copied from DirichletBC
  if ( form.trial_space() != form.test_space() )
  {
    error(
      "MySlipBC is implemented only for identical test and trial space" );
  }

  if ( entities_.empty() || this->invalid_mesh() )
  {
    if ( ( method_ == topological ) || ( method_ == geometric ) )
    {
      entities_.clear();

      // Build set of boundary facets
      size_t const tdim = mesh().topology_dimension();
      for ( FacetIterator f( mesh() ); !f.end(); ++f )
      {
        bool const on_boundary =
          ( f->num_entities( tdim ) == 1 ) && !f->is_shared();

        // loop over all conditions in reverse order
        for ( size_t c = conditions.size(); c > 0; --c )
        {
          if ( conditions[c - 1].second.enclosed( *f, on_boundary ) )
          {
            // Get cell to which facet belongs (there may be two, but pick
            // first)
            Cell cell( mesh(), f->entities( tdim )[0] );
            entities_.push_back( cell.index() );
            entities_.push_back( cell.index( *f ) );
            entities_.push_back( c - 1 );
            break;
          }
        }
      }
    }

    this->update_mesh_dependency();
  }

  // Simple check
  form.check(A, b);

  // Check compatibility of function g and the test (sub)space
  FiniteElementSpace const & space = form.trial_space();
  ufc::finite_element* fe = space.element().create_sub_element( this->sub_system() );

  for ( size_t c = 0; c < conditions.size(); ++c )
  {
    Coefficient & g = conditions[c].first;

    if ( ( fe->value_rank() != g.rank() )
         || ( fe->value_dimension( 0 ) != g.dim( 0 ) ) )
    {
      error(
        "Rank and/or value dimension mismatch between function and space.\n"
        "Function : rank = %d, dim = %d; Space : rank = %d, dim = %d.",
        g.rank(),
        g.dim( 0 ),
        fe->value_rank(),
        fe->value_dimension( 0 ) );
    }
  }
  // Initial checks finished

  // TODO: Here we need to do some things
  _map< size_t, real > boundary_values;
  computeBCTopological( boundary_values, space, this->sub_system() );

  // Setting values in A and b, copied from DirichletBC
  message( "Applying boundary conditions to linear system" );

  // Modify RHS vector (b[i] = value)
  b.set( values.data(), boundary_values.size(), dofs.data() );

  // Modify linear system (A_ii = 1)
  if ( not dolfin_get< bool >( "Krylov keep PC" ) )
  {
    A.ident( boundary_values.size(), dofs.data() );
    A.apply();
  }

  // Finalise changes to b
  b.apply();
}

// Copied from DirichletBC. Hope it works
void MySlipBC::computeBCTopological( _map< size_t, real > & boundary_values,
                                     FiniteElementSpace const & space,
                                     SubSystem const &          sub_system )
{
  // Special case
  if ( entities_.empty() )
  {
    if ( !space.mesh().is_distributed() )
    {
      warning( "Found no facets matching domain for boundary condition." );
    }
    return;
  }

  // Iterate over facets
  DofMap const &        dof_map = space.dofmap();
  std::vector< size_t > cell_dofs( dof_map.num_element_dofs );
  ScratchSpace          scratch( space, sub_system );
  for ( std::vector< size_t >::const_iterator it = entities_.begin();
        it != entities_.end(); )
  {
    // Get cell number and local facet number
    size_t const c_index = *( it++ );
    size_t const f_index = *( it++ );
    size_t const c       = *( it++ );

    // Create cell
    Cell cell( mesh(), c_index );
    scratch.cell.update( cell );

    // Tabulate dofs on cell for the full space dofmap
    dof_map.tabulate_dofs( cell_dofs.data(), scratch.cell, cell );

    // Interpolate function on cell
    conditions[c].first.interpolate( scratch.coefficients.data(), scratch.cell,
                                     *scratch.finite_element, f_index );

    // Tabulate which dofs of the subdofmap are on the facet
    scratch.dof_map->tabulate_facet_dofs( scratch.facet_dofs.data(), f_index );

    // Pick values for facet
    for ( size_t i = 0; i < scratch.dof_map->num_facet_dofs(); ++i )
    {
      size_t const dof     = cell_dofs[scratch.offset + scratch.facet_dofs[i]];
      real const   value   = scratch.coefficients[scratch.facet_dofs[i]];
      boundary_values[dof] = value;
    }
  }
}















