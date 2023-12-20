#include <dolfin.h>

using namespace dolfin;
//-----------------------------------------------------------------------------
class Bottom : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return x[1] <= 0.5 ;
  }
};
class Top : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return x[1] >= 0.5;
  }
};

int main(int argc, char **argv)
{
  dolfin_init(argc, argv);

  UnitSquare mesh(50, 50);
  
  Bottom            bottom;
  Top               top;
  // Create mesh function over the cell facets
  MeshValues<size_t, Cell> sub_domains(mesh);

  // Mark all facets as sub domain 2
  sub_domains = 2;
  top.mark(sub_domains, 0);
  bottom.mark(sub_domains, 1);

  // Save sub domains to file
  File("subdomains.pvd") << sub_domains;
  
  dolfin_finalize();

  return 0;
}
