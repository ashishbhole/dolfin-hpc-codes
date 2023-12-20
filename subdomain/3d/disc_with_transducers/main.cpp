#include <dolfin.h>

using namespace dolfin;
//-----------------------------------------------------------------------------
class PZ1 : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return x[2] >= 0.0;
  }
};

class AL : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return x[2] <= 0.0 && x[2] >= -0.001035;
  }
};

class PZ2 : public SubDomain
{
  bool inside( const real * x, bool on_boundary ) const
  {
    return x[2] <= -0.001035;
  }
};

int main(int argc, char **argv)
{
  dolfin_init(argc, argv);

  Mesh mesh("disc_with_transducers.bin");
  
  AL  aluminum_disc;
  PZ1 transducer_top;
  PZ2 transducer_bottom;

  // Create mesh function over the cell facets
  MeshValues<size_t, Cell> sub_domains(mesh);

  // Mark all facets as sub domains as 3
  sub_domains = 3;
  transducer_top.mark(sub_domains, 0);
  aluminum_disc.mark(sub_domains, 1);
  transducer_bottom.mark(sub_domains, 2);

  // Save sub domains to file
  File("subdomains.pvd") << sub_domains;
  
  dolfin_finalize();

  return 0;
}
