NSESolver::NSESolver(Mesh& mesh,
                     NodeNormal& node_normal,
                     Array<BoundaryCondition*>& bc_mom,
                     BoundaryCondition& bc_con,
                     real nu)
  : mesh(mesh),
    node_normal(node_normal),
    bc_mom(bc_mom),
    bc_con(bc_con),
    nu(nu)
{
}
//-----------------------------------------------------------------------------
void NSESolver::solve()
{
  if(dolfin::MPI::processNumber() == 0)
    dolfin_set("output destination","terminal");

  // Set time step (proportional to the minimum cell diameter) 
  GetMinimumCellSize(mesh, hmin);

  real k = 0.15*hmin/ubar;

  Function u; // velocity
  Function up; // primal velocity
  Function u0; // velocity from previous time step 
  Function uc; // velocity linearized convection 
  Function dtu; // Time derivative of velocity
  Function dtup; // Time derivative of primal velocity
  Function p;   // pressure
  Function p0;   // pressure from previous iteration
  Function pp;   // primal pressure
  Function vol_inv;  // 1 / volume of element
  Function res_m;  // momentum residual
  Function delta1, delta2; // stabilization coefficients


