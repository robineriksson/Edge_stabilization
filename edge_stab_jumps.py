# Edge stabilization for Galerkin approximations
# of convection-diffusion-reation problems.
#
# Implementing a stabilization method by
# adding a term penalizing the gradient jumps
# accross element boundaries
#
# Method presented by E. Burman & P. Hansbo
# in Computer methods in applied mechanics and engineering.

from dolfin import *
import numpy as np
import sys

set_log_level(40) #only show errors and ignore warnings.

def compute(nx,ny,degree):
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx,ny)#,"crossed")


    V = FunctionSpace(mesh, 'Lagrange', degree=degree)
    # Define Dirichlet boundary (x = 0 or x = 1)
    edge_eps = DOLFIN_EPS
    other_edge_eps = DOLFIN_EPS
    def boundary(x):
        #return on_boundary
        #return x[1] < DOLFIN_EPS or x[1] > 1.0 - DOLFIN_EPS or x[0] < DOLFIN_EPS
        #return x[0] > 1- DOLFIN_EPS or x[0] <  DOLFIN_EPS
        return x[0] <  edge_eps or x[0] > (1 - edge_eps)  or x[1] < edge_eps or x[1] > (1.0 - edge_eps)

    def boundary1(x):
        return x[0] < DOLFIN_EPS or x[1] > (1 - DOLFIN_EPS) or ( x[0] > (1 - DOLFIN_EPS) and x[1] > 0.7 - DOLFIN_EPS)

    def boundary0(x):
        return x[1] < DOLFIN_EPS or (x[0] > (1 - DOLFIN_EPS) and x[1] < 0.7 - DOLFIN_EPS)
    # Defined constants
    sig = 1
    #sig = 0
    #eps = 1e-5
    #eps = 1e-2

    eps = 0
    gamma = 0.025
    a_w_hyp = 0.05
    a_w_gau = 0.2
    #beta = Constant((-0.8216,-0.57))
    #beta = Constant((-0.6,-0.8))

    beta = Constant((1, 0))

    # Exact solution
    #u_e = Expression("exp(-1*pow((x[0]-0.5),2)/a_w - 3*pow((x[1]-0.5),2)/a_w)", a_w = a_w_gau)
    u_e = Expression("0.5*(1-tanh((x[0]-0.5)/a_w))",a_w=a_w_hyp)
    # Interpolate for use in f.
    W = FunctionSpace(mesh, 'Lagrange', degree = 5)
    u_e_inter = interpolate(u_e,W)

    # Define boundary condition
    bc = DirichletBC(V, u_e_inter, boundary)
    #bc = DirichletBC(V, Constant(0.0), boundary)

    #bc0 = DirichletBC(V,Constant(0.0), boundary0)
    #bc1 = DirichletBC(V,Constant(1.0), boundary1)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Build f
    #f = Expression("sigma*exp(-pow((x[0]-0.5),2)/a - 3*pow((x[1] - 0.5),2)/a) + beta[0]*(-2*(x[0]-0.5)*exp(-pow((x[0]-0.5),2)) + 3*pow((x[1]-0.5),2)/a)/a +  (ep*exp( -pow(x[0],2) + x[0] - 3*pow(x[1],2) + 3*x[1]-1)/a * (-8*a + 4*pow(x[0],2) - 4*pow(x[0],2) + 36*x[1] + 10))/pow(a,2)", a=a_w, sigma=sig, ep=eps, degree = 2)
    f = sig*u_e_inter + dot(beta,grad(u_e_inter)) - div(eps * grad(u_e_inter))
    #f = Expression("(1-2*(x[0]-0.5)/a_w)*exp(-pow((x[0]-0.5),2)/a_w - 3*pow((x[1]-0.5),2)/a_w)", a_w = a_w_gau)
    #f = Constant(0.0)
    # Build a
    a = (sig * u * v + inner(beta, grad(u) * v) + eps * inner(grad(u), grad(v))) * dx

    # Build J
    # Term penalizing gradient jumps accross element boundaries
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    n = FacetNormal(mesh)
    #np = n('+')
    #t = as_vector([-np[1], np[0]])


    J = 0.5 * gamma * h_avg**2 * dot(jump(grad(u), n), jump(grad(v), n))* dS

    # Shock capturing Term
    #In two dimensions, psi consists of only one term since faces and edges coincide. Note that grad(u)*t is always single-valued since v is H1-conforming.

    # note sure of the above. As it's taken from the other article [3] and the expression looks a bit different. It loops over vertexes and edges in facets. But we have a loop over vertexes with a check for largest jump over any edge inside the vertex.

    #psi = h_avg*(0 + 10*h_avg) * abs(jump(grad(u),np))
    #eps_e = 1
    #phi = tanh(dot(t,grad(u))/eps_e)*t
    #sc = psi*dot(phi,grad(v))*dS



    # RHS
    L = f * v * dx

    # Compute solution
    u = Function(V)
    solve(a + J == L,u,bc)
    #solve(a == L,u,bc)

    #solve(a + J == L,u, [bc0,bc1])
    #solve(a + J + sc == L,u,[bc0,bc1])

    # Plot solution and mesh

    #uvec = u.vector()
    #uMax = uvec.max()
    #print uMax

    #plot(u)
    # Dump solution to file in VTK format
    file = File("40With.pvd")
    file << u
    #interactive()


    #Calculate Errors
    #E4 = errornorm(u_e, u, normtype='l2', degree=3)
    # Manual implementation (Bug in fenics --> manual implementation)
    def errornorm(u_e, u, Ve):
        u_Ve = interpolate(u, Ve)
        u_e_Ve = interpolate(u_e, Ve)
        e_Ve = Function(Ve)
        # Subtract degrees of freedom for the error field
        e_Ve.vector()[:] = u_e_Ve.vector().array() - u_Ve.vector().array()
            # More efficient computation (avoids the rhs array result above)
        #e_Ve.assign(u_e_Ve)                      # e_Ve = u_e_Ve
        #e_Ve.vector().axpy(-1.0, u_Ve.vector())  # e_Ve += -1.0*u_Ve
        error = e_Ve**2*dx
        return sqrt(assemble(error, mesh=Ve.mesh())), e_Ve
    Ve = FunctionSpace(mesh, 'Lagrange', degree = 5)
    E1, e_Ve = errornorm(u_e, u, Ve)

    # H1 seminorm
    error = inner(grad(e_Ve), grad(e_Ve))*dx
    E2 = sqrt(assemble(error))

    # Infinity norm based on nodal values
    u_e_V = interpolate(u_e, V)
    E3 = abs(u_e_V.vector().array() - u.vector().array()).max()

    # Collect error measures in a dictionary with self-explanatory keys
    errors = {  'L2': E1,
                'H1': E2,
                'Linf': E3}
    return errors

# Perform experiments
degree = int(sys.argv[1])
h = []  # element sizes
E = []  # errors
#N = [20,40,80,160]
N = [40]
for nx in N:
    h.append(1.0/nx)
    E.append(compute(nx, nx, degree))  # list of dicts

# Convergence rates
from math import log as ln  # log is a dolfin name too
error_types = E[0].keys()

    #for error_type in sorted(error_types):
    #print '\nError norm based on', error_type
"""
from prettytable import PrettyTable
tt = PrettyTable()
tt.add_column('N',N)
for eType in sorted(error_types):
        lE = []
        for i in range(0,len(E)):
            lE.append(E[i][eType])
        tt.add_column(eType,lE)
print tt
"""
print 'N\t**h**',
for eType in error_types:
    print '**\t**',eType,
print ""
for i in range(0, len(E)):
    print '%2d'%N[i],
    print '\t%8.2E'% h[i],
    for error_type in error_types:
        #Ei   = E[i][error_type]  # E is a list of dicts
        print '\t%8.1E'%E[i][error_type],
        #Eim1 = E[i-1][error_type]
        #r = ln(Ei/Eim1)/ln(h[i]/h[i-1])
    print ""

        #print 'N=%2d\tE=%8.2E' % (N[i], E[i])

"""
f = gaus(...)
robin@debian:~/gits/edge_stabilization_fenics$ python edge_stab_jumps.py 1
N	**h** **	** Linf **	** H1 **	** L2
20 	5.00E-02 	5.89E-03 	1.36E-01 	1.00E-03
40 	2.50E-02 	1.46E-03 	6.78E-02 	2.44E-04
80 	1.25E-02 	3.31E-04 	3.39E-02 	7.11E-05
160 	6.25E-03 	1.16E-04 	1.70E-02 	4.11E-05

f = tanh(...)
robin@debian:~/gits/edge_stabilization_fenics$ python edge_stab_jumps.py 1
N	**h** **	** Linf **	** H1 **	** L2
20 	5.00E-02 	5.05E-02 	6.75E-01 	6.21E-03
40 	2.50E-02 	9.49E-03 	2.81E-01 	1.06E-03
80 	1.25E-02 	2.33E-03 	1.38E-01 	2.58E-04
160 	6.25E-03 	6.58E-04 	6.85E-02 	7.22E-05
"""
