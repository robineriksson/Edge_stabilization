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
from math import pi
import numpy as np
import sys

set_log_level(40) #only show errors and ignore warnings.

def compute(h_cell):
    # Create mesh and define function space
    mesh = CircleMesh(Point(0,0),1.0,h_cell)


    V = FunctionSpace(mesh, 'Lagrange', degree=2)
    # Define Dirichlet boundary (x = 0 or x = 1)
    def boundary(x, on_boundary):
        #r_0 = 0.25
        #return x[0]**2 + x[1]**2 <  r_0**2 - DOLFIN_EPS
        return on_boundary
    # Defined constants
    sig = 1
    eps = 0;
    #eps = 1e-5
    #eps = 0.1
    #eps = 0.001

    gamma = 0.025

    beta = Expression(('-2*pi*x[1]','2*pi*x[0]'))


    # IC
    class u0Expression(Expression):
        def __init__(self,x_01,x_02,r_0):
            self.x_01 = x_01
            self.x_02 = x_02
            self.r_0 = r_0


        def eval(self,value, x):
            test = (x[0]-x_01)**2 + (x[1]-x_02)**2
            #print "test: %f, r^2: %f" %(test,r_0**2)
            if test <= r_0**2:
                value[0] = 1
            #    print "hello"
            else:
                value[0] = 0
    # end of class





    # Define boundary condition
    bc = DirichletBC(V, Constant(0.0), boundary)

    #dt = 0.1
    dt = 0.5*h_cell/2/pi

    x_01 = 0.3
    x_02 = 0
    r_0 = 0.25

    u0 = u0Expression(x_01,x_02,r_0)
    #u0 = Expression("0.5*(1- tanh( (pow((x[0] - x_01),2) + pow((x[1] - x_02),2))/pow(r_0,2) - 1))",x_01 = x_01, x_02 = x_02, r_0 = r_0 )
    #u0 = conditional(gt( (x[0]-x_01)**2 + (x[1]-x_02)**2,r_0**2 ), Constant(1.0), Constant(0.0))
    u1 = interpolate(u0,V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)

    # Build f
    f = Constant(0.0)
    # Build a
    a = (sig * u * v + dt*(inner(beta, grad(u) * v) + eps * inner(grad(u), grad(v)))) * dx

    # Build J
    # Term penalizing gradient jumps accross element boundaries
    h = CellSize(mesh)
    h_avg = (h('+') + h('-'))/2
    n = FacetNormal(mesh)


    J = dt*0.5 * gamma * h_avg**2 * dot(jump(grad(u), n), jump(grad(v), n))* dS



    # RHS
    L = (u1 + dt*f)*v*dx

    A = assemble(a + J) # assemble once before time stepping
    #A = assemble(a)

    # Compute solution
    u = Function(V) # unknown function at a new time level.
    T = 1
    t = dt

    while t <= T:
        b = assemble(L)
        bc.apply(A,b)
        solve(A, u.vector(), b)

        t += dt
        u1.assign(u)
        #plot(u)
        #interactive()

    u_e = interpolate(u0, V)

    plot(u)
    plot(u_e)
    interactive()

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



    #solve(a == L,u,bc)
    # Plot solution and mesh
    #plot(u)
    # Dump solution to file in VTK format
    file = File("RVcompU.pvd")
    file << u
    file = File("RVcompE.pvd")
    file << u_e
    #interactive()



    return E1

# Perform experiments
h = []  # element sizes
E = []  # errors
#N = [20,40,80,160]
#N = [40]
#h = [1.0/4, 1.0/8, 1.0/16, 1.0/32]
h = [1.0/16]
for i in range(0,len(h)):
    #h.append(1.0/nx)
    E.append(compute(h[i]))  # list of dicts


for err in E:
    print err
print ""


#p = np.polyfit(h,E,1);
#print p

"""
no J
0.245775750725
0.207385735203
0.17350450779
0.146014984365

[ 0.43544407  0.14214164]
"""

"""
with J
0.249538747793
0.20933984302Nn
0.174346095067
0.146470450084

[ 0.45075504  0.14210093]
"""
