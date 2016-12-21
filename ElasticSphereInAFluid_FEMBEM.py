from dolfin import *
import bempp.api
from bempp.api import fenics_interface
import numpy as np
from bempp.api.utils.linear_operator import aslinearoperator
from bempp.api.fenics_interface import FenicsOperator
from matplotlib import pyplot as plt
#Parameters ----------------------------------------
freq = 1000.
c0 = 1500. #sound speed of exterior fluid
rho0 = 1000. #Density of exterior fluid
k0 = 2*np.pi*freq/c0 #Wavenumber of exterior fluid
omega = 2*pi*freq
#Propagation direction vector
d = np.array([0., 0., 1.])
d /= np.linalg.norm(d)
#Lame parameters of the solid
Y = 200E9
nu = 0.3
lambda_s = Y*nu/((1+nu)*(1-2*nu))
mu_s = Y/(2*(1+nu))
rho_s = 8000
cp = np.sqrt((lambda_s+2*mu_s)/rho_s)
cs = np.sqrt(mu_s/rho_s)
print('Compressional Sound speed: ' + str(cp) + ' m/s')
print('Shear Sound speed: ' + str(cs) + ' m/s')
#Generate mesh
mesh = Mesh('Sphere.xml')
#mesh = UnitCubeMesh(10,10,10)

#Define fenics and bempp space
fenics_space = VectorFunctionSpace(mesh,'CG',1)
p_space = FunctionSpace(mesh,'CG',1)

#Get the pressure trace mapping from FENICS
trace_space, trace_matrix  = fenics_interface.coupling.fenics_to_bempp_trace_data(p_space)
#Define pressure space
bempp_space = bempp.api.function_space(trace_space.grid,"P",1)
print("FEM DOFs (Degrees of Freedom): {0}".format(fenics_space.dim()))
print("BEM DOFs: {0}".format(bempp_space.global_dof_count))
#Fenics linear elastic weak Form
#Define trial and test vector functions
u = TrialFunction(fenics_space)
du = TestFunction(fenics_space)
#Define pressure test and trial functions
p = TrialFunction(p_space)
dp = TestFunction(p_space) #Need this for coupling
#Get normal vectors
n = FacetNormal(mesh)

#Linear elasticity
#Edu = 0.5*(grad(du)+transpose(grad(du))) #Incremental strain
Edu = sym(grad(du))
Eu = sym(grad(u))
#Eu = 0.5*(grad(u)+transpose(grad(u))) #Incremental strain
I = Identity(len(u)) #Identity tensor
Stress = lambda_s*I*tr(Eu) + 2.*mu_s*Eu

#Linear elastic Weak form
DomainIntegral = (inner(Stress,Edu) - rho_s*omega**2*dot(u,du))*dx
SurfaceIntegral = p*dot(du,n)*ds
# BEM operators
id_op = bempp.api.operators.boundary.sparse.identity(trace_space, bempp_space, bempp_space)

dlp = bempp.api.operators.boundary.helmholtz.double_layer(trace_space, bempp_space, bempp_space, k0)
slp = bempp.api.operators.boundary.helmholtz.single_layer(bempp_space, bempp_space, bempp_space, k0)

#Create blocked matrix
print("Constructing block matrix...")
blocked = bempp.api.BlockedDiscreteOperator(3, 3)
trace_op = aslinearoperator(trace_matrix)
blocked[0,0] = FenicsOperator(DomainIntegral).weak_form()
blocked[0,1] = FenicsOperator(SurfaceIntegral).weak_form()*trace_op.adjoint()
blocked[1,1] = (.5*id_op - dlp).weak_form()
blocked[1,2] = slp.weak_form()
blocked[2,0] = trace_op*FenicsOperator(-rho0*omega**2*dot(u,n)*dp*ds).weak_form()
blocked[2,2] = bempp.api.operators.boundary.sparse.identity(bempp_space, trace_space, trace_space).weak_form()
print("Block matrix constructed.")
#Define incident Wave
def p_inc(x, n, domain_index, result):
    result[0] = np.exp(1j*k0*np.dot(x, d))
p_inc = bempp.api.GridFunction(trace_space, fun=p_inc)
print("Constructing preconditioner...")
preCond = bempp.api.BlockedDiscreteOperator(3, 3)
preCond[0,0] = bempp.api.InverseSparseDiscreteBoundaryOperator(blocked[0,0].sparse_operator.tocsc())
preCond[1,1] = bempp.api.InverseSparseDiscreteBoundaryOperator(bempp.api.operators.boundary.sparse.identity(trace_space, trace_space, trace_space).weak_form())
preCond[2,2] = bempp.api.InverseSparseDiscreteBoundaryOperator(blocked[2,2].sparse_operator.tocsc())
print("Preconditioner constructed")
#Right hand side
rhs_u = np.zeros(fenics_space.dim())
rhs_p = p_inc.projections(trace_space)
rhs_lambda = np.zeros(bempp_space.global_dof_count)
rhs = np.concatenate([rhs_u,rhs_p,rhs_lambda])

#Solve system of equations
# Create a callback function to count the number of iterations
it_count = 0
def count_iterations(x):
    global it_count
    it_count += 1

from scipy.sparse.linalg import gmres
print("Solving...")
soln, info = gmres(blocked, rhs, callback=count_iterations,M=preCond,tol=1e-18)
print("Solving complete.")
#soln = np.load("SolutionVector.npy")
#Extract BEM dirchlet and neumann data
u_nodes = soln[0:fenics_space.dim()]
dirichlet_data = soln[fenics_space.dim():fenics_space.dim()+trace_space.global_dof_count]
dirichlet_fun = bempp.api.GridFunction(trace_space, coefficients=dirichlet_data)
neumann_data = soln[fenics_space.dim()+trace_space.global_dof_count:]
neumann_fun = bempp.api.GridFunction(bempp_space, coefficients=neumann_data)

#Create plot Grid
#bempp.api.global_parameters.hmat.eps = 1E-2
Nx = 200
Nz = 200
zmin,zmax,xmin,xmax=[-3,3,-3,3]
xplt = np.linspace(xmin,xmax,Nx)
zplt = np.linspace(zmin,zmax,Nz)
X,Z = np.meshgrid(xplt,zplt)
points = np.vstack((X.ravel(), np.zeros(Nx*Nz),Z.ravel()))

x, y, z =points
bem_x = np.sqrt(x**2 + z**2) > 1.0
#Evaluate potentials

#bempp.api.global_parameters.hmat.eps = 1E-6
print("Constructing potentials...")
slp_pot= bempp.api.operators.potential.helmholtz.single_layer(bempp_space,points[:,bem_x],k0)
dlp_pot= bempp.api.operators.potential.helmholtz.double_layer(trace_space,points[:,bem_x],k0)
print("Potentials constructed")
#plot_me[bem_x] += np.exp(1j*k0*(points[0,bem_x]*d[0]+points[1,bem_x]*d[1]+points[2,bem_x]*d[2]))
plot_me = np.zeros(Nx*Nz, dtype='complex')
plot_me[bem_x] = +np.squeeze(dlp_pot.evaluate(dirichlet_fun)) - np.squeeze(slp_pot.evaluate(neumann_fun))
#Q = dlp_pot*dirichlet_fun
#S = slp_pot*neumann_fun
#plot_me[bem_x] += np.squeeze(Q.T)
#plot_me[bem_x] -= np.squeeze(S.T)
plot_me = plot_me.reshape((Nx,Nz))
#plot_me=np.load("SolutionPlot.npy")
#plot_me = plot_me.transpose()[::-1]

# Plot the image
contourLines = np.linspace(0,2.8,10)
fig=plt.figure()
plt.contourf(X,Z,abs(plot_me),contourLines,extent=[xmin,xmax,zmin,zmax])
plt.xlabel('x')
plt.ylabel('z')
plt.colorbar()
plt.title("FEM-BEM Coupling for Helmholtz")
plt.show()

#Plot pressure on sphere
from AnalyticalSoln import ElasticSphere_Analytical
coords = trace_space.grid.leaf_view.vertices
pA,dpdrA = ElasticSphere_Analytical(coords[0,:],coords[1,:],coords[2,:])
dirichlet_fun_A = bempp.api.GridFunction(trace_space, coefficients=pA)
neumann_fun_A = bempp.api.GridFunction(bempp_space,coefficients=dpdrA)
