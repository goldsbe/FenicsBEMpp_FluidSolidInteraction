import numpy as np
from scipy.special import spherical_jn as jn
from scipy.special import lpmv as Pn
from scipy.special import spherical_yn as nn
import matplotlib.pyplot as plt
import pdb
#Physical parameters of the sphere
radius =1.
freq = 1000.
omega = 2*np.pi*freq
Y = 200E9 #Young's modulus
nu = 0.3 #Poisson's ratio
rho_s = 8000 # Density of the solid
rho_0 = 1000 #Density of fluid
c0 = 1500 #Sound speed of fluid
#Calculate lame parameters of the solid
lambda_s = Y*nu/((1+nu)*(1-2*nu))
mu_s = Y/(2*(1+nu))
cp = np.sqrt((lambda_s+2*mu_s)/rho_s) #Compressional sound speed
cs = np.sqrt(mu_s/rho_s) #Shear sound speed
print('Compressional Sound speed: ' + str(cp) + ' m/s')
print('Shear Sound speed: ' + str(cs) + ' m/s')
#Calculate wavenumbers
k1 = omega/cp
k2 = omega/cs
k3 = omega/c0
x1 = k1*radius
x2 = k2*radius
x3 = k3*radius

#Define 2D grid
Nx = 200
Nz = 200
zmin,zmax,xmin,xmax=[-3,3,-3,3]
xplt = np.linspace(xmin,xmax,Nx)
zplt = np.linspace(zmin,zmax,Nz)
X,Z = np.meshgrid(xplt,zplt)
points = np.vstack((X.ravel(), np.zeros(Nx*Nz),Z.ravel()))

# Loop over points
N = 100 #Number of terms in series
p = np.zeros(Nx*Nz,dtype='complex')
for x,z,ii in zip(X.ravel(),Z.ravel(),range(Nx*Nz)):
    r = np.sqrt(x**2+z**2)
    if r > radius:
        cosTheta = z/r
        pn = np.zeros(N+1,dtype='complex')
        for n in range(N+1):
            #Compute alphas for x1,x2,x3 using intermediate angles
            anx1 = -x1*jn(n,x1,derivative=True)/jn(n,x1)
            anx2 = -x2*jn(n,x2,derivative=True)/jn(n,x2)
            anx3 = -x3*jn(n,x3,derivative=True)/jn(n,x3)

            #Calculate numerator for Eq.30
            jnppx2 = -jn(n+1,x2,derivative=True) - n/x2**2*jn(n,x2) + n/x2*jn(n,x2,derivative=True)
            jnppx1 = -jn(n+1,x1,derivative=True) - n/x1**2*jn(n,x1) + n/x1*jn(n,x1,derivative=True)
            num = anx1/(anx1+1) - (n**2+n)/(n**2+n-1-0.5*x2**2+anx2)
            den = (n**2+n-0.5*x2**2+2*anx1)/(anx1+1) - (n**2+n)*(anx2+1)/(n**2+n-1-0.5*x2**2+anx2)
            num2 = x1*jn(n,x1,derivative=True)/(x1*jn(n,x1,derivative=True)-jn(n,x1)) - 2*(n**2+n)*jn(n,x2)/((n**2+n-2)*jn(n,x2)+x2**2*jnppx2)
            den2 = nu/(1.-2.*nu)*x1**2*(jn(n,x1)-jnppx1)/(x1*jn(n,x1,derivative=True)-jn(n,x1)) - 2*(n**2+n)*(jn(n,x2)-x2*jn(n,x2,derivative=True))/((n**2+n-2)*jn(n,x2)+x2**2*jnppx2)
            #Calculate EQ.30
            xi = -(x2**2)/2.*num/den

            #Compute the other intermediate angles
            deltan = -jn(n,x3)/nn(n,x3)
            betan = -x3*nn(n,x3,derivative=True)/nn(n,x3)

            #Compute Eq. 29
            Phin = -rho_0/rho_s*xi

            #Compute eta
            eta = np.arctan(deltan*(Phin+anx3)/(Phin+betan))

            #Compute eq. 28
            cn = -(2.*n+1)*(-1j)**(n+1)*np.sin(eta)*np.exp(1j*eta)
            pn[n] = cn*(jn(n,k3*r)-1j*nn(n,k3*r))*Pn(0,n,cosTheta)
        if abs(pn[-1]) > 1e-6:
            print("Need more N Values!!!")
        p[ii] = np.sum(pn)

pPlt = p.reshape(X.shape)
p_inc = np.exp(-1j*k3*Z)
contourLines = np.linspace(0,2.8,10)
plt.figure(1)
plt.contourf(X,Z,np.abs(pPlt),contourLines)
plt.colorbar()
plt.show()
