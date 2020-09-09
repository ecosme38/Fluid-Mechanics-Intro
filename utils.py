"""Auxiliary functions used in several notebooks of the repo."""

import numpy as np

Nx, Ny = 100, 80    # Number of grid points in the domain
Lx, Ly = 1000, 800  # Size of the domain in km

def make_grid():
    """Make a 1000 x 800 km^2 grid with Nx x Ny grid points. Returns:
    * Nx, Ny
    * dx, dy: grid steps
    * xm, ym: meshgrid arrays
    """
    dx = np.array([Lx/Nx, Ly/Ny])   # grid steps, regular, in km
    x, y =np.arange(0,Lx,dx[0]), np.arange(0,Ly,dx[1]) # Zonal and meridional coordinates in km
    ym, xm = np.meshgrid(y,x)
    return dx, xm, ym

dx, xm, ym = make_grid()

def grid_param():
    return dx, Nx, Ny, xm, ym

def set_boundaries_to_zero(field):
    f = np.copy(field)
    f[0,:]  = f[-1,:] = f[:,0]  = f[:,-1] = 0
    return f

def von_neuman_euler(field, axis=None):
    """Apply Von Neuman boundary conditions to the field."""
    f = np.copy(field)
    if axis is 0 or None:
        f[0,:]  = f[1,:]
        f[-1,:] = f[-2,:]
    if axis is 1 or None:
        f[:,0]  = f[:,1]
        f[:,-1] = f[:,-2]
    return f

def derivative(field, axis=0):
    """Compute partial derivative along given axis using second-order centered scheme."""
    f = 0.5*( np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis) ) / dx[axis]
    f = von_neuman_euler(f, axis=axis)
    return f

def gradient(field):
    """Compute gradient of input scalar field."""
    fx, fy = derivative(field, axis=0), derivative(field, axis=1)
    return fx, fy

def divergence(u, v):
    """Compute divergence of a 2D-vector with components u, v."""
    f = derivative(u, axis=0) + derivative(v, axis=1)
    return f

def rotational(u, v):
    """Compute vertical component of rotational of a 2D-vector with components u, v."""
    f = derivative(v, axis=0) - derivative(u, axis=1)
    return f

def laplacian(field):
    """Compute Laplacian of field."""
    fx = np.roll(field, -1, axis=0) -2 * field + np.roll(field, 1, axis=0)
    fy = np.roll(field, -1, axis=1) -2 * field + np.roll(field, 1, axis=1)
    f = fx + fy
    f = von_neuman_euler(f)
    return f

def make_sla(x,y):
    """Create a 2D field with arbitrary analytical functions"""
    x,y = x/100, y/100
    a = 0.6*(x-4)**2 + 0.3*(y-3)**2
    b = 0.45*(x-8)**2 + 0.55*(y-5)**2
    d = 0.2*(x-2.5)**2 + 0.9*(y-6)**2
    c = np.exp(-a/2)-0.5*np.exp(-b/2)-0.6*np.exp(-d)
    c /= np.max(c)
    return c