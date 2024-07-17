import numpy as np

# Classes ------------------------------------------------------------------------

class CrossSections:
    def __init__(self, Sigma_t, Sigma_s, Sigma_f, nu):
        self.Sigma_t = Sigma_t
        self.Sigma_s = Sigma_s
        self.Sigma_f = Sigma_f
        self.nu = nu

class Quadrature:
    def __init__(self, mu, w):
        self.mu = mu
        self.w = w
        self.N_mu = len(mu)

class Mesh:
    def __init__(self, Nx, dx, Nt, dt,quadrature):
        self.dx = dx
        self.Nx = Nx
        self.dt = dt
        self.Nt = Nt
        self.quad = quadrature
        x_edge = np.zeros(Nx+1)
        for i in range(1,Nx+1):
            x_edge[i] = x_edge[i-1]+dx[i-1]
        x = np.zeros(Nx+2)
        x[1:-1] = x_edge[1:]-0.5*dx
        x[0] = x_edge[0]
        x[-1] = x_edge[-1]
        self.x = x
        self.x_edge = x_edge
        t = np.zeros(Nt+1)
        for i in range(1,Nt+1):
            t[i] = t[i-1]+dt
        self.t = t

class Source:
    def __init__(self, q, v,lb,rb):
        self.q = q
        self.v = v
        self.lb = lb
        self.rb = rb

class Problem:
    def __init__(self, cross_sections, mesh, source):
        self.xs = cross_sections
        self.mesh = mesh
        self.source = source

class iter_data:
    def __init__(self, n_iter,epsilon = 1e-10, n_max = 1000):
        self.n_iter = n_iter
        self.difference= []
        self.epsilon = epsilon
        self.n_max = n_max

class State:
    def __init__(self, flux, current):
        self.flux = flux
        self.current = current

        self.psi_edge = []
        self.psi_bar = []
        self.F = np.zeros_like(flux)
        self.F_edge = np.zeros_like(current)
        self.Pl = 0.0
        self.Pr = 0.0
        self.residual = []
        self.iterations = []
