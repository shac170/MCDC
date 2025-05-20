import math
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from numba import objmode, literal_unroll

import mcdc.type_ as type_
import mcdc.adapt as adapt
import mcdc.src.geometry as geometry
import mcdc.src.mesh as mesh_
import mcdc.src.physics as physics
import mcdc.src.surface as surface_

from mcdc.adapt import toggle
from mcdc.constant import *
from mcdc.type_ import iqmc_score_list
import scipy as sp
from scipy.sparse import bsr_array

# =============================================================================
# Preprocess functions
# =============================================================================


def losm_preprocess(mcdc):
    # set bank source
    print("LOSM pre-processing")
    # generate material index
    losm_generate_material_idx(mcdc)



def losm_generate_material_idx(mcdc):
    """
    This algorithm is meant to loop through every spatial cell of the
    hybrid mesh and assign a material index according to the material_ID at
    the center of the cell.

    Therefore, the whole cell is treated as the material located at the
    center of the cell, regardless of whethere there are more materials
    present.

    A crude but quick approximation.
    """
    mesh = mcdc["technique"]["losm"]["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    # create particle to utilize cell finding functions
    P_temp_arr = adapt.local_array(1, type_.particle)
    P_temp = P_temp_arr[0]
    # set default attributes
    P_temp["alive"] = True

    x_mid = 0.5 * (mesh["x"][1:] + mesh["x"][:-1])
    y_mid = 0.5 * (mesh["y"][1:] + mesh["y"][:-1])
    z_mid = 0.5 * (mesh["z"][1:] + mesh["z"][:-1])

    # loop through every cell
    for t in range(Nt):
        for i in range(Nx):
            x = x_mid[i]
            for j in range(Ny):
                y = y_mid[j]
                for k in range(Nz):
                    z = z_mid[k]

                    # assign cell center position
                    P_temp["t"] = t
                    P_temp["x"] = x
                    P_temp["y"] = y
                    P_temp["z"] = z
                    P_temp["material_ID"] = -1
                    P_temp["cell_ID"] = -1
                    P_temp["g"] = 0

                    # set material_ID
                    geometry.locate_particle(P_temp_arr, mcdc)

                    # assign material index
                    mcdc["technique"]["losm"]["material_idx"][t, i, j, k] = P_temp[
                        "material_ID"
                    ]


def losm_create_problem(mcdc):
    mesh = mcdc["technique"]["losm"]["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1

    x_mid = 0.5 * (mesh["x"][1:] + mesh["x"][:-1])
    # initialize material data
    siga = np.zeros_like(x_mid)
    sigt = np.zeros_like(x_mid)
    sigf = np.zeros_like(x_mid)
    nu = np.ones_like(x_mid)
    q = np.zeros_like(x_mid)

    # loop through every cell
    for t in range(1):
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # assign material index
                    mat_idx = mcdc["technique"]["losm"]["material_idx"][t, i, j, k] 
                    sigt[i] = mcdc["materials"][mat_idx]["total"]
                    sigf[i] = mcdc["materials"][mat_idx]["fission"]    
                    siga[i] = mcdc["materials"][mat_idx]["capture"] + sigf[i]   
                    nu[i] = mcdc["materials"][mat_idx]["nu_f"] 
                    for source in mcdc["sources"]:
                        if (
                            mesh["t"][t + 1] <= source["time"][1]
                            and mesh["t"][t] >= source["time"][0]
                        ):
                            
                            if source["box"] == 0:
                                if (
                                    x_mid[i] == source["x"]
                                ):
                                    q[i] = source["prob"] 
                                else:
                                    in_x = mesh["x"][i] <= source["x"]  <= mesh["x"][i+1]

                                    if in_x:
                                        dx = 1
                                        if (mesh["x"][i] != -INF) and (mesh["x"][i] != INF):
                                            dx = mesh["x"][i + 1] - mesh["x"][i]
                                        q[i] = source["prob"] / dx
                                        
                            else:
                                
                                in_x = mesh["x"][i] >= source["box_x"][0] and mesh["x"][i+1] <= source["box_x"][1]
                                if in_x:
                                    dx = 1
                                    if (mesh["x"][i] != -INF) and (mesh["x"][i] != INF):
                                        dx = mesh["x"][i + 1] - mesh["x"][i]
                                    q[i] = source["prob"] * dx

    # dict for problem
    problem={'x_mesh':mesh["x"],
            'siga':siga,
            'sigt':sigt,
            'sigf':sigf,
            'nu':nu,
            'v':1,
            'source':q}
    return problem

# =============================================================================
# Utility functions
# =============================================================================
def losm_create_state(phi,J):
    Nx = int(len(phi))
    state = np.zeros(Nx*8)
    for i in range(Nx):
        for i1 in range(4):
            state[i*8+i1] = phi[i]
        for i2 in range(4):
            state[i*8+4+i2] = J[i]
    return state

def losm_convert_moments(c,mx,mt,mtx,x_mesh,t0,t1):
    Nx = len(c)
    dx = x_mesh[1:] - x_mesh[:-1]
    x_mid = (x_mesh[1:] + x_mesh[:-1]) / 2
    dt = t1 - t0
    t_mid = (t0 + t1) / 2

    F = np.zeros((4,Nx))
    for i in range(Nx):
        F[0,i] =  (c[i]
            + 2*(x_mesh[i]-x_mid[i])*mx[i]/dx[i]
            + 2*(t0-t_mid)*mt[i]/dt
            + 4*(x_mesh[i]-x_mid[i])*(t0-t_mid)*mtx[i]/(dt*dx[i]))
            
        F[1,i] =  (c[i]
            + 2*(x_mesh[i+1]-x_mid[i])*mx[i]/dx[i]
            + 2*(t0-t_mid)*mt[i]/dt
            + 4*(x_mesh[i+1]-x_mid[i])*(t0-t_mid)*mtx[i]/(dt*dx[i]))
            
        F[2,i] =  (c[i]
            + 2*(x_mesh[i+1]-x_mid[i])*mx[i]/dx[i]
            + 2*(t1-t_mid)*mt[i]/dt
            + 4*(x_mesh[i+1]-x_mid[i])*(t1-t_mid)*mtx[i]/(dt*dx[i]))
        
        F[3,i] =  (c[i]
            + 2*(x_mesh[i]-x_mid[i])*mx[i]/dx[i]
            + 2*(t1-t_mid)*mt[i]/dt
            + 4*(x_mesh[i]-x_mid[i])*(t1-t_mid)*mtx[i]/(dt*dx[i]))
    return F
 
@toggle("hybrid")
def losm_cell_volume(x, y, z, mesh):
    """
    Calculate the volume of the cartesian spatial cell.

    """
    dx = dy = dz = 1
    if (mesh["x"][x] != -INF) and (mesh["x"][x] != INF):
        dx = mesh["x"][x + 1] - mesh["x"][x]
    if (mesh["y"][y] != -INF) and (mesh["y"][y] != INF):
        dy = mesh["y"][y + 1] - mesh["y"][y]
    if (mesh["z"][z] != -INF) and (mesh["z"][z] != INF):
        dz = mesh["z"][z + 1] - mesh["z"][z]
    dV = dx * dy * dz
    return dV

# =============================================================================
# Solver Functions
# =============================================================================

def make_matrix(problem,old_state,closure,dt,sparse):
    """
    Creates the block matrix for the linear discontinous low order second moment method solve.

    """
    siga = problem["siga"]
    sigt = problem["sigt"]
    sigf = problem["sigf"]
    nu = problem["nu"]
    x = problem["x_mesh"]
    v = problem["v"]
    source = problem["source"]
    dx = x[1:]-x[:-1]
    Nx = len(dx)
    
    if not sparse:
        # making A 
        A = np.zeros((Nx*8,Nx*8))
        for i in range(Nx):
            L,C,R = make_submatrices(siga[i],sigt[i],sigf[i],nu[i],dx[i],dt,v)
            if i> 0:
                A[(i-1)*8:(i)*8,(i)*8:(i+1)*8] = L
            A[(i)*8:(i+1)*8,(i)*8:(i+1)*8] = C
            if i<Nx-1:
                A[(i+1)*8:(i+2)*8,(i)*8:(i+1)*8] = R
    else:
        data = []
        indices = []
        indptr = [0]

        for i in range(Nx):
            L, C, R = make_submatrices(siga[i], sigt[i], sigf[i], nu[i], dx[i], dt, v)
            row_blocks = []
            row_indices = []

            if i > 0:
                row_blocks.append(L)
                row_indices.append(i - 1)

            row_blocks.append(C)
            row_indices.append(i)

            if i < Nx - 1:
                row_blocks.append(R)
                row_indices.append(i + 1)

            data.extend(row_blocks)
            indices.extend(row_indices)
            indptr.append(len(indices))

        A = bsr_array((data, indices, indptr), shape=(Nx*8, Nx*8), blocksize=(8, 8))

    # making b
    b = np.zeros(Nx*8)

    P = closure["P"]
    T = closure["T"]
    F = closure["F"]
    plt.clf()
    plt.plot(F[0,:])
    plt.plot(F[1,:])
    plt.plot(F[2,:])
    plt.plot(F[3,:])
    plt.title("F")
    plt.show()
    for i in range(1,Nx-1):
        phi_old = old_state[i*8:i*8+4]
        J_old = old_state[i*8+4:i*8+8]
        b[i*8] = dx[i]*dt*source[i]/4 + P[0,i-1] + P[1,i] + (phi_old[2]+2*phi_old[3])*dx[i]/(v*6) 
        b[i*8+1] = dx[i]*dt*source[i]/4 - P[0,i] - P[1,i+1] + (2*phi_old[2]+phi_old[3])*dx[i]/(v*6)
        b[i*8+2] = dx[i]*dt*source[i]/4 - P[2,i] - P[3,i+1]
        b[i*8+3] = dx[i]*dt*source[i]/4 + P[2,i-1] + P[3,i]

        b[i*8+4] = (1/3)*(T[0,i-1]+T[1,i])  - F[0,i-1] + F[1,i] + (J_old[2]+2*J_old[3])*dx[i]/(v*6) 
        b[i*8+5] = -(1/3)*(T[0,i]+T[1,i+1]) + F[0,i] - F[1,i] + (2*J_old[2]+J_old[3])*dx[i]/(v*6)
        b[i*8+6] = -(1/3)*(T[2,i]+T[3,i+1]) + F[2,i] - F[3,i]
        b[i*8+7] = (1/3)*(T[2,i-1]+T[3,i])  - F[2,i-1] + F[3,i]
    bc = "zero"
    if bc == "zero":
        i = 0
        phi_old = old_state[i*8:i*8+4]
        J_old = old_state[i*8+4:i*8+8]
        b[i*8] = dx[i]*dt*source[i]/4  + P[1,i] + (phi_old[2]+2*phi_old[3])*dx[i]/(v*6) 
        b[i*8+1] = dx[i]*dt*source[i]/4 - P[0,i] - P[1,i+1] + (2*phi_old[2]+phi_old[3])*dx[i]/(v*6)
        b[i*8+2] = dx[i]*dt*source[i]/4 - P[2,i] - P[3,i+1]
        b[i*8+3] = dx[i]*dt*source[i]/4  + P[3,i]

        b[i*8+4] = (1/3)*(T[1,i])  - F[0,i-1] + F[1,i] 
        b[i*8+5] = -(1/3)*(T[0,i]+T[1,i+1]) + F[0,i] - F[1,i] 
        b[i*8+6] = -(1/3)*(T[2,i]+T[3,i+1]) + F[2,i] - F[3,i]
        b[i*8+7] = (1/3)*(T[3,i])  - F[2,i-1] + F[3,i]
        
        i = Nx-1
        phi_old = old_state[i*8:i*8+4]
        J_old = old_state[i*8+4:i*8+8]
        b[i*8] = dx[i]*dt*source[i]/4 + P[0,i-1] + P[1,i] + (phi_old[2]+2*phi_old[3])*dx[i]/(v*6) 
        b[i*8+1] = dx[i]*dt*source[i]/4 - P[0,i] + (2*phi_old[2]+phi_old[3])*dx[i]/(v*6)
        b[i*8+2] = dx[i]*dt*source[i]/4 - P[2,i] 
        b[i*8+3] = dx[i]*dt*source[i]/4 + P[2,i-1] + P[3,i]

        b[i*8+4] = (1/3)*(T[0,i-1]+T[1,i])  - F[0,i-1] + F[1,i] 
        b[i*8+5] = -(1/3)*(T[0,i]) + F[0,i] - F[1,i] 
        b[i*8+6] = -(1/3)*(T[2,i]) + F[2,i] - F[3,i]
        b[i*8+7] = (1/3)*(T[2,i-1]+T[3,i])  + F[2,i-1] + F[3,i]
    
    return A, b

def make_submatrices(siga,sigt,sigf,nu,dx,dt,v):
    """
    Creates the submatrix for the linear discontinous low order second moment method block matrix.

    """
    s1 = (siga-nu*sigf)*dx*dt
    s2 = sigt*dx*dt
    dxv =dx/v

    L = np.array([[0,-dt/12,-dt/24,0,0,-dt/6,-dt/12,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,-dt/24,-dt/12,0,0,-dt/12,-dt/6,0],
                  [0,-dt/18,-dt/36,0,0,-dt/12,-dt/24,0],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [0,-dt/36,-dt/18,0,0,-dt/24,-dt/12,0]])
    
    R = np.array([[0,0,0,0,0,0,0,0],
                  [-dt/12,0,0,-dt/24,dt/6,0,0,dt/12],
                  [-dt/24,0,0,-dt/12,dt/12,0,0,dt/6],
                  [0,0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0,0],
                  [dt/18,0,0,dt/36,-dt/12,0,0,-dt/24],
                  [dt/36,0,0,dt/18,-dt/24,0,0,-dt/12],
                  [0,0,0,0,0,0,0,0]])
    
    C = np.array([
            [dxv/6+dt/12+s1/9,dxv/12+s1/18,dxv/12+s1/36,dxv/6+dt/24+s1/18,                  0,dt/6,dt/12,0],
            [dxv/12+s1/18,dxv/6+dt/12+s1/9,dxv/6+dt/24+s1/18,dxv/12+s1/36,                  -dt/6,0,0,-dt/12],
            [-dxv/12+s1/36,-dxv/6+dt/24+s1/18,dxv/3-dxv/6+dt/12+s1/9,dxv/6-dxv/12+s1/18,    -dt/12,0,0,-dt/6],
            [-dxv/6+dt/24+s1/18,-dxv/12+s1/36,dxv/6-dxv/12+s1/18,dxv/3-dxv/6+dt/12+s1/9,   0,dt/12,dt/6,0],
            
            [0,dt/18,dt/36,0,           dxv/6+dt/12+s2/9,dxv/12+s2/18,dxv/12+s2/36,dxv/6+dt/24+s2/18],
            [-dt/18,0,0,-dt/36,         dxv/12+s2/18,dxv/6+dt/12+s2/9,dxv/6+dt/24+s2/18,dxv/12+s2/36],
            [-dt/36,0,0,-dt/18,         -dxv/12+s2/36,-dxv/6+dt/12+s2/18,dxv/3-dxv/6+dt/12+s2/9,dxv/6-dxv/12+s2/18],
            [0,dt/36,dt/18,0,           -dxv/6+dt/24+s2/18,-dxv/12+s2/36,dxv/6-dxv/12+s2/18,dxv/3-dxv/6+dt/12+s2/9]])

    return L, C, R 

def losm_step_time(problem,old_state,closure,dt,save_a=False,sparse=True):
    """
    Performs one timestep of the linear discontinous low-order second moment method

    """
    A, q = make_matrix(problem,old_state,closure,dt,sparse)
    if save_a:
        plt.clf()
        plt.pcolormesh(A)
        plt.colorbar()
        plt.show()
        plt.clf()
    if sparse:
        new_state = sp.sparse.linalg.spsolve(A,q)
    else:
        new_state = np.linalg.solve(A,q)
    return new_state

def losm_average_state(state):
    Nx = int(len(state)/8)
    phi = np.zeros(Nx)
    J = np.zeros(Nx+1)
    for i in range(Nx+1):
        if i< Nx:
            # phi_c is the average of the 4 corner values
            phi[i]=np.average(state[i*8:i*8+4])
            
            # J is the average of the 4 edge values (or two on the boundaries)
            if i>0:
                J[i]=(state[i*8+4]+state[i*8+7]+state[(i-1)*8+4]+state[(i-1)*8+7])/4
            else:
                J[i]=(state[i*8+4]+state[i*8+7])/2
        else:
            J[i]=(state[(i-1)*8+4]+state[(i-1)*8+7])/2
    return phi,J