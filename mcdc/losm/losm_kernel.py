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
    idx_census = mcdc["idx_census"]
    epsilon = mcdc["technique"]["ww"]["epsilon"]
    
    space_scheme = epsilon[WW_SPACE_DISC] 
    initial_conditions = epsilon[WW_HYBRID_IC]
    if space_scheme == HYBRID_FV:
        time_scheme = epsilon[WW_TIME_DISC] 

    mesh = mcdc["technique"]["losm"]["mesh"]
    Nt = len(mesh["t"]) - 1
    Nx = len(mesh["x"]) - 1
    Ny = len(mesh["y"]) - 1
    Nz = len(mesh["z"]) - 1
    dt = mesh["t"][mcdc["idx_census"]+1]-mesh["t"][mcdc["idx_census"]]
    x_mid = 0.5 * (mesh["x"][1:] + mesh["x"][:-1])
    # initialize material data
    siga = np.zeros_like(x_mid)
    sigt = np.zeros_like(x_mid)
    sigf = np.zeros_like(x_mid)
    sigs = np.zeros_like(x_mid)
    nu = np.ones_like(x_mid)
    q = np.zeros_like(x_mid)
    q_old = np.zeros_like(x_mid)

    # loop through every cell
    t = mcdc["idx_census"]
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                # assign material index
                mat_idx = mcdc["technique"]["losm"]["material_idx"][t, i, j, k] 
                sigt[i] = mcdc["materials"][mat_idx]["total"]
                sigf[i] = mcdc["materials"][mat_idx]["fission"]    
                siga[i] = mcdc["materials"][mat_idx]["capture"] + sigf[i]   
                sigs[i] = mcdc["materials"][mat_idx]["scatter"]
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
                                q[i] = source["prob"] * (dx/dt)/2
                                ##
                    elif (
                        mesh["t"][t + 1] >= source["time"][1]
                        and mesh["t"][t] <= source["time"][0]
                    ):
                        if source["box"] == 0:
                            if (
                                x_mid[i] == source["x"]
                            ):
                                dx = mesh["x"][i + 1] - mesh["x"][i]
                                q[i] = source["prob"]  / (dx * dt)
                            else:
                                in_x = mesh["x"][i] <= source["x"]  <= mesh["x"][i+1]

                                if in_x:
                                    if (mesh["x"][i] != -INF) and (mesh["x"][i] != INF):
                                        dx = mesh["x"][i + 1] - mesh["x"][i]
                                    q[i] = source["prob"] * (dx/dt)
                                    
                        else:
                            
                            in_x = mesh["x"][i] >= source["box_x"][0] and mesh["x"][i+1] <= source["box_x"][1]
                            if in_x:
                                dx = 1
                                if (mesh["x"][i] != -INF) and (mesh["x"][i] != INF):
                                    dx = mesh["x"][i + 1] - mesh["x"][i]
                                q[i] = source["prob"] * (dt)
                    if (
                        mesh["t"][t ] >= source["time"][1]
                        and mesh["t"][t-1] <= source["time"][0]
                    ):
                        if source["box"] == 0:
                            if (
                                x_mid[i] == source["x"]
                            ):
                                dx = mesh["x"][i + 1] - mesh["x"][i]
                                q_old[i] = source["prob"]  / (dx * dt)

    if mcdc["surfaces"][0]["BC"] == BC_REFLECTIVE:
        lb = 1
    else:
        lb = 0
    if mcdc["surfaces"][-1]["BC"] == BC_REFLECTIVE:
        rb = 1
    else:
        rb = 0

    if space_scheme == HYBRID_FV:
        # dict for problem
        problem={'x_mesh':mesh["x"],
                'siga':siga,
                'sigt':sigt,
                'sigf':sigf,
                'sigs':sigs,
                'nu':nu,
                'v':1,
                'source':q,
                'previous_source':q_old,
                'space_scheme':space_scheme,
                'intial_condition':initial_conditions,
                'time_scheme':time_scheme,
                'lb':lb,
                'rb':rb}
        
        
    elif space_scheme == HYBRID_FE:
        # dict for problem
        problem={'x_mesh':mesh["x"],
                'siga':siga,
                'sigt':sigt,
                'sigf':sigf,
                'nu':nu,
                'v':1,
                'source':q,
                'space_scheme':space_scheme,
                'intial_condition':initial_conditions,
                'lb':lb,
                'rb':rb}
    else:
        print("bad space_scheme",space_scheme,HYBRID_FE,HYBRID_FV)
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
        x0, x1 = x_mesh[i], x_mesh[i+1]
        xm = x_mid[i]
        dx0 = (x0 - xm) * (2/(x1-x0))
        dx1 = (x1 - xm) * (2/(x1-x0))
        dt_half = (0.5 * dt) * (2/(dt))

        F[0,i] =  (c[i]
            + 6*dx0 *mx[i]/dx[i]
            - 6*dt_half*mt[i]/dt
            - 36*dx0*dt_half*mtx[i]/(dt*dx[i]))
            
        F[1,i] =  (c[i]
            + 6*dx1 *mx[i]/dx[i]
            - 6*(t0-t_mid)*mt[i]/dt
            - 36*dx1*(t0-t_mid)*mtx[i]/(dt*dx[i]))
            
        F[2,i] =  (c[i]
            + 6*dx1 *mx[i]/dx[i]
            + 6*dt_half*mt[i]/dt
            + 36*dx1*dt_half*mtx[i]/(dt*dx[i]))
        
        F[3,i] =  (c[i]
            + 6*dx0 *mx[i]/dx[i]
            + 6*dt_half*mt[i]/dt
            + 36*dx0*dt_half*mtx[i]/(dt*dx[i]))
    '''
    plt.clf()
    plt.plot(F[0,:])
    plt.plot(F[1,:])
    plt.plot(F[2,:])
    plt.plot(F[3,:])
    plt.show()
    '''
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

def make_FE_matrix(problem,old_state,closure,dt,sparse):
    """
    Creates the block matrix for the linear discontinous (finite element) low order second moment method solve.

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
            L,C,R = make_FE_submatrices(siga[i],sigt[i],sigf[i],nu[i],dx[i],dt,v,i)
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
            L, C, R = make_FE_submatrices(siga[i], sigt[i], sigf[i], nu[i], dx[i], dt, v,i)
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

def make_FE_submatrices(siga,sigt,sigf,nu,dx,dt,v,i):
    """
    Creates the submatrix for the linear discontinous (finite element) low order second moment method block matrix.

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
    #boundary condition (Not implemented yet)
    if i == 0:
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

def make_FV_matrix(problem,old_state,closure,dt):
    # getting variables from problem and state
    phi_prev = np.copy(old_state[0])
    J_prev = np.copy(old_state[1])

    F_prev = closure["previous_F"]

    F = closure["F"]
    Pl = closure["Pl"]
    Pr = closure["Pr"]

    q = problem["source"]
    q_old = problem["previous_source"]
    x_mesh = problem["x_mesh"]
    dx = x_mesh[1:] - x_mesh[:-1]
    Nx = len(x_mesh) - 1
    v = problem["v"]
    Sigma_a = problem["siga"]
    Sigma_f = problem["sigf"]
    Sigma_t = problem["sigt"]
    Sigma_s = Sigma_t - Sigma_a
    nu = problem["nu"][0]
    time_scheme = problem["time_scheme"]

    # hardwired boundary conditions (bad)
    left_bc = problem["lb"]
    right_bc = problem["rb"]

    # Spatial edge calculations
    dx_edge = np.zeros(Nx + 1)
    dx_edge[1:-1] = (dx[:-1] + dx[1:]) / 2
    dx_edge[0] = dx[0] / 2
    dx_edge[-1] = dx[-1] / 2

    # Compute edge cross section
    Sigma_t_edge = np.zeros(Nx + 1)
    Sigma_t_edge[1:-1] = (Sigma_t[:-1] * dx[:-1] + Sigma_t[1:] * dx[1:]) / (
        dx[:-1] + dx[1:]
    )
    Sigma_t_edge[0] = Sigma_t[0]
    Sigma_t_edge[-1] = Sigma_t[-1]


    # Compute modified Sigma_t based on the method
    if time_scheme == HYBRID_BE:
        Sigma_t_mod = Sigma_t + 1 / (v * dt)
    elif time_scheme == HYBRID_CN:
        Sigma_t_mod = Sigma_t + 2 / (v * dt)

    Sigma_t_mod_edge = np.zeros(Nx + 1)
    Sigma_t_mod_edge[1:-1] = (Sigma_t_mod[:-1] * dx[:-1] + Sigma_t_mod[1:] * dx[1:]) / (
        dx[:-1] + dx[1:]
    )
    Sigma_t_mod_edge[0] = Sigma_t_mod[0]
    Sigma_t_mod_edge[-1] = Sigma_t_mod[-1]



    # Compute Q based on the method
    
    if time_scheme == HYBRID_BE:
        q0 = q + phi_prev[1:-1] / (v * dt)
        q1 = J_prev / (v * dt)
    elif time_scheme == HYBRID_CN:
        '''
        q0 = (
            q
            + q
            + (J_prev[:-1] - J_prev[1:]) / dx
            + (Sigma_s + nu * Sigma_f + 2 / (v * dt) - Sigma_t) * phi_prev
        )
        q1 = (
            (previous_state.F[1:] - previous_state.F[:-1])
            + (previous_state.flux[:-1] - previous_state.flux[1:]) / 3
        ) / dx_edge + (2 / (v * dt) - (Sigma_t_edge)) * J_prev

        '''
        q0 = (J_prev[:-1] - J_prev[1:])/dx + q + q_old + (Sigma_s + nu*Sigma_f+2/(v*dt)-Sigma_t)*phi_prev[1:-1] 
        q1 = (F_prev[1:] - F_prev[:-1]+(phi_prev[:-1]-phi_prev[1:])/3)/dx_edge + (2/(v*dt)-Sigma_t_edge)*J_prev
    
    # Coefficients for the tridiagonal matrix
    a = np.zeros(Nx + 1)
    b = np.zeros(Nx + 2)
    c = np.zeros(Nx + 1)
    d = np.zeros(Nx + 2)

    for i in range(0, Nx):
        a[i] = -1 / (3 * Sigma_t_mod_edge[i] * dx_edge[i])
        b[i + 1] = (
            1 / (3 * Sigma_t_mod_edge[i] * dx_edge[i])
            + 1 / (3 * Sigma_t_mod_edge[i + 1] * dx_edge[i + 1])
            + (Sigma_t_mod[i] - Sigma_s[i] - nu * Sigma_f[i]) * dx[i]
        )
        c[i + 1] = -1 / (3 * Sigma_t_mod_edge[i + 1] * dx_edge[i + 1])
        d[i + 1] = (
            q0[i] * dx[i]
            - (F[i + 2] - F[i + 1]) / (Sigma_t_mod_edge[i + 1] * dx_edge[i + 1])
            + (F[i + 1] - F[i]) / (Sigma_t_mod_edge[i] * dx_edge[i])
            + q1[i] / Sigma_t_mod_edge[i]
            - q1[i + 1] / Sigma_t_mod_edge[i + 1]
        )

    # Boundary conditions

    # Left
    # Vacuum
    if left_bc == 0:
        b[0] = 1 / (3 * Sigma_t_mod_edge[0] * dx_edge[0]) + 0.5
        c[0] = -1 / (3 * Sigma_t_mod_edge[0] * dx_edge[0])
        d[0] = (
            Pl
            - (F[1] - F[0]) / (Sigma_t_mod_edge[0] * dx_edge[0])
            - q1[0] / Sigma_t_mod_edge[0]
        )
    # Reflective
    elif left_bc == 1:
        b[0] = (
            1 / (3 * Sigma_t_mod_edge[0] * dx[0])
            + (Sigma_t_mod[0] - Sigma_s[0] - nu * Sigma_f[0]) * dx[0]
        )
        c[0] = -1 / (3 * Sigma_t_mod_edge[0] * dx[0])
        d[0] = (
            q0[0] * dx[0]
            + (F[1] - F[0]) / (Sigma_t_mod_edge[0] * dx_edge[0])
            + q1[0] / Sigma_t_mod_edge[0]
        )

    # Right
    if right_bc == 0:
        a[-1] = 1 / (3 * Sigma_t_mod_edge[-1] * dx_edge[-1])
        b[-1] = -1 / (3 * Sigma_t_mod_edge[-1] * dx_edge[-1]) - 0.5
        d[-1] = (
            -Pr
            - (F[-1] - F[-2]) / (Sigma_t_mod_edge[-1] * dx_edge[-1])
            - q1[-1] / Sigma_t_mod_edge[-1]
        )

    elif right_bc == 1:
        a[-1] = -1 / (3 * Sigma_t_mod[-1] * dx[-1])
        b[-1] = (
            1 / (3 * Sigma_t_mod[-1] * dx[-1])
            - (Sigma_t_mod[-1] - Sigma_s[-1] - nu * Sigma_f[-1]) * dx[-1]
        )
        d[-1] = (
            q0[-1] * dx[-1]
            + (F[-1] - F[-2]) / (Sigma_t_mod[-1] * dx[-1])
            + q1[-1] / Sigma_t_mod[-1]
        )
    return a,b,c,d

def restore_current(problem,old_state,closure,dt,phi):

    q = problem["source"]
    x_mesh = problem["x_mesh"]
    dx = x_mesh[1:] - x_mesh[:-1]
    Nx = len(x_mesh) - 1
    v = problem["v"]
    Sigma_a = problem["siga"]
    Sigma_f = problem["sigf"]
    Sigma_t = problem["sigt"]
    Sigma_s = problem["sigs"]
    nu = problem["nu"]
    time_scheme = problem["time_scheme"]
    # getting variables from problem and state
    phi_prev = np.copy(old_state[0])
    J_prev = np.copy(old_state[1])

    F_prev = closure["F"]

    F = closure["F"]

    # hardwired boundary conditions (bad)
    left_bc = 0#problem.source.lb
    right_bc = 0#problem.source.rb

    # Spatial edge calculations
    dx_edge = np.zeros(Nx + 1)
    dx_edge[1:-1] = (dx[:-1] + dx[1:]) / 2
    dx_edge[0] = dx[0] / 2
    dx_edge[-1] = dx[-1] / 2

    # Compute edge cross section
    Sigma_t_edge = np.zeros(Nx + 1)
    Sigma_t_edge[1:-1] = (Sigma_t[:-1] * dx[:-1] + Sigma_t[1:] * dx[1:]) / (
        dx[:-1] + dx[1:]
    )
    Sigma_t_edge[0] = Sigma_t[0]
    Sigma_t_edge[-1] = Sigma_t[-1]


    # Compute modified Sigma_t based on the method
    if time_scheme == HYBRID_BE:
        Sigma_t_mod = Sigma_t + 1 / (v * dt)
    elif time_scheme == HYBRID_CN:
        Sigma_t_mod = Sigma_t + 2 / (v * dt)

    Sigma_t_mod_edge = np.zeros(Nx + 1)
    Sigma_t_mod_edge[1:-1] = (Sigma_t_mod[:-1] * dx[:-1] + Sigma_t_mod[1:] * dx[1:]) / (
        dx[:-1] + dx[1:]
    )
    Sigma_t_mod_edge[0] = Sigma_t_mod[0]
    Sigma_t_mod_edge[-1] = Sigma_t_mod[-1]

    # Compute Q based on the method
    if time_scheme == HYBRID_BE:
        q0 = q + phi_prev[1:-1] / (v * dt)
        q1 = J_prev / (v * dt)
    elif time_scheme == HYBRID_CN:
        q0 = (J_prev[:-1] - J_prev[1:])/dx + q + q + (Sigma_s + nu*Sigma_f+2/(v*dt)-Sigma_t)*phi_prev[1:-1]
        q1 = (F_prev[1:] - F_prev[:-1]+(phi_prev[:-1]-phi_prev[1:])/3)/dx_edge + (2/(v*dt)-Sigma_t_edge)*J_prev

    # Restore current using the updated scalar flux
    J = (
        (phi[:-1] - phi[1:]) / (3 * Sigma_t_mod_edge * dx_edge)
        + q1 / Sigma_t_mod_edge
        + (F[1:] - F[:-1]) / (Sigma_t_mod_edge * dx_edge)
    )

    #RESIDUAL CALCULATION (SHOULD ADD)
    '''
    if time_scheme == HYBRID_BE:
        res_balance = (
            dx / (v * dt) * (phi[1:-1]-phi_prev[1:-1])
            + J[1:]
            - J[:-1]
            + (Sigma_t - Sigma_s - nu * Sigma_f) * dx * phi[1:-1]
            - q * dx
        )
        
        res_sm = (
            dx_edge / (v * dt) * (J-J_prev)
            + (phi[1:] - phi[:-1]) / 3
            + Sigma_t_edge * dx_edge * J
            + F[:-1]
            - F[1:]
        )

    elif time_scheme == HYBRID_CN:
        res_balance = (
            dx / (v * dt) * (phi[1:-1]-phi_prev[1:-1])
            + 0.5
            * (
                J[1:]
                - J[:-1]
                + (Sigma_t - Sigma_s - nu * Sigma_f) * dx * phi[1:-1]
                - q * dx
            )
            + 0.5
            * (
                J_prev[1:]
                - J_prev[:-1]
                + (Sigma_t - Sigma_s - nu * Sigma_f) * dx * phi_prev[1:-1]
                - q * dx
            )
        )
        res_sm = (
            dx_edge / (v * dt) * (J-J_prev)
            + 0.5
            * ((phi[1:] - phi[:-1]) / 3 + Sigma_t_edge * dx_edge * J + F[:-1] - F[1:])
            + 0.5
            * (
                (phi_prev[1:]-phi_prev[:-1]) / 3
                + Sigma_t_edge * dx_edge * J_prev
                + F_prev[:-1]
                - F_prev[1:]
            )
        )

        if left_bc == 0:
            res_lb = J[0] + 0.5 * phi[0] - Pl
        elif left_bc == 1:
            res_lb = J[0]
        if right_bc == 0:
            res_rb = J[-1] - 0.5 * phi[-1] + Pr
        elif right_bc == 1:
            res_rb = J[-1]

        residual = [res_balance, res_sm, res_lb, res_rb]
        '''
    return J

def losm_FE_step_time(problem,old_state,closure,dt,sparse=True): # finite element
    """
    Performs one timestep of the linear discontinous low-order second moment method

    """
    # Create A matrix and b vector
    A, q = make_FE_matrix(problem,old_state,closure,dt,sparse)

    # Solve for x
    if sparse:
        new_state = sp.sparse.linalg.spsolve(A,q)
    else:
        new_state = np.linalg.solve(A,q)

    # reconstruct from solution vector
    new_flux, new_current = reconstruct_from_state(new_state)

    #return new state (x)
    return new_flux, new_current

def losm_FV_step_time(problem, old_state,closure,dt): # finite volume
    """
    Perform one timestep of the LOSM method using TDMA.

    Parameters:
    current_state (State): Current state of the system containing flux, current, F, Pl, and Pr.
    previous_state (State): Previous state of the system containing flux and current.
    problem (Problem): Problem parameters containing cross-sections, source term, mesh, and quadrature.
    method (str): Method for time-stepping: "BE" (Backward Euler) or "CN" (Crank-Nicolson). Default is "BE".

    Returns:
    State: Updated state with new scalar flux and current.
    """
    # Solve the tridiagonal system using TDMA (Thomas algorithm)
    a,b,c,d = make_FV_matrix(problem,old_state,closure,dt)
    phi = tdma(a, b, c, d)

    J = restore_current(problem,old_state,closure,dt,phi)


    return phi, J



## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def tdma(a, b, c, d):
    """
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    """
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def reconstruct_from_state(state):
    Nx = int(len(state)/8)
    phi =  np.zeros((4,Nx))
    J =  np.zeros((4,Nx))
    
    for i in range(Nx):
        for ix in range(4):
            phi[ix,i]=state[i*8+ix]
            J[ix,i]=state[i*8+4+ix]
    return phi,J