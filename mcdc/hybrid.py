import numpy as np

import mcdc.type_ as type_
import mcdc.adapt as adapt
import kernel
def get_material(x,y,z,t,mcdc):
    P: type_.particle_record #= adapt.local_particle_record()
    # Make and return particle
    P["x"] = x
    P["y"] = y
    P["z"] = z
    P["t"] = t
    P["ux"] = 0
    P["uy"] = 0
    P["uz"] = 0
    P["g"] = g
    P["E"] = E
    P["w"] = 1.0

    P["sensitivity_ID"] = 0
    mat = kernel.get_particle_material(P,mcdc)
    return mat 

mcdc['tally']['mesh']

for i in range(Nx):
for j in range(Ny):
    for k in range(Nz):
        x_center = x[i]
        y_center = y[j]
        z_center = z[k]
        material_grid[i, j, k] = get_material_at_point(x_center, y_center, z_center)

