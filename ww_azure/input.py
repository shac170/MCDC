import numpy as np

import mcdc

# =============================================================================
# Set model
# =============================================================================
# Infinite medium with isotropic plane surface at the center
# Based on Ganapol LA-UR-01-1854 (AZURV1 benchmark)
# Effective scattering ratio c = 1.1

# Set materials
m = mcdc.material(
    capture=np.array([1.0 / 3.0]),
    scatter=np.array([[1.0 / 3.0]]),
    fission=np.array([1.0 / 3.0]),
    nu_p=np.array([2.3]),
)

# Set surfaces
s1 = mcdc.surface("plane-x", x=-1e10, bc="reflective")
s2 = mcdc.surface("plane-x", x=1e10, bc="reflective")

# Set cells
mcdc.cell([+s1, -s2], m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic pulse at x=t=0

mcdc.source(point=[0.0, 0.0, 0.0], isotropic=True, time=[1e-10, 1e-10])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

# Tally: cell-average, cell-edge, and time-edge scalar fluxes
mcdc.tally(
    scores=["flux","total"],
    x=np.linspace(-20.5, 20.5, 202),
    t=np.linspace(0.0, 20.0, 21),
)
# Setting
mcdc.setting(N_particle=1e4)
mcdc.setting(active_bank_buff=1e20, census_bank_buff=1e3)
mcdc.implicit_capture()
#mcdc.population_control(population_control=False)
mcdc.population_control(pct="combing")  #combing or combing-weight
mcdc.time_census(np.linspace(0.0, 20.0, 41))


use_windows = 1

if (use_windows == 1):
  import h5py
  filename = "ref.h5"
  f1 = h5py.File(filename,"r")
  time_bins_ww = np.array(f1["tally"]["grid"]["t"])
  x_bins_ww = np.array(f1["tally"]["grid"]["x"])
  flux_ww = np.array(f1["tally"]["flux"]["mean"])
  min_value_flux = np.min(flux_ww[flux_ww!=0])
  flux_ww[flux_ww==0] = min_value_flux/2
  windows_cl = flux_ww*1
  windows_cl /= np.max(windows_cl)
  print(np.max(windows_cl),np.min(windows_cl))
  mcdc.weight_window(
      width = 2.5,
      x = np.linspace(-20.5, 20.5, 202),
      t = np.linspace(0.0, 20.0, 21),
      window = windows_cl
  )
elif (use_windows == 2):
  mcdc.weight_window()


# Run
mcdc.run()
