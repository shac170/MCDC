import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py


# Get results
with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    t = f["tally/grid/t"][:]
    dt = t[1:] - t[:-1]
    K = len(t) - 1

    ww_phi = f["tally/flux/mean"][:]
    ww_phi_sd = f["tally/flux/sdev"][:]

    # Normalize
    for k in range(K):
        ww_phi[k] /= dx * dt[k]
        ww_phi_sd[k] /= dx * dt[k]

with h5py.File("ref.h5", "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    t = f["tally/grid/t"][:]
    dt = t[1:] - t[:-1]
    K = len(t) - 1

    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]

    # Normalize
    for k in range(K):
        phi[k] /= dx * dt[k]
        phi_sd[k] /= dx * dt[k]

stdev_ww = []
stdev = []
for i in range(len(t)-1):
    stdev.append(np.linalg.norm(phi_sd[i,:][phi[i,:]!=0]/(phi[i,:][phi[i,:]!=0])))
    stdev_ww.append(np.linalg.norm(ww_phi_sd[i,:][ww_phi[i,:]!=0]/(ww_phi[i,:][ww_phi[i,:]!=0])))
stdev = np.array(stdev)
stdev_ww = np.array(stdev_ww)

plt.clf()
plt.plot(t[1:],stdev,marker='x',label='analog')
plt.plot(t[1:],stdev_ww,marker='o',label='ww')
plt.yscale('log')
plt.legend()
plt.show()