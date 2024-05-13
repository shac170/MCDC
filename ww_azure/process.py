import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py


# Get results
with h5py.File("ref.h5", "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    t = f["tally/grid/t"][:]
    dt = t[1:] - t[:-1]
    K = len(t) - 1

    phi = f["tally/flux/mean"][:]
    phi_sd = f["tally/flux/sdev"][:]

    total = f["tally/total/mean"][:]
    total_sd = f["tally/total/sdev"][:]

    # Normalize
    for k in range(K):
        phi[k] /= dx * dt[k]
        phi_sd[k] /= dx * dt[k]

with h5py.File("output.h5", "r") as f:
    x = f["tally/grid/x"][:]
    dx = x[1:] - x[:-1]
    x_mid = 0.5 * (x[:-1] + x[1:])
    t = f["tally/grid/t"][:]
    dt = t[1:] - t[:-1]
    K = len(t) - 1

    ww_phi = f["tally/flux/mean"][:]
    ww_phi_sd = f["tally/flux/sdev"][:]

    total = f["tally/total/mean"][:]
    total_sd = f["tally/total/sdev"][:]

    # Normalize
    for k in range(K):
        ww_phi[k] /= dx * dt[k]
        ww_phi_sd[k] /= dx * dt[k]

# Flux - average
fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1 = plt.axes()#xlim=(-21.889999999999997, 21.89), ylim=(-0.042992644459595206, 0.9028455336514992))
ax1.grid()
ax1.set_xlabel(r"$x$")
ax1.set_ylabel(r"Flux")
ax1.set_title(r"$\bar{\phi}_{k,j}$")
ax1.set_yscale('log')
line1, = ax1.plot([], [], "-b", label="ref")
line2, = ax1.plot([], [], "-r", label="ww")

fb1 = ax1.fill_between([], [], [], alpha=0.2, color="b")
text1 = ax1.text(0.02, 0.9, "", transform=ax1.transAxes)
ax1.legend()

# Flux standard deviation
ax2.set_xlim(-21.889999999999997, 21.89)
ax2.set_ylim(0.01, 2.0)
ax2.grid()
ax2.set_xlabel(r"$x$")
ax2.set_ylabel(r"Flux standard deviation")
ax2.set_title(r"$\sigma_{\phi_{k,j}}$")
ax2.set_yscale('log')
line3, = ax2.plot([], [], "-b", label="ref")
line4, = ax2.plot([], [], "-r", label="ww")

fb2 = ax1.fill_between([], [], [], alpha=0.2, color="b")
text2 = ax2.text(0.02, 0.9, "", transform=ax2.transAxes)
ax2.legend()

def animate(k):
    global fb1, fb2
    fb1.remove()
    fb2.remove()
    line1.set_data(x_mid, phi[k, :])
    line2.set_data(x_mid, ww_phi[k, :])
    fb1 = ax1.fill_between(
        x_mid, phi[k, :] - phi_sd[k, :], phi[k, :] + phi_sd[k, :], alpha=0.2, color="b"
    )
    text1.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[k], t[k + 1]))
    
    line3.set_data(x_mid, phi_sd[k, :]/phi[k,:])
    line4.set_data(x_mid, ww_phi_sd[k, :]/ww_phi[k,:])
    fb2 = ax1.fill_between(
        x_mid, phi[k, :] - phi_sd[k, :], phi[k, :] + phi_sd[k, :], alpha=0.2, color="b"
    )
    text2.set_text(r"$t \in [%.1f,%.1f]$ s" % (t[k], t[k + 1]))
    
    return line1, line2, line3, line4, text1, text2


simulation = animation.FuncAnimation(fig, animate, frames=K)
simulation.save(
    "azurv1.gif",
    fps=4,
    writer="imagemagick",
    savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0},
)
plt.show()