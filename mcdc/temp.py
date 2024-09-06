def ww_update(data, mcdc, update_index):
    idx_n0 = mcdc["idx_census"]
    idx_n1 = idx_n0 - 1
    idx_n2 = idx_n1 - 1

    dt = abs(
        mcdc["technique"]["ww"]["mesh"]["t"][idx_n0]
        - mcdc["technique"]["ww"]["mesh"]["t"][idx_n1]
    )
    dx = abs(
        mcdc["technique"]["ww"]["mesh"]["z"][1:]
        - mcdc["technique"]["ww"]["mesh"]["z"][:-1]
    )
    N_particle = mcdc["setting"]["N_particle"]
    flux = get_tally(idx_n1, mcdc, data, 'flux')
    flux *= mcdc["technique"]["integrated_source"] / (dx * dt * N_particle)

    width = mcdc["technique"]["ww"]["width"]
    epsilon = mcdc["technique"]["ww"]["epsilon"]
    method = mcdc["technique"]["ww"]["auto"]
    save = mcdc["technique"]["ww"]["save"]

    if method == WW_USER:
        return

    if method == WW_PREVIOUS:
        ww_previous_method(mcdc, idx_n0, flux, save, update_index)

    elif method == WW_ALPHA:
        ww_alpha_method(mcdc, data, idx_n0, idx_n2, flux, dt, dx, N_particle, epsilon, save, update_index)

    elif method == WW_LEAKAGE:
        ww_leakage_method(mcdc, data, idx_n0, idx_n1, flux, dt, dx, epsilon, save, update_index)

    elif method == WW_HYBRID:
        ww_hybrid_method(mcdc, data, idx_n0, idx_n1, flux, dt, epsilon, save, update_index)

    apply_ww_modifications(mcdc, idx_n0, epsilon, update_index)


def ww_previous_method(mcdc, idx_n0, flux, save, update_index):
    if idx_n0 < 1:
        return
    mcdc["technique"]["ww"]["center"][idx_n0, update_index, 0, 0, :] = flux / np.max(flux)
    if save:
        mcdc["technique"]["ww"]["phi_previous"][idx_n0, update_index] = flux


def ww_alpha_method(mcdc, data, idx_n0, idx_n2, flux, dt, dx, N_particle, epsilon, save, update_index):
    if idx_n0 < 2:
        return
    old_flux = get_tally(idx_n2, mcdc, data, 'flux')
    old_flux *= mcdc["technique"]["integrated_source"] / (dx * dt * N_particle)

    alpha = np.ones_like(flux)
    mask = old_flux != 0
    alpha[mask] = np.log(flux[mask] / old_flux[mask])

    if epsilon[WW_LIMIT_LEAKAGE] != 0:
        print_error("LEAKAGE LIMITING NOT AVAILABLE YET")

    if epsilon[WW_LIMIT_GAMMA] != 0:
        gamma = compute_gamma(mcdc, idx_n0, update_index)
        alpha[alpha > gamma] = gamma[alpha > gamma]
        if save:
            mcdc["technique"]["ww"]["gamma"][idx_n0, update_index, 0, 0, :] = gamma

    new_flux = flux * np.exp(alpha * dt)
    mcdc["technique"]["ww"]["center"][idx_n0, update_index, 0, 0, :] = new_flux / np.max(new_flux)
    
    if save:
        mcdc["technique"]["ww"]["phi_tilde"][idx_n0, update_index, 0, 0, :] = new_flux
        mcdc["technique"]["ww"]["phi_previous"][idx_n0, update_index] = flux
        mcdc["technique"]["ww"]["phi_old"][idx_n0, update_index] = old_flux
        mcdc["technique"]["ww"]["alpha"][idx_n0, update_index] = alpha


def ww_hybrid_method(mcdc, data, idx_n0, idx_n1, flux, dt, epsilon, save, update_index):
    if epsilon[WW_TIME_INTERP] != 0 and idx_n0 < 2:
        return

    old_state, problem = get_state(idx_n1, mcdc, data)
    new_state = losm_timestep(old_state, old_state, problem)
    new_flux = new_state.flux[1:-1]

    mcdc["technique"]["deterministic"]["flux"][idx_n0, update_index, 0, 0, :, 0] = new_state.flux
    mcdc["technique"]["deterministic"]["current"][idx_n0, update_index, 0, 0, :, 0] = new_state.current
    mcdc["technique"]["deterministic"]["ic_flux"][idx_n0, update_index, 0, 0, :, 0] = old_state.flux
    mcdc["technique"]["deterministic"]["ic_current"][idx_n0, update_index, 0, 0, :, 0] = old_state.current
    mcdc["technique"]["deterministic"]["sm_factor"][idx_n0, update_index, 0, 0, :, 0] = old_state.F

    mcdc["technique"]["ww"]["center"][idx_n0, update_index, 0, 0, :] = new_flux / np.max(new_flux)

    if save:
        mcdc["technique"]["ww"]["phi_tilde"][idx_n0, update_index, 0, 0, :] = new_flux


def ww_leakage_method(mcdc, data, idx_n0, idx_n1, flux, dt, dx, epsilon, save, update_index):
    gamma = compute_gamma(mcdc, idx_n0, update_index)
    current = get_tally(idx_n1, mcdc, data, 'current')
    Q_tally = compute_Q_tally(mcdc, idx_n0, current, dx, gamma, flux, dt)

    mask = gamma != 0
    new_flux = np.copy(flux)
    new_flux[mask] = flux[mask] * np.exp(-gamma[mask] * dt) + Q_tally[mask] / gamma[mask] * (1 - np.exp(-gamma[mask] * dt))
    mcdc["technique"]["ww"]["center"][idx_n0, update_index, 0, 0, :] = new_flux / np.max(new_flux)

    if save:
        mcdc["technique"]["ww"]["phi_tilde"][idx_n0, update_index, 0, 0, :] = new_flux
        mcdc["technique"]["ww"]["phi_previous"][idx_n0, update_index] = flux
        mcdc["technique"]["ww"]["current"][idx_n0, update_index] = current
        mcdc["technique"]["ww"]["gamma"][idx_n0, update_index, 0, 0, :] = gamma
        mcdc["technique"]["ww"]["Q"][idx_n0, update_index, 0, 0, :] = Q_tally


def compute_Q_tally(mcdc, idx_n0, current, dx, gamma, flux, dt):
    det = mcdc["technique"]["deterministic"]
    source = np.squeeze(det["source"])[idx_n0, :]
    speed = np.zeros(len(np.squeeze(det["material_idx"])[idx_n0, :]))
    for i in range(len(speed)):
        mat_idx = np.squeeze(det["material_idx"])[idx_n0, :][i]
        speed[i] = mcdc["materials"][mat_idx]["speed"]
    return speed * (source - (current[1:] - current[:-1]) / dx)


def compute_gamma(mcdc, idx_n0, update_index):
    det = mcdc["technique"]["deterministic"]
    Sigma_c, Sigma_f, speed, gamma = [np.zeros(len(np.squeeze(det["material_idx"])[idx_n0, :])) for _ in range(4)]
    materials = mcdc["materials"]

    for i in range(len(Sigma_c)):
        mat_idx = np.squeeze(det["material_idx"])[idx_n0, :][i]
        Sigma_c[i] = materials[mat_idx]["capture"][0] + materials[mat_idx]["scatter"][0]
        Sigma_f[i] = materials[mat_idx]["fission"][0]
        speed[i] = materials[mat_idx]["speed"]
        gamma[i] = speed[i] * (Sigma_c[i] + Sigma_f[i] - materials[mat_idx]["nu_f"] * Sigma_f[i])

    return gamma


def apply_ww_modifications(mcdc, idx_n0, epsilon, update_index):
    center = mcdc["technique"]["ww"]["center"][idx_n0, update_index, 0, 0, :]
    if epsilon[WW_MIN] > 0:
        center = center * (1 - epsilon[WW_MIN]) + epsilon[WW_MIN]

    if epsilon[WW_WOLLABER] > 0:
        w_min = epsilon[WW_WOLLABER + 1]
        center = (center) * (1 + (1 / epsilon[WW_WOLLABER] - 1) * np.exp(-(center - w_min) / epsilon[WW_WOLLABER]))

    mcdc["technique"]["ww"]["center"][idx_n0, update_index, 0, 0, :] = center
