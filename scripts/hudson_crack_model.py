#!/usr/bin/env python3
"""
hudson_crack_model.py

Hudson (1980/1981) effective medium model for randomly oriented fluid-saturated
penny-shaped cracks in an isotropic background (ν = 0.25 fixed).

Physics
-------
Cracks are initially randomly and uniformly distributed on the unit sphere.
In a stress field, cracks close when their normal-stress exceeds the closure
pressure σ_c.  The surviving (open) crack population is orientation-biased,
producing velocity anisotropy even though no single preferred crack direction
was prescribed.

Sign convention: tensile stress is POSITIVE (standard elasticity).
Crack is OPEN when  σ_n(n̂) = n̂ᵀ σ n̂  >  −σ_c
  (i.e. not more compressive than the closure threshold σ_c).

Hudson (1980) first-order stiffness correction (crack frame, n̂ = ê₃):
  ΔC₃₃₃₃ = −ε (λ+2μ)²/μ  U₁      [P-wave normal to cracks]
  ΔC₁₁₃₃ = ΔC₂₂₃₃ = −ε λ(λ+2μ)/μ U₁
  ΔC₁₁₁₁ = ΔC₂₂₂₂ = ΔC₁₁₂₂ = −ε λ²/μ  U₁
  ΔC₁₃₁₃ = ΔC₂₃₂₃ = −ε μ U₃          [S-wave shear across cracks]
  ΔC₁₂₁₂ = 0                            [in-plane shear unaffected]

where for ν = 0.25 (λ = μ):
  U₁_dry  = 4(1−ν)/(1−2ν) = 6.0           (dimensionless)
  U₃       = (16/3)(1−ν)/(2−ν) ≈ 2.286     (dimensionless)
  U₁_wet  = U₁_dry / (1 + K_f / (π r M))   M = λ+2μ

For general n̂ the correction is obtained by rotating the crack-frame tensor
via the 3D rotation R that maps ê₃ → n̂.
"""

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
NU       = 0.25       # Poisson's ratio (fixed throughout)
RHO      = 2750.0     # kg/m³  crustal density
K_WATER  = 2.2e9      # Pa  bulk modulus of water

# ── Fibonacci lattice on unit sphere ─────────────────────────────────────────

def fibonacci_sphere(N=300):
    """N approximately uniformly distributed unit vectors on the sphere."""
    golden = (1 + np.sqrt(5)) / 2
    i      = np.arange(N)
    theta  = np.arccos(1 - 2*(i + 0.5)/N)
    phi    = 2*np.pi * i / golden % (2*np.pi)
    return np.column_stack([np.sin(theta)*np.cos(phi),
                             np.sin(theta)*np.sin(phi),
                             np.cos(theta)])

_N_HATS = fibonacci_sphere(300)   # default, reused for efficiency

# ── Background isotropic stiffness (6×6 Voigt) ───────────────────────────────

def background_stiffness(mu, nu=NU):
    """6×6 Voigt stiffness for isotropic background."""
    lam = 2*nu*mu / (1 - 2*nu)    # = mu for nu=0.25
    C = np.zeros((6, 6))
    C[0,0]=C[1,1]=C[2,2] = lam + 2*mu
    C[0,1]=C[0,2]=C[1,2] = lam
    C[1,0]=C[2,0]=C[2,1] = lam
    C[3,3]=C[4,4]=C[5,5] = mu
    return C

# ── Rotation matrix mapping ê₃ → n̂ (via Rodrigues) ─────────────────────────

def _rotation_e3_to_n(n_hat):
    """
    3×3 rotation matrix R such that  R @ [0,0,1] = n_hat.
    Uses Rodrigues' rotation formula.  Handles the degenerate case n ≈ ±ê₃.
    """
    n = n_hat / np.linalg.norm(n_hat)
    e3 = np.array([0., 0., 1.])
    if np.abs(n[2] - 1.) < 1e-8:      # n ≈ +ê₃: identity
        return np.eye(3)
    if np.abs(n[2] + 1.) < 1e-8:      # n ≈ −ê₃: 180° about ê₁
        return np.diag([1., -1., -1.])
    axis  = np.cross(e3, n)
    sin_t = np.linalg.norm(axis)
    cos_t = float(np.dot(e3, n))
    axis /= sin_t
    K = np.array([[0,       -axis[2],  axis[1]],
                  [ axis[2], 0,       -axis[0]],
                  [-axis[1], axis[0],  0      ]])
    return np.eye(3) + sin_t*K + (1 - cos_t)*(K @ K)

# ── Hudson (1980) crack compliance coefficients ───────────────────────────────

def _compliance_coeffs(mu, nu=NU, K_f=K_WATER, r_crack=0.01):
    """
    Hudson (1980) dimensionless crack compliance coefficients.

    U1 : normal compliance (reduced by fluid incompressibility)
    U3 : shear compliance (same for wet and dry cracks)

    For ν=0.25:
        U1_dry = 4(1−ν)/(1−2ν) = 6.0
        U3     = (16/3)(1−ν)/(2−ν) ≈ 2.286
    """
    U1_dry = 4*(1 - nu) / (1 - 2*nu)
    U3     = (16/3)*(1 - nu) / (2 - nu)
    lam    = 2*nu*mu / (1 - 2*nu)
    M      = lam + 2*mu                     # P-wave modulus
    # Fluid correction: crack normal compliance is reduced for saturated cracks
    # Hudson (1980): U1_wet = U1_dry / (1 + K_f / (π r M))
    U1_wet = U1_dry / (1.0 + K_f / (np.pi * r_crack * M))
    return U1_wet, U3

# ── 4th-rank Hudson correction for one crack set ─────────────────────────────

def _hudson_4d(n_hat, epsilon, mu, nu=NU, K_f=K_WATER, r_crack=0.01):
    """
    Hudson (1980) first-order 3×3×3×3 stiffness correction for a single set
    of parallel penny-shaped cracks with unit normal n_hat and crack density ε.

    Strategy:
      1. Compute correction in the crack frame (n̂ = ê₃) using the explicit
         formulas from Hudson (1980).
      2. Rotate the 4th-rank tensor back to the global frame using
         R such that  R ê₃ = n̂.
    """
    lam   = 2*nu*mu / (1 - 2*nu)
    M     = lam + 2*mu
    U1, U3 = _compliance_coeffs(mu, nu, K_f, r_crack)

    # Correction in crack frame (ê₃ = crack normal)
    # Only these Voigt components are non-zero:
    #   (3,3), (1,3),(2,3),(3,1),(3,2), (1,1),(2,2),(1,2),(2,1) from U1 term
    #   (4,4),(5,5) = C₂₃₂₃, C₁₃₁₃ from U3 term
    DC_frame = np.zeros((3, 3, 3, 3))

    # ── U₁ block (normal compliance) ─────────────────────────────────────────
    # Written out for clarity; ê₃ = (0,0,1) in crack frame
    c_nn  = -(M**2 / mu) * epsilon * U1   # C₃₃₃₃ coeff
    c_tn  = -(lam * M  / mu) * epsilon * U1  # C₁₁₃₃, C₂₂₃₃ coeff
    c_tt  = -(lam**2   / mu) * epsilon * U1  # C₁₁₁₁, C₂₂₂₂, C₁₁₂₂ coeff

    # Build from known crack-frame components
    # (i,j,k,l) in crack frame; 0=x,1=y,2=z
    DC_frame[2,2,2,2] = c_nn
    for ij in [(0,0),(1,1)]:
        DC_frame[ij[0],ij[1],2,2] = c_tn
        DC_frame[2,2,ij[0],ij[1]] = c_tn
    for ij in [(0,0),(1,1)]:
        for kl in [(0,0),(1,1)]:
            DC_frame[ij[0],ij[1],kl[0],kl[1]] += c_tt
    DC_frame[0,0,1,1] += c_tt   # C₁₁₂₂
    DC_frame[1,1,0,0] += c_tt

    # ── U₃ block (shear compliance) ───────────────────────────────────────────
    # C₁₃₁₃ = C₂₃₂₃ = −ε μ U₃   (C₁₂₁₂ = 0)
    # Must set ALL 4 symmetric equivalents for each independent shear pair
    # so that the subsequent symmetrisation averages to the correct value.
    c_sh = -mu * epsilon * U3
    for (a, b) in [(0, 2), (1, 2)]:          # two independent shear pairs
        for ia, ib, ic, id in [(a,b,a,b), (b,a,a,b), (a,b,b,a), (b,a,b,a)]:
            DC_frame[ia,ib,ic,id] += c_sh

    # Full major + minor symmetry: C_ijkl = C_jikl = C_ijlk = C_jilk
    DC_frame = 0.25*(DC_frame +
                     DC_frame.transpose(1,0,2,3) +
                     DC_frame.transpose(0,1,3,2) +
                     DC_frame.transpose(1,0,3,2))

    # ── Rotate crack-frame correction to global frame ─────────────────────────
    R  = _rotation_e3_to_n(n_hat)
    DC = np.einsum('ia,jb,kc,ld,abcd->ijkl', R, R, R, R, DC_frame)
    return DC

# ── Convert 3×3×3×3 ↔ 6×6 Voigt ─────────────────────────────────────────────

_VPAIRS = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]

def _to_voigt(C4):
    C6 = np.zeros((6,6))
    for a,(i,j) in enumerate(_VPAIRS):
        for b,(k,l) in enumerate(_VPAIRS):
            C6[a,b] = C4[i,j,k,l]
    return C6

def _from_voigt(C6):
    C4 = np.zeros((3,3,3,3))
    for a,(i,j) in enumerate(_VPAIRS):
        for b,(k,l) in enumerate(_VPAIRS):
            v = C6[a,b]
            C4[i,j,k,l]=C4[j,i,k,l]=C4[i,j,l,k]=C4[j,i,l,k] = v
    return C4

# ── Effective stiffness for a stress field ────────────────────────────────────

def effective_stiffness(sigma, mu,
                        epsilon0=0.05, K_f=K_WATER,
                        r_crack=0.01, sigma_c=10e6,
                        n_hats=_N_HATS, nu=NU,
                        closure_width_frac=0.10):
    """
    Hudson effective stiffness 6×6 Voigt [Pa] at a point with stress σ.

    Parameters
    ----------
    sigma              : (3,3) stress tensor [Pa]; tensile positive
    mu                 : background shear modulus [Pa]
    epsilon0           : total isotropic crack density (MCMC param)
    K_f                : fluid bulk modulus [Pa]
    r_crack            : crack aspect ratio c/a  (MCMC param)
    sigma_c            : crack closure pressure [Pa]  (MCMC param)
                         crack is OPEN when σ_n(n̂) > −σ_c
    n_hats             : (N,3) crack normal sample (Fibonacci sphere)
    closure_width_frac : sigmoid transition width as fraction of σ_c.
                         Removes the hard step-function plateau that makes
                         the likelihood flat in pressure beyond the threshold.
                         κ = closure_width_frac × σ_c; κ→0 → hard threshold.

    Returns
    -------
    C_eff      : (6,6) Voigt effective stiffness [Pa]
    open_frac  : effective mean open fraction (weighted by sigmoid)
    """
    N = len(n_hats)
    eps_per = epsilon0 / N

    # Vectorised normal stress for all crack normals
    sigma_n = np.einsum('ki,ij,kj->k', n_hats, sigma, n_hats)

    # Soft sigmoid closure: weight each orientation continuously.
    # w=1 when crack is well open, w=0 when well clamped.
    # This keeps dφ/dP ≠ 0 everywhere and resolves the saturation plateau.
    kappa = closure_width_frac * abs(sigma_c) + 1e3   # floor 1 kPa for stability
    weights = 1.0 / (1.0 + np.exp(-(sigma_n + abs(sigma_c)) / kappa))
    open_frac = float(weights.mean())

    C0 = background_stiffness(mu, nu)
    DC4 = np.zeros((3,3,3,3))
    for k in range(N):
        if weights[k] > 1e-6:   # skip negligible contributions
            DC4 += weights[k] * _hudson_4d(n_hats[k], eps_per, mu, nu, K_f, r_crack)

    return C0 + _to_voigt(DC4), open_frac

# ── Christoffel equation → S-wave splitting ───────────────────────────────────

def christoffel_splitting(C_voigt, p_hat, rho=RHO):
    """
    Phase velocities and fast-axis azimuth for ray direction p̂.

    Returns
    -------
    phi_fast : fast-S azimuth [° from N, 0–180]
    delta_V  : VS1 − VS2  [m/s]
    VS1, VS2 : fast and slow S-wave speeds [m/s]
    """
    C4    = _from_voigt(C_voigt)
    p     = p_hat / np.linalg.norm(p_hat)
    Gamma = np.einsum('ijkl,j,l->ik', C4, p, p)
    vals, vecs = np.linalg.eigh(Gamma)        # ascending
    idx   = np.argsort(vals)[::-1]            # descending
    vals  = np.maximum(vals[idx], 0.) / rho   # V²
    vecs  = vecs[:, idx]

    VS1, VS2 = np.sqrt(vals[1]), np.sqrt(vals[2])
    pol       = vecs[:, 1]                     # fast-S polarisation
    pol      /= np.linalg.norm(pol) + 1e-30
    phi_fast  = float(np.degrees(np.arctan2(pol[0], pol[1])) % 180.)
    return phi_fast, float(VS1 - VS2), float(VS1), float(VS2)

# ── Ray integration ───────────────────────────────────────────────────────────

def integrate_splitting(sigma_stack, ds_array, p_hat, mu,
                        epsilon0=0.05, K_f=K_WATER,
                        r_crack=0.01, sigma_c=10e6,
                        n_hats=_N_HATS, rho=RHO, nu=NU):
    """
    Integrate S-wave splitting along a ray.

    Parameters
    ----------
    sigma_stack : (N_pts, 3, 3) stress tensors along the ray [Pa]
    ds_array    : (N_pts,)  segment lengths [m]
    p_hat       : (3,) ray propagation direction

    Returns
    -------
    phi_pred : path-weighted circular mean fast-axis azimuth [°]
    dt_pred  : accumulated delay time [s]
    """
    N    = len(sigma_stack)
    phis = np.zeros(N)
    dVs  = np.zeros(N)
    VSs  = np.zeros(N)

    for i, sig in enumerate(sigma_stack):
        C_eff, _ = effective_stiffness(
            sig, mu, epsilon0, K_f, r_crack, sigma_c, n_hats, nu)
        phis[i], dVs[i], VSs[i], _ = christoffel_splitting(C_eff, p_hat, rho)

    # Weight by local splitting strength × segment length
    w     = dVs * ds_array
    w_sum = w.sum()
    if w_sum < 1e-30:
        phi_pred = float(np.mean(phis))
    else:
        ang2     = 2.*np.radians(phis)
        phi_pred = float(np.degrees(
            np.arctan2((w*np.sin(ang2)).sum(),
                       (w*np.cos(ang2)).sum())) / 2. % 180.)

    V_bar    = VSs.mean() if VSs.mean() > 0 else 3500.
    dt_pred  = float(np.sum(dVs / max(V_bar**2, 1.) * ds_array))
    return phi_pred, dt_pred

# ── Unit tests ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    mu = 30e9
    print('=' * 65)
    print('Hudson crack model — corrected unit tests')
    print('=' * 65)

    # Background VS
    VS_bg = np.sqrt(mu / RHO)
    print(f'Background VS = {VS_bg:.0f} m/s  (μ = {mu/1e9:.0f} GPa, ρ = {RHO:.0f} kg/m³)')

    # ── Test A: isotropic medium, no stress, large σ_c → all cracks open
    sigma0 = np.zeros((3,3))
    Ca, fa = effective_stiffness(sigma0, mu, epsilon0=0.05,
                                  sigma_c=1e30, n_hats=_N_HATS)
    phiA, dVA, VS1A, VS2A = christoffel_splitting(Ca, np.array([0.,0.,1.]))
    print(f'\nA. All cracks open, isotropic stress:')
    print(f'   open={fa:.0%}  VS1={VS1A:.0f}  VS2={VS2A:.0f}  δV={dVA:.1f} m/s '
          f'({100*dVA/VS_bg:.2f}%)  — should ≈ 0 (random → isotropic)')
    assert dVA < 5., f'Random open cracks in zero stress should give <5 m/s splitting, got {dVA:.1f}'

    # ── Test B: all cracks closed (sigma_c = 0 → open only when tensile > 0)
    sigma0 = np.zeros((3,3))
    Cb, fb = effective_stiffness(sigma0, mu, epsilon0=0.05,
                                  sigma_c=0., n_hats=_N_HATS)
    _, dVB, _, _ = christoffel_splitting(Cb, np.array([0.,0.,1.]))
    print(f'\nB. Zero stress, σ_c=0 (all cracks neutral/closed):')
    print(f'   open={fb:.0%}  δV={dVB:.1f} m/s  — should ≈ 0')

    # ── Test C: horizontal EW compression → fast axis should be ~N-S (≈90°)
    # EW compression (σ_xx > σ_yy) closes EW-normal cracks preferentially
    sigma_c_test = 40e6   # 40 MPa closure pressure
    sigma_ew = np.array([[80e6, 0., 0.],
                          [0., 10e6, 0.],
                          [0., 0., 40e6]])
    Cc, fc = effective_stiffness(sigma_ew, mu, epsilon0=0.10,
                                  sigma_c=sigma_c_test, n_hats=_N_HATS)
    phiC, dVC, VS1C, VS2C = christoffel_splitting(Cc, np.array([0.,0.,1.]))
    print(f'\nC. EW compression (σ_xx={sigma_ew[0,0]/1e6:.0f} MPa > '
          f'σ_yy={sigma_ew[1,1]/1e6:.0f} MPa):')
    print(f'   open={fc:.0%}  φ_fast={phiC:.1f}° (expect ≈90°=N-S)  '
          f'δV={dVC:.1f} m/s ({100*dVC/VS_bg:.2f}%)')

    # ── Test D: single set of horizontal cracks (n̂=ê₃, all open)
    # Should recover Hudson's result: C₄₄ = C₅₅ = μ(1 - εU₃)
    n_horiz = np.zeros((1, 3)); n_horiz[0, 2] = 1.
    eps_test = 0.05
    U1, U3 = _compliance_coeffs(mu, NU)
    DC4_one = _hudson_4d(n_horiz[0], eps_test, mu)
    C_one   = background_stiffness(mu) + _to_voigt(DC4_one)
    C44_expected = mu * (1 - eps_test * U3)
    C44_got = C_one[3,3]; C55_got = C_one[4,4]
    ok = abs(C44_got - C44_expected) / mu < 1e-6
    print(f"\nD. Single horizontal crack set (ê₃ normal, ε={eps_test}):")
    print(f"   U₁_wet={U1:.3f}  U₃={U3:.3f}  (dimensionless)")
    print(f"   C₄₄ expected = μ(1−εU₃) = {C44_expected/1e9:.4f} GPa")
    print(f"   C₄₄ computed = {C44_got/1e9:.4f} GPa  {'✓' if ok else '✗ ERROR'}")
    print(f"   C₅₅ = {C55_got/1e9:.4f} GPa  (should = C₄₄)")

    # ── Test E: random cracks, no stress → isotropic → δV≈0
    Ce_rand, _ = effective_stiffness(np.zeros((3,3)), mu, epsilon0=0.05,
                                     sigma_c=1e30, K_f=K_WATER, n_hats=_N_HATS)
    phiE, dVE, VS1E, VS2E = christoffel_splitting(Ce_rand, np.array([0.,0.,1.]))
    print(f"\nE. Random cracks, zero stress, all open (expect δV≈0):")
    print(f"   VS1={VS1E:.0f}  VS2={VS2E:.0f}  δV={dVE:.2f} m/s")
    # ── Test F: expected SWS for realistic Axial parameters
    sigma_typical = np.array([[30e6, 0., 0.],
                               [0., 5e6, 0.],
                               [0., 0., 15e6]])   # mild EW compression
    Cf, ff = effective_stiffness(sigma_typical, mu, epsilon0=0.05,
                                  K_f=K_WATER, r_crack=0.01,
                                  sigma_c=20e6, n_hats=_N_HATS)
    phiF, dVF, _, _ = christoffel_splitting(Cf, np.array([0.,0.,1.]))
    L_typical = 2000.   # m path length
    dt_est    = dVF / VS_bg**2 * L_typical * 1e3   # ms
    print(f'\nF. Typical Axial stress (ε₀=0.05, σ_c=20 MPa):')
    print(f'   open={ff:.0%}  φ_fast={phiF:.1f}°  δV={dVF:.1f} m/s  '
          f'→ δt≈{dt_est:.1f} ms over {L_typical:.0f} m path')

    print('\nAll tests complete.')
