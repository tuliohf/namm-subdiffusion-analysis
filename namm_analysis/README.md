# Na-MMT subdiffusion — trajectory analysis codes

Post-processing codes for

> T. Honorio, *Anomalous water and Na⁺ transport in Na-montmorillonite nanopores*, J. Phys. Chem. C (2026), \[DOI to add].

Inputs are the LAMMPS `compute msd` / `compute msd/nongauss` output (for the MSD fits) and LAMMPS custom dumps of unwrapped coordinates (`dump ... custom N file id xu yu zu`), one file per species and trajectory: `dump.Iw_unwrap_<c0>_<rep>.lammpstrj` (Na⁺) and `dump.Ow_unwrap_<c0>_<rep>.lammpstrj` (water oxygen). The dumps use a five-phase schedule (dense early windows, then every 20,000 steps = 40 ps, timestep 2 fs).

Requirements: Python ≥ 3.9, NumPy, SciPy (pandas only for post-processing the CSV files).

## Contents

| File | Quantities | Paper |
|---|---|---|
| `msd_fits.py` | two-crossover fit of MSD$_\parallel(t)$ ($D_\alpha$, $\alpha$, $t_b$, $t_f$), $D_F=D_\alpha t_f^{\alpha-1}$, per-replica spread of $D_F$, convergence with trajectory length | Theory (two-crossover model), SI "Fitting considerations" |
| `gs_fs_run.py` | $P_\parallel(r,t)=2\pi r\,G_{s,\parallel}$, $G_{s,\perp}(z,t)$, $F_{s,\parallel}(q,t)=\langle J_0(q\,\Delta r_\parallel)\rangle$, $F_{s,\perp}(q,t)=\langle\cos(q\,\Delta z)\rangle$, NGP$_\parallel$ (d = 2), NGP$_\perp$ (d = 1), MSD$_\parallel$, MSD$_\perp$; multi-origin averaging, $q^*=2.8$ Å⁻¹ | Methods (van Hove / ISF / NGP), Results "Displacement statistics" |
| `residence.py` | survival functions and continuous residence times $\tau_\mathrm{site}$, $\tau_\mathrm{inner}$, $\tau_\mathrm{outer}$, $\tau_\mathrm{mobile}$, $\tau_\mathrm{Na\text{-}shell}$, $\tau_\mathrm{surf}$ | Methods, SI "Residence-time analysis" |
| `tamsd.py` | time-averaged MSD per particle, ratio to the ensemble MSD, block stationarity, ergodicity-breaking parameter EB(Δ) and EB(T), split by interlayer | SI "Time-averaged MSD … and ergodicity" |
| `jumps.py` | cage-centre jump detection, Kaplan–Meier waiting-time survival Ψ(τ), β_loc, β_w, τ*, ⟨τ⟩, ⟨cos θ⟩, D_RW, jumps per ion, van Hove tails $G_s(\Delta x,t)/\sigma$ and excess kurtosis | SI "Waiting-time and displacement distributions" |
| `sampling.py` | corrections for the non-uniform dump schedule: time-density-weighted origins, phase-boundary exclusion, phase-5 grid snapping, cumulative-minimum filter | SI "Residence-time analysis" |
| `io_lammps.py` | dump reader, frame alignment, uniform-tail extraction | — |
| `tests/make_synthetic.py` | Brownian test trajectories with the same dump schedule | — |
| `tests/make_synthetic_msd.py` | synthetic `compute msd` files following the two-crossover model | — |

## Usage

```bash
# two-crossover fit (replicas of one state are averaged; species: water or na)
python3 msd_fits.py -i MSD_16_A.txt MSD_16_B.txt MSD_16_C.txt --species water \
    --per-replica --convergence -o fit_16_water

# van Hove, ISF, directional NGP (one output folder per dump)
python3 gs_fs_run.py -i dump.Ow_unwrap_16_1.lammpstrj dump.Iw_unwrap_16_1.lammpstrj \
    -o gsfs --n-lags 25 --n-bins 60 --max-pairs 3000 --q 2.8

# residence times (framework O and substitution sites as plain x y z tables)
python3 residence.py --na dump.Iw_unwrap_16_1.lammpstrj --water dump.Ow_unwrap_16_1.lammpstrj \
    --framework Ofw_16.xyz --sites subs_16.xyz -o res_16_1

# time-averaged MSD (replicas pooled as extra particles)
python3 tamsd.py -i dump.Iw_unwrap_16_*.lammpstrj -o tamsd_Na_16 --by-gallery

# jumps and waiting times (R = 2.6 Å for Na+, 2.0 Å for water)
python3 jumps.py -i dump.Iw_unwrap_16_*.lammpstrj -o jumps_Na_16 --R 2.6 --tf 1.7
```

Replica averaging (mean ± SEM on a common log-t grid) is done on the CSV outputs.

## Conventions

- $d_{001}=L_z/2$ (two clay layers per cell). Pore sizes are reported as $d_{001}$, not by the LAMMPS index c0.
- MSD$_\parallel$ in the paper is the per-component in-plane value, $\tfrac12\langle\Delta x^2+\Delta y^2\rangle$. `gs_fs_run.py` writes the full in-plane value $\langle\Delta x^2+\Delta y^2\rangle$; `tamsd.py` uses the per-component value.
- NGP: $\alpha_2=\langle r^4\rangle/[(1+2/d)\langle r^2\rangle^2]-1$ with d = 2 (in-plane) and d = 1 (out-of-plane).
- Cutoffs: inner/outer/mobile Na⁺ at 3.3 and 5.5 Å (minima of the $r_\mathrm{min}$(Na–O$_\mathrm{fw}$) distribution); Na⁺ hydration shell 3.15 Å; water–surface 3.5 Å (geometric).
- Residence times are integrals of $S(t)$ up to the last reliable lag; none is reported when $S(T)>0.05$.
- Gallery assignment uses the z of the mobile species, not of the substitution atoms.

## Provenance

The analyses were run interactively in 2026 with AI assistants (Claude, Anthropic; ChatGPT, OpenAI), on the author's trajectories and under the author's direction.

- `gs_fs_run.py`, the two sampling functions in `sampling.py`, and the fitting functions in `msd_fits.py` are the code used for the paper (recovered from the session logs; changes listed in the file headers).
- `residence.py`, `tamsd.py`, `jumps.py` are re-implementations of the published methods, written after the fact because the original scripts were not preserved in full. They have been tested on synthetic Brownian trajectories but not yet checked against the published values.

## Licence

\[To choose, e.g. MIT.]

## Citation

If you use these codes, please cite the paper above.
