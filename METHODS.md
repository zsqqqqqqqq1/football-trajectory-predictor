# Model and validation

## Flight model

Let `r` and `v` denote ball-centre position and velocity, and let `u = v - w` denote velocity relative to the air. The acceleration is the sum of gravity and `(rho * pi * R^2 / (2 * m)) * |u| * (-Cd * u + b × u)`. The default predictor uses still air. The fitted variables are the three components of release velocity, a constant drag coefficient `Cd`, and two components of the effective aerodynamic vector `b`. The two components are expressed in a plane perpendicular to the early reference direction. They summarize the net transverse force over a short flight.

The likelihood compares predicted positions with the two measured positions. Gaussian mode uses independent coordinate errors. Rounding mode uses the intervals implied by the displayed precision, with a default half-width of 0.005 m. The wide and strong priors share the same hard bounds. The strong prior concentrates drag near 0.25 and gives greater weight to smaller transverse forces. The assumptions apply to free flight from a known release point and time.

Four Metropolis chains approximate the parameter posterior. SciPy's DOP853 solver propagates each parameter draw to the first forward crossing or first descending ground contact. A two-dimensional kernel density estimate gives the crossing mode and the highest-density region containing 95% of the conditional probability. The paired validation compares the conditional crossing mean with the reference point. This point-error measure differs from the mode reported for the example.

## Synthetic fixture

[data/validation_cases.json](data/validation_cases.json) contains 24 independent simulated launches. Each has clean coordinates at 0.05 and 0.10 s and a reference crossing at x = 11 m. Both priors receive the same centimetre-rounded positions for each launch. The fit uses release position, timestamps and rounded observations. Generator parameters and held-out crossing coordinates are retained for verification and scoring. Group summaries describe horizontal displacement from the release direction using thresholds of 0.15 and 0.40 m. They characterize this fixture's shot mix.

The optional `python validate.py` command compares a local posterior approximation with a mixture of extended Kalman filters on simulated trajectories. Both methods observe at the same contest times. Its covariance ellipses use a different construction from the main predictor's density contours.

## MATLAB cross-check

Run `python run_analysis.py --deterministic-only` to export the fitted parameters. From the repository root in MATLAB, run `run('matlab/verify_cases.m')`. The script independently integrates the same initial conditions with `ode113` at two tolerance settings. It checks the first crossing against Python and compares the gravity-only trajectory with its analytical solution. Agreement between solvers verifies the numerical calculation.

## Optional field-data checks

Download the original data from the cited providers and pass their locations explicitly:

```sh
python issia.py --csv /path/to/ISSIA-3D.csv
python dfl_j03woh.py --root /path/to/dfl-xml-directory
```

- **ISSIA-3D** uses the 25 Hz three-dimensional ball annotations distributed with SoccerNet-v3D. The adapter scores selected airborne segments against a virtual plane 11 m ahead of each start point. [Data release](https://github.com/mguti97/SoccerNet-v3D/releases)
- **DFL J03WOH** uses the event and raw observed positions XML files for match `DFL-MAT-J03WOH`. The adapter scores airborne segments against the measured field goal line. [Original dataset](https://doi.org/10.6084/m9.figshare.28196177) · [Data paper](https://doi.org/10.1038/s41597-025-04505-y)

Both checks use an airborne start and 0.04 and 0.08 s follow-up frames, reflecting the source frame rate. They provide supplementary evidence about short-window prediction on real tracks. They do not measure coverage for confirmed release-time penalty kicks at 0.05 and 0.10 s.

## Aerodynamic references

- The illustrative drag coefficients 0.15, 0.18 and 0.22 in the fixed-coefficient sensitivity analysis are approximate readings of Figure 4 at specific speeds and orientations of experimental balls. [Hong, Goff and Asai](https://doi.org/10.1177/1754337118794561)
- Free-flight and wind-tunnel studies inform the scale of the drag and transverse force parameters. The prior distributions are modelling choices. [Goff et al.](https://doi.org/10.1088/1361-6404/aa6fcd) · [Asai et al.](https://doi.org/10.1038/s41598-026-56275-1)
