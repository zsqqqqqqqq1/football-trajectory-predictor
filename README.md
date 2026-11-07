# Football Trajectory Predictor

Estimate where a football will cross a target plane from two early three-dimensional position measurements. The predictor reports the most probable crossing point, a 95% predictive region, and the probability that the ball reaches the plane before hitting the ground. The worked example is [Problem B of the 2026 University Physics Competition](https://www.uphysicsc.com/2026contest.html).

## Install and run

Use Python 3.12 in a virtual environment. Run these commands from the repository root on Windows, macOS, or Linux.

```sh
python -m pip install -r requirements.txt
python predict.py example_input.json --noise rounding --prior wide --output results/example
```

The example uses centimetre rounding as an explicit observation assumption. For positions measured with independent 1 cm errors per coordinate, use `--noise gaussian --sigma 0.01`. Select `--prior strong` to test stronger aerodynamic regularization. The tested environment used Python 3.12.7, NumPy 1.26.4, SciPy 1.17.0, and Matplotlib 3.9.2.

## Predict another trajectory

Copy [example_input.json](example_input.json) and edit the values below. Use metres and seconds. Choose the positive x direction toward the target plane, y laterally, and z vertically upward from the ground. For goal probabilities, choose the lateral origin at the centre of the goal.

| Field | Meaning |
| --- | --- |
| `r0` | Ball-centre position at release, treated as known. For the example, the ball starts at z = 0.11 m. |
| `times` | Two increasing positive times measured from release. Use `[0.05, 0.10]` to follow the contest protocol. |
| `observations` | Two rows of ball-centre coordinates in `[x, y, z]` order, corresponding to those times. |
| `target_x` | Forward target plane. The default is x = 11 m. |

Pass the new file to `predict.py`. For best separation of runs, select a distinct `--output` directory for each input and configuration.

- `summary.json` reports the crossing probability, the goal-plane mode and 95% region when sufficient crossing samples are available, and the sampling diagnostics. The geometric goal probability uses the standard 7.32 m by 2.44 m rectangle centred on y = 0. It counts trajectories that cross inside this rectangle among all sampled trajectories.
- `samples.npz` stores fitted parameter draws and the associated `[time, lateral position, height]` values. Each sample records whether it crossed, hit the ground first, or reached the integration time limit.
- `density.npz` contains the density grid and contour level for the 95% highest-density region, conditional on crossing before ground contact.

Check `diagnostics.reliable` before interpreting a run. A diagnostic warning may warrant more retained draws or a longer warm-up. Rounding mode requires a trajectory compatible with all six measurement intervals. If the observations include appreciable measurement noise before rounding, choose an error model that reflects that measurement process.

## Reproduce and validate

| Command | Result |
| --- | --- |
| `python -m unittest discover -p "test_*.py" -v` | Checks the equations, integration events, input handling and observation likelihood. |
| `python run_analysis.py --deterministic-only` | Reproduces point fits, identifiability and drag sensitivity, and exports input for MATLAB. |
| `python reproduce_example.py --workers 4` | Reproduces the four example posterior results with their documented seeds and sampling settings. Use `--workers 1` for serial execution. |
| `python plot_results.py` | Plots the example posterior distributions after the preceding command. Exports PDF, SVG and PNG. |
| `python compare_rounding_priors.py --draws 4000 --workers 4` | Compares the two priors on 24 frozen launches using observations at 0.05 and 0.10 s. Reports paired point error, coverage and area. |

The paired comparison saves a checkpoint after each completed fit. Repeat the same command to resume. To revisit incomplete pairs or pairs with sampling warnings, run `python compare_rounding_priors.py --draws 8000 --workers 4 --retry-failed`. The separate MATLAB integration and optional field-data checks are described in [METHODS.md](METHODS.md). Prediction and synthetic validation use only the files in this repository.

## Reference example

| Observation assumption | Prior | Mode `(y, z)`, m | 95% region area, m² |
| --- | --- | --- | --- |
| Gaussian, 1 cm per axis | Wide | (2.79, 2.03) | 3.02 |
| Gaussian, 1 cm per axis | Strong | (3.00, 1.90) | 1.66 |
| Centimetre rounding | Wide | (2.78, 1.93) | 0.30 |
| Centimetre rounding | Strong | (2.73, 2.01) | 0.27 |

The number of decimal places in the supplied coordinates does not establish measurement accuracy. These predictions are conditional on the observation and aerodynamic assumptions described in [METHODS.md](METHODS.md).
