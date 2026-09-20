# Tire-law parameters

Data: concrete (9 runs); sloped sidewalk (11 runs).
Protocol: 1.5 s windows, vx pinned to the wheel speed, IMU slope input, Adam from the scooter preset, 150 steps, front and rear tire tied. Bold = fitted, (held) = left at the preset.

## All-data fit (the reported parameters)

| parameter | brush | fiala | dugoff | burckhardt | pacejka |
|---|---|---|---|---|---|
| C_alpha, zero-slip cornering stiffness (N/rad) | **16,113** | **27,447** | **26,170** | **11,350** | **26,842** |
| cp (N/m^2) | **1.29e+07** | - | - | - | - |
| contact_length a (m) | 0.025 (held) | - | - | - | - |
| cy (N/rad) | - | **27,523** | **26,170** | - | **26,842** |
| cx (N per unit kappa) | - | 4,500 (held) | 4,500 (held) | - | 4,500 (held) |
| Burckhardt c1 | - | - | - | 1.280 (held) | - |
| Burckhardt c2 | - | - | - | **21.70** | - |
| Burckhardt c3 | - | - | - | 0.520 (held) | - |
| Pacejka C_y | - | - | - | - | 1.30 (held) |
| Pacejka E_y | - | - | - | - | -0.20 (held) |
| mu | 0.80 (held) | 0.80 (held) | 0.80 (held) | - | 0.80 (held) |
| align_gain | 0.10 (held) | - | - | - | - |
| yaw_inertia (kg m^2) | **7.06** | **7.12** | **7.03** | **7.02** | **7.06** |
| yaw_damping (1/s) | **0.288** | **0.313** | **0.298** | **0.254** | **0.305** |
| residual steer zero (deg motor) | **+0.044** | **+0.042** | **+0.041** | **+0.054** | **+0.041** |

## Held-out fit (runs 20260911_163726, 20260911_191928 excluded)

| parameter | brush | fiala | dugoff | burckhardt | pacejka |
|---|---|---|---|---|---|
| C_alpha, zero-slip cornering stiffness (N/rad) | **15,857** | **26,319** | **24,817** | **11,248** | **25,585** |
| cp (N/m^2) | **1.27e+07** | - | - | - | - |
| contact_length a (m) | 0.025 (held) | - | - | - | - |
| cy (N/rad) | - | **26,389** | **24,817** | - | **25,586** |
| cx (N per unit kappa) | - | 4,500 (held) | 4,500 (held) | - | 4,500 (held) |
| Burckhardt c1 | - | - | - | 1.280 (held) | - |
| Burckhardt c2 | - | - | - | **21.51** | - |
| Burckhardt c3 | - | - | - | 0.520 (held) | - |
| Pacejka C_y | - | - | - | - | 1.30 (held) |
| Pacejka E_y | - | - | - | - | -0.20 (held) |
| mu | 0.80 (held) | 0.80 (held) | 0.80 (held) | - | 0.80 (held) |
| align_gain | 0.10 (held) | - | - | - | - |
| yaw_inertia (kg m^2) | **7.07** | **7.13** | **7.03** | **7.02** | **7.07** |
| yaw_damping (1/s) | **0.299** | **0.325** | **0.307** | **0.260** | **0.316** |
| residual steer zero (deg motor) | **+0.024** | **+0.023** | **+0.023** | **+0.032** | **+0.023** |

## Preset (starting point of every fit)

| parameter | brush | fiala | dugoff | burckhardt | pacejka |
|---|---|---|---|---|---|
| C_alpha, zero-slip cornering stiffness (N/rad) | 4,500 | 4,498 | 4,500 | 4,498 | 4,500 |
| cp (N/m^2) | 3.6e+06 | - | - | - | - |
| contact_length a (m) | 0.025 | - | - | - | - |
| cy (N/rad) | - | 4,500 | 4,500 | - | 4,500 |
| cx (N per unit kappa) | - | 4,500 | 4,500 | - | 4,500 |
| Burckhardt c1 | - | - | - | 1.280 | - |
| Burckhardt c2 | - | - | - | 8.84 | - |
| Burckhardt c3 | - | - | - | 0.520 | - |
| Pacejka C_y | - | - | - | - | 1.30 |
| Pacejka E_y | - | - | - | - | -0.20 |
| mu | 0.80 | 0.80 | 0.80 | - | 0.80 |
| align_gain | 0.10 | - | - | - | - |
| yaw_inertia (kg m^2) | 8.00 | 8.00 | 8.00 | 8.00 | 8.00 |
| yaw_damping (1/s) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| residual steer zero (deg motor) | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
