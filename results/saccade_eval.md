# G12 Through-Saccade Tracking Evaluation

## G12 Through-Saccade Tracking — Summary

    Rate |    Fix RMS |  Fix Gross |    Sac RMS |  Sac Gross | Delta Gross |  Reseeds | DoD
-----------------------------------------------------------------------------------------------
   2000Hz |    0.59' (0.0098deg) |     0.000 |   33.05' (0.5508deg) |     0.243 |      +0.242 |       27 | PASS
   4000Hz |    0.08' (0.0014deg) |     0.000 |   23.80' (0.3966deg) |     0.119 |      +0.119 |       29 | PASS

## Lock-Rate vs. Velocity (median across seeds)

    Rate |       0k-2k rows/s |       2k-5k rows/s |      5k-10k rows/s |     10k-20k rows/s |     20k-40k rows/s |     40k-80k rows/s |    80k-200k rows/s
----------------------------------------------------------------------------------------------------------------------------------------------------------------
   2000Hz |               1.00 |               0.83 |               0.22 |               0.20 |               0.14 |               0.07 |                n/a
   4000Hz |               1.00 |               0.84 |               0.56 |               0.34 |               0.11 |               0.12 |               0.11

## Diagnosis

### Rate = 2000 Hz

- Fixation RMS: 0.59' (0.0098 deg) — PASSES 0.1 deg DoD
- Fixation gross-error rate: 0.000
- Saccade RMS: 33.05' (0.5508 deg) — FAILS 0.1 deg — see lock-rate diagnosis below
- Saccade gross-error rate: 0.243
- Saccade-vs-fixation gross margin: +0.242 (EXCEEDS 10-15pp target)
- Lock-loss detected at velocities: 5k-10k rows/s (lock=0.22), 10k-20k rows/s (lock=0.20), 20k-40k rows/s (lock=0.14), 40k-80k rows/s (lock=0.07)
  ROOT CAUSE: saccade motion blur reduces fine-NCC below NCC_LOCK_LOSS_THR=0.35 at high velocities. The filter cannot recover perp position during the blur window because (1) the blurred observation is uninformative, and (2) the IMM saccade prediction draws a random saccade direction (not the actual one), so prediction diverges from truth. Recovery occurs after the saccade ends and NCC rises, but the coarse-anchor reseed may place the cloud at the wrong position if it fires mid-saccade. This is the physics limit at the operating rate; the G13 rate-sweep will quantify where tracking is reliable.

### Rate = 4000 Hz

- Fixation RMS: 0.08' (0.0014 deg) — PASSES 0.1 deg DoD
- Fixation gross-error rate: 0.000
- Saccade RMS: 23.80' (0.3966 deg) — FAILS 0.1 deg — see lock-rate diagnosis below
- Saccade gross-error rate: 0.119
- Saccade-vs-fixation gross margin: +0.119 (within 10-15pp target)
- Lock-loss detected at velocities: 10k-20k rows/s (lock=0.34), 20k-40k rows/s (lock=0.11), 40k-80k rows/s (lock=0.12), 80k-200k rows/s (lock=0.11)
  ROOT CAUSE: saccade motion blur reduces fine-NCC below NCC_LOCK_LOSS_THR=0.35 at high velocities. The filter cannot recover perp position during the blur window because (1) the blurred observation is uninformative, and (2) the IMM saccade prediction draws a random saccade direction (not the actual one), so prediction diverges from truth. Recovery occurs after the saccade ends and NCC rises, but the coarse-anchor reseed may place the cloud at the wrong position if it fires mid-saccade. This is the physics limit at the operating rate; the G13 rate-sweep will quantify where tracking is reliable.


## Per-Stream Results (rate=2000 Hz)

 Seed |  n_fix |  n_sac |    Fix RMS |  Fix Grs |    Sac RMS |  Sac Grs |  Reseeds
--------------------------------------------------------------------------------
    0 |   1000 |      0 |    0.04' (0.08r) | 0.000   |     n/a (nanr) |   n/a   |        0
    1 |   1000 |      0 |    0.04' (0.09r) | 0.000   |     n/a (nanr) |   n/a   |        0
    2 |   1000 |      0 |    0.04' (0.09r) | 0.000   |     n/a (nanr) |   n/a   |        0
    3 |   1000 |      0 |    0.05' (0.09r) | 0.000   |     n/a (nanr) |   n/a   |        0
    4 |    979 |     21 |    2.15' (4.47r) | 0.001   |   34.60' (71.9r) | 0.286   |        5
    5 |    962 |     38 |    0.44' (0.92r) | 0.000   |   36.16' (75.1r) | 0.158   |        6
    6 |   1000 |      0 |    0.05' (0.10r) | 0.000   |     n/a (nanr) |   n/a   |        0
    7 |   1000 |      0 |    0.09' (0.18r) | 0.000   |     n/a (nanr) |   n/a   |        0
    8 |    997 |      3 |    0.64' (1.33r) | 0.000   |   11.07' (23.0r) | 0.000   |        1
    9 |    925 |     75 |    1.21' (2.51r) | 0.000   |   72.49' (150.6r) | 0.387   |       12
   10 |    992 |      8 |    0.38' (0.78r) | 0.000   |   43.59' (90.6r) | 0.625   |        2
   11 |    997 |      3 |    1.94' (4.03r) | 0.001   |    0.38' (0.8r) | 0.000   |        1

## Per-Stream Results (rate=4000 Hz)

 Seed |  n_fix |  n_sac |    Fix RMS |  Fix Grs |    Sac RMS |  Sac Grs |  Reseeds
--------------------------------------------------------------------------------
    0 |   2000 |      0 |    0.04' (0.09r) | 0.000   |     n/a (nanr) |   n/a   |        0
    1 |   2000 |      0 |    0.04' (0.09r) | 0.000   |     n/a (nanr) |   n/a   |        0
    2 |   1996 |      4 |    0.19' (0.40r) | 0.000   |    4.50' (9.4r) | 0.000   |        1
    3 |   1997 |      3 |    0.05' (0.11r) | 0.000   |    0.28' (0.6r) | 0.000   |        0
    4 |   1975 |     25 |    0.05' (0.10r) | 0.000   |   42.74' (88.8r) | 0.360   |        2
    5 |   1934 |     66 |    0.05' (0.10r) | 0.000   |   28.90' (60.0r) | 0.167   |        6
    6 |   2000 |      0 |    0.04' (0.09r) | 0.000   |     n/a (nanr) |   n/a   |        0
    7 |   2000 |      0 |    0.07' (0.14r) | 0.000   |     n/a (nanr) |   n/a   |        0
    8 |   1993 |      7 |    0.25' (0.53r) | 0.000   |    4.43' (9.2r) | 0.000   |        1
    9 |   1851 |    149 |    0.04' (0.09r) | 0.000   |   44.06' (91.5r) | 0.208   |       17
   10 |   1990 |     10 |    0.11' (0.24r) | 0.000   |   41.67' (86.6r) | 0.100   |        2
   11 |   2000 |      0 |    0.05' (0.11r) | 0.000   |     n/a (nanr) |   n/a   |        0