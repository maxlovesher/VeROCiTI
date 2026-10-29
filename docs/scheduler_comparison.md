# Signal scheduling: comparison

Generated 2026-10-13 by `city flow model/scripts/compare_schedulers.py` on the built-in 5-junction simulator (mixed Indian traffic: cars, two-wheelers, autos, buses, trucks). Each figure is the mean ± standard deviation over 8 random seeds; every policy sees the same seeds. 300 s warm-up, then 30 min measured.

All policies share the same safety rules: the controller decides every 3 s, minimum green is 30 s, yellow 6 s. VeROCiTI fair adds aging (one full point of priority per 90 s on red) and a 120 s wait guarantee.

**Longest wait** is the longest single standstill of each vehicle, including vehicles still stuck when the run ended, so a policy can't hide starvation by never letting the starved cars finish.

## Balanced demand

Every route equally likely, 64 vehicles in the network.

| Policy | Throughput (veh/h) | Avg travel time (s) | Avg stopped time (s) | 95th pct longest wait (s) | Worst single wait (s) | Waits over 2 min (%) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed time (round robin) | 2622 ± 22 | 82 ± 1 | 35 ± 0 | 82 ± 0 | 87 ± 0 | **0.0 ± 0.0** |
| FCFS | 2892 ± 46 | 74 ± 1 | 24 ± 0 | **36 ± 1** | **42 ± 1** | **0.0 ± 0.0** |
| SRTF | 2554 ± 44 | 82 ± 2 | 34 ± 2 | 98 ± 8 | 163 ± 22 | 2.3 ± 1.0 |
| Longest queue first | **2980 ± 32** | **72 ± 1** | **24 ± 1** | 39 ± 1 | 66 ± 10 | **0.0 ± 0.0** |
| Max-pressure | 2930 ± 62 | 73 ± 1 | 25 ± 1 | 42 ± 1 | 92 ± 21 | **0.0 ± 0.1** |
| Congestion only | 2892 ± 41 | 74 ± 1 | 25 ± 1 | 39 ± 1 | 64 ± 4 | **0.0 ± 0.0** |
| VeROCiTI fair | 2878 ± 51 | 75 ± 2 | 24 ± 1 | 36 ± 1 | **42 ± 1** | **0.0 ± 0.0** |

## Rush hour on the main road

The J1-J3-J4 east-west corridor carries 8x the demand of each side street, 64 vehicles.

| Policy | Throughput (veh/h) | Avg travel time (s) | Avg stopped time (s) | 95th pct longest wait (s) | Worst single wait (s) | Waits over 2 min (%) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed time (round robin) | 1906 ± 62 | 110 ± 4 | 50 ± 2 | 84 ± 0 | 87 ± 0 | **0.0 ± 0.0** |
| FCFS | 1994 ± 32 | 106 ± 2 | 40 ± 1 | **41 ± 0** | **52 ± 8** | **0.0 ± 0.0** |
| SRTF | 1863 ± 50 | 113 ± 3 | 48 ± 3 | 61 ± 5 | 134 ± 44 | 0.2 ± 0.2 |
| Longest queue first | **2397 ± 41** | **90 ± 2** | **27 ± 1** | 68 ± 7 | 161 ± 21 | 0.5 ± 0.3 |
| Max-pressure | 2037 ± 40 | 104 ± 2 | 38 ± 2 | 43 ± 3 | 192 ± 44 | 0.6 ± 0.4 |
| Congestion only | 2131 ± 45 | 100 ± 2 | 36 ± 2 | 50 ± 3 | 100 ± 9 | **0.0 ± 0.0** |
| VeROCiTI fair | 2017 ± 28 | 105 ± 2 | 39 ± 1 | 41 ± 0 | 53 ± 11 | **0.0 ± 0.0** |

Who pays for the main road's priority (average stopped time per trip):

| Policy | Main road (s) | Side streets (s) |
|---|---:|---:|
| Fixed time (round robin) | 70 | 28 |
| FCFS | 58 | 17 |
| SRTF | 74 | 12 |
| Longest queue first | 24 | 31 |
| Max-pressure | 49 | 22 |
| Congestion only | 43 | 26 |
| VeROCiTI fair | 54 | 19 |

## Heavy rush hour (near saturation)

Same 8x main-road split with 130 vehicles in the network.

| Policy | Throughput (veh/h) | Avg travel time (s) | Avg stopped time (s) | 95th pct longest wait (s) | Worst single wait (s) | Waits over 2 min (%) |
|---|---:|---:|---:|---:|---:|---:|
| Fixed time (round robin) | 4203 ± 58 | 102 ± 1 | 48 ± 1 | 84 ± 0 | 87 ± 0 | **0.0 ± 0.0** |
| FCFS | **4794 ± 71** | **90 ± 1** | **32 ± 1** | **39 ± 1** | **42 ± 0** | **0.0 ± 0.0** |
| SRTF | 2941 ± 283 | 131 ± 10 | 78 ± 7 | 191 ± 54 | 578 ± 152 | 12.1 ± 3.2 |
| Longest queue first | 4336 ± 85 | 100 ± 2 | 41 ± 2 | 74 ± 6 | 156 ± 41 | 0.3 ± 0.3 |
| Max-pressure | 4132 ± 190 | 103 ± 4 | 45 ± 4 | 52 ± 7 | 120 ± 20 | 0.1 ± 0.1 |
| Congestion only | 4215 ± 123 | 101 ± 3 | 43 ± 2 | 45 ± 2 | 121 ± 86 | **0.0 ± 0.0** |
| VeROCiTI fair | 4448 ± 153 | 97 ± 3 | 38 ± 2 | 41 ± 0 | 59 ± 31 | **0.0 ± 0.0** |

Who pays for the main road's priority (average stopped time per trip):

| Policy | Main road (s) | Side streets (s) |
|---|---:|---:|
| Fixed time (round robin) | 82 | 32 |
| FCFS | 63 | 20 |
| SRTF | 228 | 25 |
| Longest queue first | 52 | 33 |
| Max-pressure | 84 | 25 |
| Congestion only | 82 | 23 |
| VeROCiTI fair | 79 | 20 |

## Ambulances

Rush-hour demand. Ambulance trip time from dispatch to hospital, mean (worst) over the seeds. Free-flow time for a 2-link trip is 82 s.

| Case | Ambulance | No preemption | Green corridor |
|---|---|---:|---:|
| One ambulance | East (J1→J3→J4) | 96 (123) | 91 (96) |
| Two ambulances, same moment, crossing at J3 | East (J1→J3→J4) | 96 (123) | 91 (96) |
|  | South (J2→J3→J5) | 127 (189) | 106 (111) |

With two ambulances, J3 can only show one emergency direction, so it goes to whichever ambulance will reach it first (ties: more severe case, then earlier dispatch). The other waits at the stop line, gets J3 as soon as the first has cleared it, and never runs a red.
