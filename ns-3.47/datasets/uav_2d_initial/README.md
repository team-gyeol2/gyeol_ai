# UAV 2D Initial Dataset

This dataset is the first planning-oriented baseline for the capstone UAV topology-control story.
It is intentionally 2D-first so the team can validate topology logic and NetAnim visualization before introducing full 3D mobility and channel effects.

## Scenario scope

- 5 UAVs
- 3 fixed buildings
- 2D map intended for NetAnim-style visualization
- Initial topology goal: every UAV pair should remain reachable within at most 2 hops
- Initial common speed: 5 m/s for all UAVs

## Files

- `scenario_manifest.json`: map settings, metric thresholds, and heuristic rules
- `obstacles.csv`: fixed building rectangles for the 2D map
- `uav_positions.csv`: time-indexed UAV positions for the baseline path set
- `link_metrics.csv`: estimated pairwise link quality, route type, and reconfiguration trigger label
- `topology_summary.csv`: per-time-step topology status summary

## Initial communication-disruption rule

Use the following working rule for the first dataset version:

- `healthy`: RSSI >= -78 dBm and PLR <= 5%
- `degraded`: -85 dBm <= RSSI < -78 dBm or 5% < PLR <= 20%
- `disconnected`: RSSI < -85 dBm or PLR > 20%

Recommended topology reconfiguration trigger:

- PLR >= 15%, or
- RSSI <= -80 dBm, or
- RTT >= 40 ms, or
- throughput <= 4 Mbps, or
- any pair needs more than 2 hops, or
- any pair becomes disconnected

These are coarse design thresholds for dataset bootstrapping, not final PHY-valid thresholds.
They should be tuned after collecting ns-3 simulation traces.

## Why this layout

- `UAV2` acts as the default relay anchor in the 5-UAV baseline.
- Peripheral UAVs move while keeping the graph diameter small.
- Buildings are placed to create selective LOS degradation without immediately destroying the whole topology.
- The dataset is structured so the next step can be:
  1. replay this geometry in ns-3,
  2. compare estimated metrics with simulated RSSI/PLR/RTT,
  3. adjust thresholds and heuristic rules.

## Suggested next steps

1. Mirror these paths and obstacles in an ns-3 scratch scenario with NetAnim output.
2. Replace estimated `link_metrics.csv` values with real ns-3 trace values.
3. Add scenario variants with different per-UAV speeds and obstacle densities.
