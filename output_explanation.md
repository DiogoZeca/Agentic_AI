# Spike Predictor — Output Reference

## API

| | |
|---|---|
| **Summary endpoint** | `POST http://spike-api.10.255.42.75.nip.io/summary` |
| **Health check** | `GET http://spike-api.10.255.42.75.nip.io/health` |
| **Update frequency** | Every 5 minutes (cron job on OSM VM — automatic) |

---

## Response Structure

```json
{
  "summary": {
    "spikes_in_15m": 0,
    "spikes_in_30m": 0,
    "spikes_in_45m": 0,
    "spikes_in_60m": 1,
    "nodes_requiring_action": 1
  },
  "per_node": [
    {
      "machine_id": 1,
      "status": "success",
      "spike_60m": true,
      "spike_imminent": false,
      "severity": "moderate",
      "alarm_source": "60m_severity",
      "scheduler_score": 23,
      "horizon": "60m",
      "recommended_action": "migrate_jobs",
      "p_spike_smoothed": 0.77,
      "consecutive_alarms": 3
    }
  ],
  "predicted_at": "2026-05-18T10:00:00+00:00",
  "machines_total": 1,
  "machines_predicted": 1,
  "machines_cold_start": 0
}
```

---

## `summary` Block — Cluster-Level Overview

| Field | Meaning |
|-------|---------|
| `spikes_in_15m` | Number of machines predicted to spike within 15 minutes |
| `spikes_in_30m` | Within 30 minutes |
| `spikes_in_45m` | Within 45 minutes |
| `spikes_in_60m` | Within 60 minutes (sustained risk, may not be imminent) |
| `nodes_requiring_action` | Machines where `recommended_action` is `migrate_jobs` or `preempt_now` |

All counts: **0 = everything is fine**. Non-zero means act.

---

## `per_node` Block — Per-Machine Detail

| Field | Type | Meaning |
|-------|------|---------|
| `machine_id` | int | Integer ID for the node (stable across calls) |
| `status` | string | `success` = prediction OK · `cold_start` = not enough history yet, ignore this machine |
| `spike_60m` | bool | `true` = 60-minute severity model says a spike is coming |
| `spike_imminent` | bool | `true` = a short-horizon model (15/30/45m) says a spike is coming soon |
| `severity` | string | `no_spike` / `moderate` / `severe` |
| `alarm_source` | string | Which model triggered the action (see table below) |
| `scheduler_score` | int | **0–100. Higher = safer.** Use directly in a K8s Score plugin. 100 = no risk, 0 = spike certain |
| `horizon` | string | Earliest alarming time window: `15m`, `30m`, `45m`, `60m`, or `null` (no alarm) |
| `recommended_action` | string | What the scheduler should do (see table below) |
| `p_spike_smoothed` | float | Smoothed spike probability (0.0–1.0). The raw number behind `scheduler_score` |
| `consecutive_alarms` | int | How many consecutive 5-minute cycles this machine has been alarming |

---

## `recommended_action` — What To Do

| Value | Meaning | Suggested response |
|-------|---------|-------------------|
| `normal` | No spike predicted | No action needed |
| `migrate_jobs` | Spike likely within 60m | Move non-critical workloads off this machine |
| `preempt_now` | Spike imminent (within 15m) | Immediately evict or stop new scheduling to this machine |

---

## `alarm_source` — Which Model Triggered the Action

| Value | Meaning |
|-------|---------|
| `none` | No alarm — `recommended_action` is `normal` |
| `binary_15m` | 15-minute binary model alarmed (most urgent) |
| `binary_30m` | 30-minute binary model alarmed |
| `binary_45m` | 45-minute binary model alarmed |
| `60m_severity` | 60-minute severity model alarmed (sustained risk, not necessarily imminent) |
| `null` | Machine is in `cold_start` — no prediction available |

---

## The `scheduler_score`?

`scheduler_score` is inside **`per_node`**, not a separate endpoint. Both `summary` and `per_node` come from the same single `POST /summary` call:

- **`summary`** → cluster-level: "is anything wrong right now?" Check `nodes_requiring_action` first.
- **`per_node[]`** → machine-level: "which specific node should I avoid?" Read `scheduler_score` per node.

---

## Live Prediction Command

```bash
python3 ~/spike/thanos.py \
  --thanos-url http://thanos-query.10.255.42.75.nip.io \
  --mode post \
  --api-url http://spike-api.10.255.42.75.nip.io/summary 2>/dev/null
```

`nodes_requiring_action: 0` = everything fine. `scheduler_score` per node: 0–100, higher = safer.

---

## Key Notes

- **`scheduler_score` is the simplest integration point.** It lives in `per_node[]`. Wire it directly into the K8s Score phase — lower score = avoid this node. No formula needed.
- **`spike_60m` vs `spike_imminent` answer different questions.** A machine can have `spike_60m: true` (sustained risk over the hour) but `spike_imminent: false` (nothing in the next 45 minutes). Use `spike_imminent` for urgent decisions, `spike_60m` for planning.
- **`status: cold_start` machines should be ignored.** Less than 60 minutes of history available. Excluded from all `summary` counts but present in `per_node` for visibility.
- **Alarms are already debounced.** A machine only fires after 2 consecutive spiking predictions (10 minutes of sustained signal). No need to add extra debounce logic on your side.
- **`consecutive_alarms` reflects persistence.** A value of 6 means 30 minutes of continuous alarming — much more serious than a value of 2.

---