# K8s Deployment — Session Anchor

## Goal
Deploy the spike prediction REST API into the OSM k3s cluster so the scheduler can call it via HTTP — no inline model execution in the scheduler pod.

**STATUS: FULLY DEPLOYED AND LIVE ✅**

---

## Infrastructure

| Component | Details |
|-----------|---------|
| OSM K8s cluster | k3s v1.29.3 · single node (`osm`) · <CLUSTER_IP> |
| Storage | `local-path` provisioner (default) · `WaitForFirstConsumer` |
| Container registry | None — image imported directly into k3s containerd |
| Docker VM | Same as OSM VM — Docker v29.1.3 runs directly on <CLUSTER_IP> |
| Target namespace | `spike` ✅ created |
| Ingress controller | nginx (not Traefik — corrected during deploy) |

---

## Live Endpoints

| Purpose | URL |
|---------|-----|
| API health | `http://spike-api.<CLUSTER_IP>.nip.io/health` |
| API ready | `http://spike-api.<CLUSTER_IP>.nip.io/ready` |
| Scheduler summary | `POST http://spike-api.<CLUSTER_IP>.nip.io/summary` |
| Thanos metrics | `http://thanos-query.<CLUSTER_IP>.nip.io` (nginx Ingress — no port-forward needed) |

---

## Current Response Format (POST /summary)

```json
{
  "summary": {
    "spikes_in_15m": 0,
    "spikes_in_30m": 0,
    "spikes_in_45m": 0,
    "spikes_in_60m": 0,
    "nodes_requiring_action": 0
  },
  "per_node": [
    {
      "machine_id": 1,
      "status": "success",
      "spike_60m": false,
      "spike_imminent": false,
      "severity": "no_spike",
      "alarm_source": "none",
      "scheduler_score": 94,
      "horizon": null,
      "recommended_action": "normal",
      "p_spike_smoothed": 0.063705,
      "consecutive_alarms": 0
    }
  ],
  "predicted_at": "2026-05-15T10:58:17+00:00",
  "machines_total": 1,
  "machines_predicted": 1,
  "machines_cold_start": 0
}
```

---

## All Steps — COMPLETE ✅

- [x] **Step 1** — K8s manifests (`namespace`, `pvc`, `populate-pvc`, `deployment`, `service`, `ingress`)
- [x] **Step 2** — Source transferred + Docker image built on OSM VM (`sudo docker build`)
- [x] **Step 3** — Image imported into k3s (`sudo docker save | sudo k3s ctr images import -`)
- [x] **Step 4** — k8s manifests + 5 model artifact dirs transferred to VM
- [x] **Step 5** — Namespace + PVC created, artifacts loaded via `kubectl cp`
- [x] **Step 6** — Deployment, Service, Ingress applied. Fixed: nginx not Traefik, deleted stale `ingress-nginx-admission` webhook
- [x] **Step 7** — Verified: `/health` and `/ready` return 200 externally
- [x] **Step 8** — Thanos adapter built + tested (32 tests). Grid reindex fixes sparse data → 24 buckets always
- [x] **Step 9** — Cron job active on OSM VM (every 5 min, logs to `~/spike/thanos.log`)
- [x] **Step 10** — API response enhanced: `spike_60m`, `spike_imminent`, `alarm_source`, `scheduler_score`, `nodes_requiring_action`. 490 tests passing.

---

## VM directory layout (OSM VM: <CLUSTER_IP>)

```
~/spike/
├── build/          ← source code (spike/ + pyproject.toml)
├── k8s/            ← k8s manifests
├── thanos.py       ← Thanos adapter (copy of training/adapters/thanos.py)
└── thanos.log      ← cron job output
```

---

## Cron job (active)

```
*/5 * * * * python3 /home/atnoguser/spike/thanos.py \
    --thanos-url http://thanos-query.<CLUSTER_IP>.nip.io \
    --mode post \
    --api-url http://spike-api.<CLUSTER_IP>.nip.io/summary \
    >> /home/atnoguser/spike/thanos.log 2>&1
```

Check logs: `tail -f ~/spike/thanos.log`

---

## To update the deployed API after a code change

```bash
# 1. Transfer changed file(s)
scp spike/api.py atnoguser@<CLUSTER_IP>:~/spike/build/spike/api.py

# 2. Rebuild (cached layers make this fast)
ssh atnoguser@<CLUSTER_IP> "cd ~/spike/build && sudo docker build -f spike/Dockerfile -t spike-api:latest ."

# 3. Re-import + rollout restart
ssh atnoguser@<CLUSTER_IP> "sudo docker save spike-api:latest | sudo k3s ctr images import -"
ssh atnoguser@<CLUSTER_IP> "kubectl rollout restart deployment/spike-api -n spike"
ssh atnoguser@<CLUSTER_IP> "kubectl rollout status deployment/spike-api -n spike"
```

---

## Artifact paths (inside container)

| Path | Content |
|------|---------|
| `/app/data/full_run/spike/` | 60m model, thresholds, calibrators (MODEL_DIR) |
| `/app/data/full_run/spike_15m/` | 15m binary model |
| `/app/data/full_run/spike_30m/` | 30m binary model |
| `/app/data/full_run/spike_45m/` | 45m binary model |
| `/app/data/full_run/spike_severe_ovr/` | OVR cascade model |

PVC mounted at `/app/data/full_run` (read-only).

---

## Key decisions made

- nginx ingress controller (not Traefik) — `ingressClassName: nginx`
- Deleted stale `ingress-nginx-admission` validating webhook (was blocking apply, safe to remove)
- No container registry — image imported via `k3s ctr images import` + `imagePullPolicy: Never`
- Thanos accessed via nginx Ingress (`thanos-query.<CLUSTER_IP>.nip.io`) — no port-forward needed
- Grid reindex in thanos.py: CPU/disk → zero-fill, memory → forward-fill (handles sparse OSM data)
- `scheduler_score` (0-100) maps directly to K8s Score plugin — no formula needed by integrator
- `spike_60m` vs `spike_imminent` disambiguates 60m severity model vs binary horizon models
- `alarm_source` field explains which model drove `recommended_action`
