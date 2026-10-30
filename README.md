# Topological Correlation of Distributed Attacks with Graph Neural Networks

A **GNN-based alert correlation function** that turns an avalanche of alerts during a distributed attack (DDoS and similar) into a single event, and traces it back to its **targets**, **attacking sources** and **propagation path**. Includes an extension that merges alerts from **two Suricata IDS probes** into DoS incidents, in a rule-based version and an AI version.

---

## Problem

During a distributed attack, thousands of external sources hit the infrastructure at once. The entry point saturates, then internal services slow down and fail in a domino effect. Operators see everything turn red without knowing whether it is an **attack, an outage, or a legitimate traffic spike**.

Threshold rules are bypassed by botnets (each source stays under the limit) and raise false positives on flash crowds. Per-flow ML (Random Forest, XGBoost) judges each flow in isolation and cannot see thousands of sources converging on one target. A distributed attack is a **graph pattern**, so a GNN is a natural fit.

## What the function returns

`correlation_attaque_distribuee(flux, metriques, alertes, fenetre="10s")` takes a traffic window and returns:

1. the **verdict** (normal, legitimate spike, attack) and the **attack type**
2. the **targeted component(s)**
3. the **attacking sources** (as subnets, indicators rather than certain attribution, because of spoofing)
4. the **impacted components** (propagation along internal dependencies)
5. a **confidence score** and an **explanation**

```python
{
  "verdict": "attaque",
  "type": "ddos_http_flood",
  "confiance": 0.94,
  "cibles": [{"composant": "ingress", "score": 0.97}],
  "sources": [{"sous_reseau": "10.8.3.0/24", "score": 0.91}],
  "impactes": ["frontend", "cart", "redis"],
  "explication": "convergence de 412 sources vers l'ingress, ..."
}
```

## Attacks covered

| Family | Examples | Priority |
|---|---|---|
| Volumetric DDoS | UDP flood, ICMP flood | 1 |
| Protocol DDoS | SYN flood | 1 |
| Application DDoS (L7) | HTTP flood | 1 |
| Slow attacks | Slowloris, slow POST | 2 |
| Distributed brute force | Credential stuffing | 2 |
| Distributed scan | Port scan spread over several IPs | 3 |

## Architecture

```
Lab (Kubernetes) ─┐                                   ┌─ Graph head: normal / legit spike / attack + type
                  ├─► Flows, metrics, alerts ─► Graph ─► Shared GAT encoder ─┼─ External nodes head: attacker or legitimate
Public datasets ──┘   (per time window)                  (2-3 layers,       └─ Internal nodes head: target / impacted / healthy
                                                          residual)
                                              Output: verdict, targets, sources, propagation path,
                                                      confidence + explanation (GAT attention, GNNExplainer)
```

**Graph (one per sliding window, 10 s with a 5 s step)**

- **Nodes:** external sources (grouped by /24 when too many) and internal components (ingress, services, pods, databases, Kubernetes nodes)
- **Edges:** network flows weighted by volume, plus internal dependencies ("A calls B", "A runs on B") built automatically from Kubernetes and Jaeger traces
- **Node features** (z-score normalized per node): volume, connections (SYN ratio, duration), diversity (unique sources, ports, URL/user-agent entropy), regularity (bots are very regular), health (CPU, memory, p95 latency, HTTP errors), alerts (Suricata, Alertmanager), node type

**Model:** GAT encoder (explainable), compared with GraphSAGE (inductive, neighbor sampling helps when thousands of sources hit one node). Weighted multi-task loss with focal loss for class imbalance. Possible evolution: spatio-temporal GNN (GAT + GRU).

## Data

**Isolated lab.** [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo) on a 3-node `kind` cluster (`gnn-lab`), exposed through an ingress, with Cilium/Hubble, a default-deny NetworkPolicy, Prometheus/Grafana and Suricata.

- Normal traffic and **legitimate spikes** (flash crowds, labeled as normal so the model does not learn "lots of traffic = attack")
- Controlled attack scenarios launched from attacker containers inside the cluster (simulated botnet)
- Each scenario records type, start, end, attacker addresses and target: these are the labels

Labeled windows collected so far (93 in total): `syn-flood-001`, `http-flood-001`, `slow-attack-001` (attacks), `normal-001`, `pic-legitime-001` (legitimate spike).

**Public datasets** (flow graph only, mainly for the graph and external-node heads): CIC-DDoS2019, CIC-IDS2017, UNSW-NB15, CTU-13.

**Splitting:** by scenario, never by window, to avoid data leakage. Some attack types and botnet sizes are held out for generalization tests.

> **Safety:** all attacks are generated only inside the isolated lab (closed network, Kubernetes NetworkPolicies). Nothing is ever directed at an outside machine, third-party service or shared network.

## Evaluation

Detection precision/recall/F1, **false-positive rate on legitimate spikes**, macro-F1 per attack type, precision/recall on sources, Top-1/Top-3 on the target, detection delay, generalization to unseen types/sizes/datasets, and inference time (must stay below the window length). Plain accuracy is not used because normal traffic dominates.

**Baselines the GNN must beat:** threshold rules (requests/s per source and target) and a per-flow Random Forest / XGBoost, especially on below-threshold attacks and false positives.

## Extension: correlating two Suricata IDS probes

Two probes see different parts of the same attack and raise duplicate alerts with different IPs and timestamps.

| Probe | Placement | Sees | Detects well |
|---|---|---|---|
| IDS 1 (perimeter) | In front of the ingress (north-south) | Real external IPs, incoming rate | Volumetric floods, SYN flood, scans |
| IDS 2 (internal) | Pod network (east-west) | Clear HTTP after the ingress, errors | HTTP flood, slow attacks, internal effects |

The ingress terminates connections, so IDS 2 no longer sees attacker IPs and the Community ID alone is not enough. Correlation also relies on the target, the time, and the `X-Forwarded-For` header.

`correlation_ids_suricata(alertes_ids1, alertes_ids2, metriques=None, fenetre="10s", mode="regles")` outputs correlated DoS/DDoS incidents (type, target, start, end, probes, sources, confidence, explanation).

Common pipeline: collection (Filebeat/Vector) → normalization → DoS filtering → deduplication (`community_id`) → windowing. Then one of:

| | `mode="regles"` | `mode="ia"` |
|---|---|---|
| Method | Explicit rules R1 to R6 (grouping, convergence, cross-confirmation, source matching, typing, closing) | Level 1: XGBoost per window + DBSCAN; Level 2: alerts as GNN graph attributes |
| Training data | None | Labeled lab scenarios |
| Explanation | Triggered rules | SHAP, GAT attention, GNNExplainer |
| Weakness | Fixed thresholds, misses slow/below-threshold attacks | Higher compute cost |

Specific metrics: compression ratio, incident purity, incident precision/recall, false positives on spikes, detection delay, and an ablation (IDS 1 only, IDS 2 only, both).

## Tech stack

`Python` · `PyTorch Geometric` · `scikit-learn` · `XGBoost` · `SHAP` · `pandas / networkx` · `Kubernetes (kind)` · `Docker` · `Cilium / Hubble` · `Prometheus / Grafana` · `Suricata` · `Locust` · `hping3 / k6` · `Streamlit`

## Getting started


```bash
git clone https://github.com/Rihemfathallah/projet-gnn-suricata.git
cd projet-gnn-suricata
pip install -r requirements.txt

make scenario     # run an attack/normal scenario in the lab
make process      # build labeled windows / graphs
make api          # serve the correlation function

streamlit run 1_infrastructure/demo/app.py   # demo
```

## Results

> Add your key numbers (F1, false positives on spikes, Top-1 target, detection delay) compared with the threshold and Random Forest baselines, plus 1 or 2 screenshots (Grafana, Hubble, Streamlit demo).

## My contributions


- Built the isolated lab (kind, Cilium/Hubble, default-deny policy, Online Boutique, Prometheus/Grafana)
- Wrote the SYN flood, HTTP flood and slow-attack scenarios and the labeled-window export
- Worked on the model, training and results slides
- [add anything else]

## Roadmap

1. Environment: isolated lab and monitoring
2. Data: normal traffic, legitimate spikes, attack scenarios, labeled windows
3. Baselines: threshold rules and Random Forest
4. GNN function: PyTorch Geometric, multi-head training, GraphSAGE comparison, metrics
5. Finalization: GNNExplainer, packaging, Streamlit demo, report
