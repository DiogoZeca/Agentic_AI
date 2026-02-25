# Agentic AI — Background Research

This file contains background research concepts that are relevant to the project direction.
It is reference material, not a requirements document. Nothing here has been confirmed by EVIDEN.

---

## What Makes a System "Agentic"

A regular system (like this API) is request/response — one call in, one answer out. An agentic system introduces an autonomous loop: the AI reasons, uses tools, observes results, and iterates until a task is complete.

**Key properties an agentic system needs:**
- **Tool use** — calls external functions (APIs, databases)
- **Autonomy** — decides what to do next without human input per step
- **Multi-step reasoning** — iterates until the task is done
- **Memory** — maintains context across steps

**This project is not agentic yet.** It is a deterministic REST API. Making it agentic would mean adding a reasoning layer on top of the API that decides which endpoints to call and why.

---

## The ReAct Pattern (the standard single-agent loop)

```
User query
    ↓
THOUGHT  — LLM reasons about what it needs
ACTION   — LLM calls a tool (e.g. get_anomalies())
OBSERVATION — tool returns result
    ↓
[loop back to THOUGHT, or emit FINAL ANSWER]
```

If this project were to add an agent layer, this is the pattern it would use — the agent would call the API endpoints as tools and reason over the results.

---

## MCP (Model Context Protocol)

The standard protocol for connecting AI agents to external tools. Anthropic-originated, now under the Linux Foundation (AAIF). Supported by Anthropic, OpenAI, Google, Microsoft, AWS.

**What it does:** Lets an AI agent call tools exposed by an MCP server using a standardised protocol (JSON-RPC 2.0 over STDIO or HTTP).

**Why it is relevant here:** If this project exposes its API endpoints as MCP tools, any MCP-compatible AI (Claude Code, Claude Desktop, others) can query the forecasting service natively without custom integration code.

**This has not been built yet and has not been requested by EVIDEN.**

---

## Real Data — What Exists for Energy Monitoring in K8s

These are tools that exist for measuring energy at infrastructure level. Relevant when real data replaces synthetic data.

| Tool | What it measures | Level |
|------|-----------------|-------|
| **RAPL** (Intel/AMD) | CPU socket power | Node (hardware counter) |
| **NVIDIA DCGM** | GPU power per card | Per GPU |
| **Kepler** (CNCF) | Pod-level energy (attributed from RAPL by CPU fraction) | Per pod (estimated) |
| **cAdvisor** | CPU, memory per container | Per pod (actual computation) |
| **node_exporter** | Node-level CPU, memory, disk, network | Per node |

**Key limitation:** True pod-level energy does not exist as a direct measurement. Kepler estimates it by distributing node-level RAPL readings across pods by resource fraction.

---

## Carbon Intensity Data Sources

| Provider | What | Frequency |
|----------|------|-----------|
| **Electricity Maps** | Grid carbon intensity by region | 5–15 min |
| **WattTime** | Marginal emissions rate | 5–15 min |

Neither has been integrated. Carbon intensity is currently simulated in `data_generator.py`.

---

## SCI Standard

ISO/IEC 21031:2024, Green Software Foundation.

```
SCI = (E × I + M) / R
```

Already implemented in this project. See RESUME.md for details.
