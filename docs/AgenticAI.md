# Agentic AI & Orchestration — Research Notes

Deep research on agentic AI orchestration, multi-agent architectures, and their intersection with power consumption and carbon emissions monitoring for AI infrastructure.

---

## 1. What Makes AI "Agentic"

Regular AI operates as single request/response — one prompt in, one answer out. **Agentic AI** introduces an autonomous loop where the AI reasons, uses tools, observes results, and iterates until the task is complete.

### The ReAct Pattern (Reasoning + Acting)

The dominant single-agent architecture, introduced by Yao et al. (2022):

```
User Query
    |
    v
+---------------------------+
|   THOUGHT (Reasoning)     |  LLM analyzes the query + all prior context.
|   "I need to look up X"  |  Produces a natural-language reasoning step.
+---------------------------+
    |
    v
+---------------------------+
|   ACTION (Tool Call)      |  The LLM emits a structured output (JSON)
|   search("X")            |  specifying which tool to call and with
+---------------------------+  what arguments.
    |
    v
+---------------------------+
|   OBSERVATION (Result)    |  The tool executes and returns its result.
|   "X is 42.7 kWh"       |  This is appended to the conversation context.
+---------------------------+
    |
    v
[Loop back to THOUGHT, or emit FINAL ANSWER]
```

The entire history of Thoughts, Actions, and Observations forms the agent's **scratchpad** — a growing context that serves as both memory and dynamic plan.

**Stop conditions:**
- The LLM emits a "final_answer" instead of "tool_call"
- A maximum iteration limit is reached (e.g., 10 loops)
- An explicit "FINISH" prefix is recognized by the orchestrator

### Other Planning Patterns

- **Plan-then-Execute:** A planner LLM decomposes the task into sub-steps upfront, then a separate executor carries them out sequentially. Optional re-planning after each step.
- **Iterative Refinement:** Generator creates a draft, Critique Agent provides notes, Refinement Agent polishes. Loops until quality threshold is met.
- **Reflexion:** After each action, a reflection step evaluates whether the approach is working. If not, the agent modifies its strategy.

### What Makes an Agent "Agentic"

Key properties:
- **Autonomy** — decides what to do next without human intervention
- **Tool use** — calls external functions (APIs, databases, code execution)
- **Planning** — decomposes complex tasks into sub-steps
- **Memory** — maintains context across multiple reasoning steps
- **Multi-step reasoning** — iterates until the task is complete

---

## 2. Multi-Agent Orchestration Patterns

### 2.1 Supervisor Pattern (Hierarchical)

```
                    [Supervisor Agent]
                    /       |        \
                   v        v         v
            [Worker A] [Worker B] [Worker C]
            (Research)  (Analysis)  (Writing)
```

- A single "boss" agent receives the task, decomposes it, and routes sub-tasks to specialized workers
- Workers execute and return results to the supervisor
- The supervisor synthesizes results and decides if more work is needed

**Strengths:** Clear control flow, easy to debug, good for sequential reasoning.
**Weaknesses:** Supervisor becomes a bottleneck; every decision flows through one LLM call.
**Best for:** Structured workflows — financial analysis, compliance checks, step-by-step pipelines.

### 2.2 Swarm / Peer Pattern (Decentralized)

```
[Agent A] <--handoff--> [Agent B] <--handoff--> [Agent C]
    ^                                                |
    |________________handoff________________________|
```

- Agents are peers with defined roles, no central coordinator
- Coordination through **handoffs**: Agent A calls `transfer_to_agent_b()`, passing control and context
- Convergence is emergent — agents propose, challenge, and refine through rules and time limits

**Strengths:** No single point of failure, natural for conversational routing, scales for exploratory tasks.
**Weaknesses:** Harder to debug, potential for infinite handoff loops.
**Best for:** Web research, customer service routing, exploratory problem-solving.

### 2.3 Pipeline / DAG Pattern (Directed Workflow)

```
[Data Collector] --> [Analyzer] --> [Forecaster] --> [Report Generator]
                         |
                         +--> [Anomaly Detector] --> [Alert Agent]
```

- Agents arranged as nodes in a Directed Acyclic Graph (DAG)
- Each agent passes output to the next in sequence
- Independent branches execute in parallel
- Dynamic DAG restructuring (2025+): agents can modify workflow in real-time

**Strengths:** Linear, deterministic, easy to debug, supports parallel execution.
**Weaknesses:** Rigid if not using dynamic restructuring.
**Best for:** Data processing pipelines, ETL workflows, structured analysis chains.

### 2.4 Mixture of Agents (MoA)

```
Layer 1:  [LLM-A] [LLM-B] [LLM-C]   (Proposers — generate diverse responses)
              \       |       /
               v      v      v
Layer 2:  [LLM-D] [LLM-E] [LLM-F]   (Aggregators — refine using all Layer 1 outputs)
              \       |       /
               v      v      v
Layer 3:      [Final Aggregator]       (Synthesizes final answer)
```

- Multiple LLMs reason independently, subsequent layers see ALL previous outputs
- LLMs exhibit **collaborativeness**: better responses when they see outputs from other models
- MoA using only open-source LLMs achieved 65.1% on AlpacaEval 2.0 vs. 57.5% by GPT-4o (ICLR 2025)

**Best for:** Maximum accuracy, diverse reasoning, consensus-building.

### When to Use Which

| Pattern | Best When | Avoid When |
|---------|-----------|------------|
| **Supervisor** | Well-defined tasks, need central control | Tasks are exploratory, supervisor bottleneck |
| **Swarm/Peer** | Conversational routing, coverage matters | Need deterministic ordering, strict audit trail |
| **Pipeline/DAG** | Clear data flow, parallel steps | Open-ended tasks, dynamic re-planning needed |
| **Mixture of Agents** | Maximum accuracy, latency acceptable | Real-time responses, cost-sensitive |
| **Hybrid** | Most production systems | Simple single-agent tasks |

---

## 3. State, Memory & Checkpointing

### Short-Term Memory (Conversation Context)

The growing list of messages (user, assistant, tool calls, observations) within the current session. This IS the context window.

Task state flows through every node as a typed dictionary. LangGraph uses **reducer logic** to merge updates:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add]       # Append-only
    current_plan: str                     # Overwrite: latest replaces previous
    tool_results: Annotated[list, add]   # Append-only
    iteration_count: int                  # Overwrite
```

### Long-Term Memory (Persistent)

- **Episodic memory:** Records of past interactions, decisions, outcomes. Stored in PostgreSQL, SQLite, or vector stores.
- **Semantic memory:** Domain knowledge, user preferences, learned patterns. Vector databases (Pinecone, Weaviate, ChromaDB) with embedding-based retrieval (RAG).

### Shared State Between Agents

- **LangGraph:** Single shared state object flows through all nodes. Reducer pattern prevents conflicts.
- **AutoGen/AG2:** Agents communicate through asynchronous messages, maintaining local state.
- **Blackboard pattern:** Shared data structure all agents read/write. Agents monitor for relevant updates.

### Checkpointing and Recovery

After each node execution, the framework saves a state snapshot to a persistence backend (PostgreSQL, Redis, S3). On failure, the workflow resumes from the last checkpoint.

Enables:
- **Time-travel debugging** — replay execution from any checkpoint
- **Human-in-the-loop** — pause at specific nodes for approval, then resume

---

## 4. Tool Integration

### MCP (Model Context Protocol)

Now the **de facto industry standard** for connecting AI agents to external tools. Originally from Anthropic (Nov 2024), donated to the **Agentic AI Foundation (AAIF)** under the Linux Foundation (Dec 2025), co-founded by Anthropic, Block, and OpenAI, with support from Google, Microsoft, AWS.

**Architecture:**

```
+------------------------------------------+
|  MCP Host (AI Application)               |
|  e.g., Claude Code, VS Code, IDE         |
|                                          |
|  [MCP Client 1] ----> [MCP Server A]    |  (Local: filesystem, via STDIO)
|  [MCP Client 2] ----> [MCP Server B]    |  (Local: database, via STDIO)
|  [MCP Client 3] ----> [MCP Server C]    |  (Remote: API, via HTTP/SSE)
+------------------------------------------+
```

**Protocol:** JSON-RPC 2.0

**Three server primitives:**
- **Tools:** Executable functions the AI can invoke (`tools/list`, `tools/call`)
- **Resources:** Read-only data sources (files, database records, API responses)
- **Prompts:** Reusable interaction templates

**Two client primitives:**
- **Sampling:** Servers request LLM completions from the host
- **Elicitation:** Servers request information from the user

**Transports:**
- **STDIO:** Local servers via subprocess stdin/stdout. Zero network overhead.
- **Streamable HTTP:** Remote servers via HTTP POST + Server-Sent Events. Supports OAuth/tokens.

### Function Calling APIs

LLM-native mechanism for tool use (OpenAI, Anthropic, Google). The LLM outputs structured JSON specifying which function to call and with what arguments. The orchestrator executes and feeds results back.

### Code Execution Sandboxes

- **E2B:** Firecracker microVMs (~150ms startup), Python/JS SDKs
- **Modal:** Serverless compute, GPU support (H200, H100), sub-second cold starts

---

## 5. Power Consumption of Agentic AI

### The Amplification Problem

A single user request to an AI agent can trigger:
- **Multiple LLM calls** — planning, reasoning, tool selection, output synthesis, self-critique
- **Tool execution** — code execution, web searches, database queries
- **Retry loops** — failed tool calls, re-planning, iterative refinement
- **Long-running sessions** — minutes to hours rather than milliseconds

### Quantitative Evidence

| Workload | Energy per request |
|----------|-------------------|
| Google search | ~0.3 Wh |
| ChatGPT single query (OpenAI claim) | ~0.34 Wh |
| Google Gemini text query | ~0.24 Wh |
| DeepSeek R1 (varies with reasoning depth) | 0.96-3.74 Wh |
| GPT-4.5 complex/long prompt | ~30 Wh |
| **Claude Code session (median)** | **~41 Wh** |
| **Claude Code full work day (power user)** | **~1,300 Wh** |

**Key findings:**
- Agent teams use **~7x more tokens** than single-agent sessions (LangChain State of Agent Engineering report)
- Energy per token: ~3-4 Joules
- All generative AI queries combined consumed **15 TWh in 2025**, projected **347 TWh by 2030**
- Inference (not training) drives most growth

### Efficiency Countermeasures

- **Model routing:** Dispatch simpler sub-tasks to smaller models — up to **70x** energy reduction per query
- **Token efficiency:** Claude Opus 4.5 achieves higher pass rates with **up to 65% fewer tokens**
- **Location optimization:** Choosing optimal data center can reduce carbon by **up to 50x**
- **Aggressive caching:** Cache tool outputs and intermediate reasoning
- **Plan-then-execute:** Create full plans before execution to minimize LLM round-trips

---

## 6. Carbon-Aware Computing for AI

### What It Means

Software that adjusts behavior based on grid carbon intensity: **do more when electricity is clean, defer when dirty.**

Data center energy demand projected to reach **1,000 TWh by 2026** (~Japan's total consumption). US data centers: 176 TWh (4.4% of national demand), projected 580 TWh (12%) by 2028.

### Real-Time Carbon Data Sources

| Provider | Data | Update Frequency |
|----------|------|-----------------|
| **WattTime** | Marginal Operating Emissions Rate (MOER) | 5-15 min |
| **Electricity Maps** | Grid carbon intensity by region | 5-15 min |

### Carbon-Aware Scheduling — Real Projects

| Project | Approach | Result |
|---------|----------|--------|
| **GREEN** (NSDI 2025) | ML cluster scheduler, temporal shifting | **41.2% carbon reduction**, 12% peak power reduction, 3.6-5.9% latency overhead |
| **CASPER** | Kubernetes load balancer, spatial+temporal shifting | **Up to 70% carbon reduction** across regions |
| **Carbon Aware KEDA Operator** (Azure) | Autoscales K8s workloads by carbon intensity | No code changes needed |
| **Eco-Orchestrator** | RL-based (CARL algorithm), aligns GPU power with grid carbon | **34.7% emission reduction**, PUE from 1.58 to 1.12 |
| **Emerald AI trial** (May 2025) | Task flexibility tiers (critical vs. deferrable) | **25% power reduction** during peak grid demand |

### SCI Standard (ISO/IEC 21031:2024)

```
SCI = ((E * I) + M) / R

E = Energy consumed by the software system
I = Location-based marginal carbon emissions for the grid
M = Embodied carbon (hardware manufacturing, disposal)
R = Functional unit (e.g., per inference, per agent session, per user request)
```

The **SCI for AI** extension (Green Software Foundation, 2025) adds two measurement boundaries:
- **Provider score:** Model development, training, deployment efficiency
- **Consumer score:** Operational impacts from inference and monitoring

---

## 7. Power Monitoring Tools for Real AI Infrastructure

### Hardware-Level

| Tool | Measures | Level | Notes |
|------|----------|-------|-------|
| **RAPL** (Intel) | CPU/DRAM power | Per-socket | Software model via hardware counters, not physical meter. Linux: `/sys/class/powercap/intel-rapl/` |
| **NVIDIA DCGM** | GPU power, utilization, temperature, energy | Per-GPU | Production standard. Integrates with Prometheus, Datadog. |
| **NVIDIA Power Profiles** (Blackwell, 2025) | Max-Q (efficiency) / Max-P (performance) modes | Per-GPU | Up to **15% energy savings** at >97% performance |

### Software-Level

| Tool | Approach | Strengths | Best For |
|------|----------|-----------|----------|
| **CodeCarbon** | Python lib; GPU+CPU+RAM power estimation; regional carbon intensity | Most accurate vs. wattmeter; easy integration | Measurement and reporting |
| **CarbonTracker** | Python lib; tracks power over time; forecasts future mid-training | Can halt/pause long runs based on carbon forecasts | Predictive control during training |
| **Kepler** (CNCF Sandbox) | eBPF-based; Prometheus exporter | **Container/pod-level** attribution in Kubernetes | Production K8s observability |
| **Scaphandre** | System-level monitoring agent | Lightweight; bare-metal and VM | Non-K8s deployments |

**Combined strategy:** CarbonTracker for predictive control during training, CodeCarbon for measurement, Kepler for container-level observability in production Kubernetes.

### Infrastructure-Level

- **PDUs and smart meters:** Rack-level and circuit-level power
- **PUE (Power Usage Effectiveness):** Total facility power / IT equipment power. Best-in-class: 1.1-1.2
- **NVIDIA 800 VDC architecture:** Up to 5% end-to-end power efficiency improvement

---

## 8. Orchestration Frameworks Comparison

| Framework | Best For | Key Strength |
|-----------|----------|-------------|
| **LangGraph** (LangChain) | Production systems, fine-grained control | Stateful graphs, checkpointing, time-travel debugging |
| **CrewAI** | Rapid prototyping of role-based teams | Crews + Flows two-layer architecture, 200-400ms latency |
| **AutoGen/AG2** (Microsoft) | Research, complex iterative dialogues | Message-passing, enterprise-grade observability |
| **OpenAI Agents SDK** | OpenAI ecosystem, handoff-based routing | Simple Agent + Handoff primitives, production-ready |
| **Anthropic Agent SDK** | Claude ecosystem | Claude-native tool use |
| **AWS Strands Agents** | AWS-native deployments | Advanced orchestration with AWS service integration |
| **Google ADK** | Google Cloud deployments | Eight built-in multi-agent patterns |

---

## 9. Practical Architecture: Carbon-Aware AI Infrastructure Orchestration

### System Overview

A system that monitors AI infrastructure, forecasts power consumption, and autonomously adjusts workload scheduling to minimize carbon emissions.

**Pattern:** Supervisor + DAG Hybrid

```
                        +---------------------------+
                        |   Orchestrator Agent      |
                        |   (Supervisor + Planner)  |
                        +---------------------------+
                           /      |       |       \
                          v       v       v        v
                 +--------+ +--------+ +--------+ +--------+
                 |Monitor | |Forecast| |Carbon  | |Schedule|
                 |Agent   | |Agent   | |Agent   | |Agent   |
                 +--------+ +--------+ +--------+ +--------+
                     |           |          |          |
                     v           v          v          v
                 [DCGM /   [Prophet+  [WattTime/ [Kubernetes
                  Prom /    TimesFM    Elec.Maps  Job Sched.
                  Kepler]   Ensemble]  API]       GPU Freq.]
```

### Agent Definitions

**Orchestrator Agent (Supervisor)**
- Receives high-level goals ("minimize carbon for next 24h")
- Decomposes into sub-tasks, delegates to specialists, synthesizes results
- Re-plans every scheduling cycle (e.g., hourly)
- LLM: High-capability model (Claude Opus or equivalent)

**Infrastructure Monitor Agent**
- Continuously monitors GPU utilization, power draw, PUE, cooling, inference rates, training status
- Pattern: DAG pipeline — sensors → aggregation → anomaly detection
- MCP servers: `nvidia-dcgm-server`, `prometheus-server`, `database-server`
- Output: Current state written to shared metrics store

**Forecast Agent**
- Runs power/carbon/SCI forecasts using Prophet + TimesFM ensemble
- Pattern: Single-agent ReAct loop
- Tools: `run_prophet_forecast()`, `run_timesfm_forecast()`, `run_ensemble_forecast()`, `evaluate_accuracy()`
- Output: Predicted `totalEnergyConsumption`, `carbonEmissions`, `sciPerInference` for next N hours

**Carbon Intelligence Agent**
- Monitors real-time grid carbon intensity, tracks renewable availability
- MCP servers: `electricity-maps-server`, `weather-api-server`
- Output: Carbon-optimal time windows for next 24 hours

**Workload Scheduler Agent**
- Adjusts scheduling to minimize carbon while meeting SLAs
- Pattern: Plan-then-execute with constraints
- Tools: `kubernetes-server`, `job-scheduler`, `batch-size-optimizer`, `gpu-frequency-scaler`
- Decision logic:
  ```
  IF forecast.carbon_intensity[t+1] > threshold AND job.is_deferrable:
      defer_job(job, to=next_low_carbon_window)
  IF current.gpu_utilization < 30% AND grid.carbon_intensity > 400 gCO2/kWh:
      scale_down_cluster(target_utilization=60%)
  IF renewable_availability[t+2] > 80%:
      schedule_training_job(start=t+2)
  ```

### One Scheduling Cycle

```
1. Orchestrator: "Begin hourly optimization cycle"

2. --> Monitor Agent: "Report current infrastructure state"
   Calls nvidia_dcgm_server.get_gpu_metrics()
   Calls prometheus_server.query("power_consumption_total")
   Writes: {gpu_util: 72%, power_draw: 3.2kW, pue: 1.35, training: active}

3. --> Carbon Agent: "Report carbon conditions"
   Calls electricity_maps_server.get_intensity("region-1")
   Calls weather_api.get_solar_forecast(next_24h)
   Writes: {carbon_intensity: 320 gCO2/kWh, low_windows: [02:00-06:00, 14:00-16:00]}

4. --> Forecast Agent: "Forecast next 24h"
   Calls run_ensemble_forecast("totalEnergyConsumption", periods=24)
   Calls run_ensemble_forecast("carbonEmissions", periods=24)
   Writes: {forecast_power: [...], forecast_emissions: [...]}

5. Orchestrator reviews:
   "Training job at 22:00 coincides with high carbon intensity"
   "Inference drops 40% at 02:00 during low-carbon window"

6. --> Scheduler Agent: "Defer training to 02:00, scale inference down at 22:00"
   Calls job_scheduler.reschedule(job_id="train-42", start="02:00")
   Calls kubernetes_server.scale(deployment="inference", replicas=2, at="22:00")

7. Orchestrator: Logs decisions, saves checkpoint, next cycle in 1 hour
```

### State Management

```python
class InfrastructureState(TypedDict):
    # Short-term: current cycle
    current_metrics: dict              # Latest GPU/power/PUE readings
    carbon_conditions: dict            # Grid intensity, renewable forecasts
    forecasts: dict                    # Prophet+TimesFM predictions
    scheduling_decisions: list         # Actions taken this cycle
    cycle_number: int

    # Long-term (persisted to PostgreSQL)
    historical_accuracy: list          # Forecast vs actual for retraining
    cumulative_carbon_saved: float     # Running total of emissions avoided
    workload_patterns: dict            # Learned patterns for better scheduling
```

### MCP Server Configuration

```
MCP Servers consumed by agents:
  nvidia-dcgm-mcp-server        (local, STDIO)   -- GPU metrics
  prometheus-mcp-server          (local, STDIO)   -- infrastructure metrics
  electricity-maps-mcp-server   (remote, HTTP)    -- carbon intensity API
  kubernetes-mcp-server         (local, STDIO)    -- cluster management
  postgresql-mcp-server         (local, STDIO)    -- time-series data store

MCP Server exposed by this system:
  carbon-optimizer-mcp-server   (remote, HTTP)    -- allows other AI systems
                                                     to query optimization
                                                     recommendations
```

### Resilience

- **Checkpointing:** Full state saved to PostgreSQL after each agent step. Resume from last checkpoint on failure.
- **Fallback:** Ensemble fails → Prophet-only → simple moving average.
- **Human-in-the-loop:** Pause before changes affecting >50% cluster capacity.
- **Observability:** All decisions, tool calls, state transitions logged with trace IDs.

---

## 10. Carbon-Structured Agentic Architecture

*Inspired by the delay-structured Agentic Core for 6G networks (Corici et al., Fraunhofer FOKUS / TU Berlin), adapted for carbon-aware AI infrastructure orchestration.*

### The Core Principle: Carbon Containment

The Fraunhofer paper proposes organizing network agents by **delay domains** — each agent operates within a timing budget it can reliably satisfy, and slow agents can never interfere with fast control paths. This prevents latency amplification.

We adapt this principle for carbon: **organize agents by carbon-impact authority**. Each layer has a carbon budget — how much emission impact its decisions can create. The golden rule:

> Agents can only influence layers at their carbon-impact level or below. Upstream propagation toward lower-impact layers is explicitly avoided. This prevents emission amplification.

### Carbon Authority Layers

```
Layer    Budget Scope          Timescale         Role
─────────────────────────────────────────────────────────────────
L4       Monthly/Quarterly     hours → days      Strategic Carbon Management
         carbon targets                          Sets SCI targets, carbon caps, compliance
                                                 Example: "Reduce SCI 15% this quarter"
                    │
                    ▼ parameterizes
L3       Daily carbon          minutes → hours   Grid-Aware Planning
         allocation                              Determines green windows, training schedules
                                                 Example: "Defer training to 10:00-15:00 (solar)"
                    │
                    ▼ constrains
L2       Hourly energy         seconds → mins    Infrastructure Optimization
         envelope                                Cluster scaling, PUE, cooling modes
                                                 Example: "Scale to 2 GPUs when carbon > 400g"
                    │
                    ▼ bounds
L1       Per-job energy        sub-sec → secs    Workload Control
         allocation                              GPU frequency, batch size, job priority
                                                 Example: "Run at 80% TDP, batch=32"
                    │
                    ▼ limits
L0       Per-request           milliseconds      Inference Routing
         energy cap                              Model selection, early stopping
                                                 Example: "Route to efficient model, SCI < 2g"
```

### Concept Mapping: Network Delay → Carbon Impact

| Delay-Structured (Network) | Carbon-Structured (AI Infra) |
|---|---|
| Control-loop delay | Carbon-loop impact |
| Timing budget per layer | Carbon budget per layer |
| Delay containment | Carbon containment |
| Latency amplification | Emission amplification |
| SLA (latency guarantee) | SCI target (carbon guarantee) |
| NWDAF analytics function | Prophet + TimesFM forecasting |
| Policy-based NF control | Carbon-budget-based workload control |
| Deterministic NFs | Deterministic workload policies |
| A2A peer coordination | Agent coordination via A2A/MCP |

### Why Carbon Domains Are Easier Than Delay Domains

Three structural advantages over the network domain:

1. **Slower timescales.** Network control operates in microseconds. Carbon optimization operates in seconds to days. An L3 agent can run a 30-second forecast without breaking anything — in a network, that would be catastrophic.

2. **More predictable patterns.** Network traffic spikes unpredictably. Energy consumption follows strong daily/weekly seasonality (Prophet captures this at ~14% sMAPE). Carbon intensity follows solar/wind patterns. Forecasting-based control is highly viable.

3. **Standardized accountability metric.** SCI (ISO/IEC 21031:2024) provides `SCI = ((E × I) + M) / R` — a per-functional-unit carbon score analogous to per-request latency SLA. The metric exists and is standardized.

### Carbon Guardrails Per Layer

The paper stresses guardrails must scale with the timing budget. For carbon:

| Layer | Guardrail Type | Mechanism | Latency |
|-------|---------------|-----------|---------|
| L0-L1 | Threshold check | "Is energy-per-inference within cap?" | Microseconds |
| L2 | Range validation | "Does scaling keep hourly energy in budget?" | Milliseconds |
| L3 | **Forecast validation** | "What does Prophet+TimesFM predict if we defer?" | Seconds |
| L4 | **Simulation/digital twin** | "Will this policy achieve the SCI target?" | Minutes |

The forecast models (Prophet+TimesFM ensemble) serve as **carbon guardrails for L3 decisions** — validating that a scheduling decision will actually reduce emissions before it's executed.

### Incremental Migration Path

Adapted from the paper's 4 deployment stages:

**Stage 1 — Carbon analytics outside the control path** *(current state)*
- Prophet+TimesFM run as standalone scripts, human reads insights
- Zero risk, zero carbon automation, pure analysis

**Stage 2 — Carbon-aware decisions embedded in workload managers**
- Forecast agent suggests scheduling changes (advisory)
- Existing schedulers incorporate carbon awareness internally
- Control semantics unchanged; carbon influence is indirect

**Stage 3 — Carbon-aware agent collectives**
- Monitor, Forecast, Carbon, Scheduler agents coordinate via A2A/MCP
- External interfaces preserved (same job submission APIs)
- Internal decision-making distributed across carbon-authority layers

**Stage 4 — Full carbon-native orchestration**
- All workload control is carbon-aware by default
- Dynamic control loops across carbon-impact layers
- SCI guarantees enforced at every level
- Only viable where carbon containment is guaranteed

### Where Our Forecast Models Sit

| Model | Carbon Layer | Role |
|-------|-------------|------|
| Prophet + trainingActive regressor | L3 Planning | Predict energy 24h ahead, identify training windows |
| TimesFM zero-shot | L3-L4 Planning/Strategic | Handle nonlinear carbon interactions |
| Ensemble (Prophet + TimesFM) | L3 Planning | Best energy forecasting via residual correction |
| Component-based SCI derivation | L4 Strategic | Carbon-per-request ratio for SCI compliance |

### Protocol Selection Per Layer

Following the paper's analysis of SBA vs A2A vs MCP:

| Layer | Protocol | Rationale |
|-------|----------|-----------|
| L0-L1 (fast, deterministic) | Direct function calls | No coordination overhead, bounded logic |
| L2 (coordination) | A2A | Peer communication between infrastructure agents |
| L3-L4 (planning/strategic) | **MCP** | Shared carbon models, forecast results, grid data |
| L4-L5 (admin/business) | MCP + A2A | Intent propagation, compliance reporting |

---

## 11. Key Open-Source Projects & References

### Research Papers

| Paper | Venue | Key Finding |
|-------|-------|-------------|
| GREEN | NSDI 2025 | Carbon-efficient ML cluster scheduler; 41.2% emission reduction |
| CASPER | IGSC 2023 / arXiv 2024 | Carbon-aware K8s load balancer; 70% reduction |
| Federated Carbon Intelligence | MRS Energy & Sustainability 2025 | Real-time optimization across heterogeneous hardware |
| Sustainable Carbon-Aware LLM Scheduling | GLSVLSI 2025 | Joint carbon and water optimization for geo-distributed LLM serving |
| Mixture-of-Agents | ICLR 2025 | Multiple LLMs collaborating outperform single models |
| How Do Agentic AI Systems Deal With Energy Concerns? | arXiv 2025 | Energy awareness analysis in agentic codebases |
| From Functions to Agents: Delay-Aware Agentic Core for 6G | Fraunhofer FOKUS / TU Berlin 2025 | Carbon-authority layers concept; incremental migration from NFs to agents |

### Open-Source Tools

| Project | Purpose | Maturity |
|---------|---------|----------|
| Carbon Aware SDK (GSF) | Carbon data aggregation from WattTime/Electricity Maps | Graduated; v1.3 |
| SCI for AI (GSF) | Standardized AI carbon measurement | Public specification |
| Kepler | eBPF container energy monitoring | CNCF Sandbox |
| CodeCarbon | Python CO2 tracking | Mature; widely adopted |
| CarbonTracker | ML training carbon + forecasting | Active |
| CASPER | Carbon-aware K8s load balancer | Research prototype |
| Carbon Aware KEDA Operator (Azure) | Carbon-aware K8s autoscaling | Azure-backed |
| NVIDIA DCGM | GPU fleet telemetry | Production-grade |
| Scaphandre | System-level energy monitoring | Active |
| LangGraph | Multi-agent orchestration framework | Production; LangChain |
| MCP (Anthropic / AAIF) | Universal agent-tool protocol | Industry standard |

### Key Sources

- Simon P. Couch: Electricity Use of AI Coding Agents (Jan 2026) — first detailed measurement of agentic energy consumption
- LangChain: State of Agent Engineering — agent teams use 7x more tokens
- Green Software Foundation: SCI specification (ISO/IEC 21031:2024) and SCI for AI extension
- Google Cloud: Choose a Design Pattern for Agentic AI Systems
- Microsoft Azure: AI Agent Orchestration Patterns
- O'Reilly: Designing Effective Multi-Agent Architectures
- IBM: What is a ReAct Agent?
- Together AI: Mixture-of-Agents (ICLR 2025)
- MCP Specification (modelcontextprotocol.io)
- Corici et al.: From Functions to Agents — Delay-aware Agentic Core for 6G (Fraunhofer FOKUS, 2025) — architectural inspiration for carbon-structured agent layers

