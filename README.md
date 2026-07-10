# FinCrime Agents: A Lean Multi-Agent Investigator, Built on LangGraph

> A multi-agent financial-crime triage system that investigates flagged transactions — sanctions
> screening, transaction-history analysis, external context — and returns an auditable, structured
> decision. The reasoning core is a **single small open-weight LLM** (Qwen2.5-3B-Instruct), adapted
> with **DoRA-SFT + GRPO (RLVR)** rather than prompted at frontier-model scale, orchestrated end to
> end as a **LangGraph** `StateGraph` (multi-node routing, subgraphs, human-in-the-loop, persistent
> memory) rather than a single ReAct loop.

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Status](https://img.shields.io/badge/status-early%20stage-orange)

---

## TL;DR — the thesis

Revolut's own public material describes a **FinCrime AI Agents Platform**: several agents doing
sanctions screening, transaction-history analysis and web search/customer contact, built for
observability, auditability and governance, with "staged evaluations" (cheap structured signals
narrow the case before expensive large-context reasoning) and LoRA fine-tuning over open-weight
models to control inference cost. None of that is public as code — it's a description, not a repo.

This project is an open-source, portfolio-scale reconstruction of those *capabilities*, using
**LangGraph** as the orchestration engine and **one small local model** instead of a fleet of
frontier ones. It is also the direct sequel to
[`lean-fraud-detection`](../lean-fraud-detection): the cheap TCN there scores every transaction and
flags ~0.5% as alerts — this project is what happens *after* the flag, investigating the case and
deciding `block / review / allow` with a written rationale.

**Why a small model, adapted, instead of a big one, prompted:** the same "efficiency beats scale"
methodology as the sibling project (TCN vs. Transformer, ~6× fewer params, no quality loss) applied
to agentic reasoning — here the axis isn't parameter count (the model is fixed-size) but **compute
per decision**: fewer tool calls and tokens to reach the same, verifiably correct, decision.

## Architecture

```
                     intake (plain Python — cheap "staged evaluation" filter)
                         │
             risk low ───┴─── risk medium/high
                │                   │
                ▼                   ▼
           auto_allow           dispatch (supervisor — decides which specialists to run)
              (end)                 │
                        ┌────────────┼────────────────┐
                        ▼            ▼                ▼
                 sanctions_agent  transaction_agent  web_search_agent
                 (ReAct + tools)  (ReAct + tools)    (conditional — only if inconclusive)
                        │            │                │
                        └────────────┴────────────────┘
                                     ▼
                         synthesize (supervisor — structured decision)
                                     │
                        decision == review ──► human_in_the_loop (interrupt())
                                     │
                                     ▼
                                 narrator (writes the case summary)
                                     │
                                     ▼
                       persist (checkpoint per case + long-term memory per entity)
```

Built as a single `StateGraph`, not a flat ReAct loop, specifically to exercise the LangGraph
primitives this project exists to practice:

| Concept | Where it shows up |
|---|---|
| Typed shared state | Pydantic/TypedDict state carrying the alert, evidence, and decisions across nodes |
| Conditional edges | `intake`'s cheap/expensive routing; `synthesize`'s `review` branch |
| Subgraphs (supervisor pattern) | Three specialist ReAct subgraphs orchestrated by a parent graph |
| Parallel fan-out / fan-in | `dispatch` invokes specialists concurrently, `synthesize` merges their evidence |
| `ToolNode` / tool-calling | Each specialist's read-only tools over the case data |
| Structured output | `with_structured_output` for both the per-specialist findings and the final decision |
| `interrupt()` (human-in-the-loop) | `review`-tier decisions pause the graph for approval before continuing |
| Checkpointer (short-term memory) | Per-case state persisted across a run |
| Store (long-term memory) | Cross-case memory keyed by entity, so a repeat offender's history persists |

## Agents

One shared model, four LLM-invoking roles (different system prompt + tool bindings, not four
separate models — loading several models at once doesn't fit a 6GB GPU, and a single adapted
backbone serving multiple tasks is the same design PRAGMA itself makes):

| Agent | Role | Tools |
|---|---|---|
| **Supervisor / Orchestrator** | Routes to specialists, synthesizes the final `block/review/allow` decision + rationale | — (reasons over specialist outputs) |
| **Sanctions Screening** | Resolves fuzzy name matches against a real sanctions list (the false-positive problem naive string matching can't solve) | sanctions list lookup, entity disambiguation |
| **Transaction History Analyst** | Velocity/pattern analysis against the card's own baseline | `get_card_profile`, `get_recent_transactions`, `get_population_fraud_rate` |
| **Customer Contact / Web Search** | External context when the above are inconclusive (only invoked conditionally) | web search (pluggable: real API or offline stub) |

## Model & training

- **Base model:** Qwen2.5-3B-Instruct (already tool-call trained), 4-bit (QLoRA). Fallback if VRAM
  doesn't allow it: Qwen3-1.7B with FP8 GRPO — a combination already proven to fit ~5GB.
- **Adaptation, one recipe, no ablation branches:** **DoRA-SFT (warm start) → GRPO refinement**,
  trained with [Unsloth](https://unsloth.ai) for the VRAM headroom (target hardware: a 6GB GPU).
  - DoRA over plain LoRA: decomposes the update into magnitude + direction, closing more of the gap
    to full fine-tuning than vanilla LoRA at the same low rank (`r=16`) — the exact regime a 6GB
    budget forces.
  - The RL reward is **verifiable, not learned** (RLVR): decision correctness is checked
    programmatically against ground-truth labels, plus a bonus for fewer tool calls/tokens and a
    format-validity term — no separate reward model to train or host.
  - Grammar-constrained decoding on the structured-output nodes takes most of the "valid format"
    burden off the reward function, so GRPO's budget goes to decision quality and efficiency, not to
    learning JSON syntax.
- **SFT data:** on the order of 1,000–2,000 curated (alert → tool calls → decision + rationale)
  trajectories distilled from a stronger teacher and kept only when independently verifiable against
  ground truth — quality over volume.

## Data (public only)

- **Case alerts:** a static fixture derived from `lean-fraud-detection`'s Sparkov test split (its
  `is_fraud=1` cases plus hard negatives near the TCN's decision boundary), committed here so this
  repo is runnable standalone — narratively "what the cheap model flagged," without a runtime
  dependency on the sibling repo.
  - Case alerts are synthetically augmented so a small fraction resemble sanctions-relevant cases
    (a cardholder name perturbed toward a real sanctions-list entry, or a hard-negative lookalike),
    giving the sanctions agent verifiable positive/negative cases to reason over.
- **Sanctions list:** the OFAC SDN list (U.S. Treasury, public domain) — downloaded by a setup
  script, not committed as data.

## Status

**Early stage — design complete, implementation not started.** This README documents the intended
architecture, agent set, and training approach as a spec for the build. Nothing here is a claimed
result yet; a results table (decision accuracy, tool-call/token efficiency, latency, vs. an untrained
baseline) will replace this section once there's a trained checkpoint to report on.

## Working notes

- Public portfolio repo: public datasets and a public sanctions list only, no employer-internal
  references.
- Code, comments, and README in English.
- Sibling repo: [`lean-fraud-detection`](../lean-fraud-detection) (the cheap TCN this project's
  alerts come from).

## Author

Alejandro Torres — AI/ML Engineer focused on time-series anomaly detection, model efficiency, and (here) applying that same efficiency lens to agentic LLM systems.

## License

MIT
