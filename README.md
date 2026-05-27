# The Reasoning Tax: Token Economics of LLM Reasoning Across Task Types and Deployment Contexts

**Sachin Gopal Wani** · Lenovo Infrastructure Solutions Group  
Planning to publish in: *Lecture Notes in Computer Science*, Springer Nature  
Venue: TPCTC 2026 (co-located with VLDB 2026)

> **Abstract.** Reasoning-capable large language models improve accuracy on difficult tasks but often do so by generating long thinking chains, creating a deployment trade-off that accuracy-only benchmarks do not capture. We introduce the Token Economy Score (TES), a marginal efficiency metric that measures the accuracy gain of a reasoning model over a non-reasoning baseline, normalized by the generated-token multiplier. We define paired and approximated variants of TES for controlled model comparisons and frontier models without direct non-reasoning counterparts. Across seven benchmarks spanning various task types , we find that task structure predicts reasoning value better than nominal difficulty. Sequential inference-chain tasks such as AIME 2025 and LiveCodeBench show high TES, while knowledge-recall tasks such as MMLU-Pro show low TES despite their difficulty. We also find systematic diminishing returns at higher reasoning effort levels, including cases where additional thinking reduces accuracy. Finally, we introduce Reasoning Cost Share and Deployment Cost Multiplier to characterize inference-cost composition and on-premises deployment economics. The results suggest that reasoning should be enabled selectively by task type, effort level, and deployment context rather than treated as a universally beneficial mode.
---

## Repository contents

```
.
├── data/
│   └── actuals_final.csv          # Full evaluation dataset (151 model-benchmark runs)
├── analysis/
│   └── tes_analysis.py            # All metric calculations and figure generation
├── figures/
│   ├── figure1_tes_formula.svg    # TES formula and interpretation diagram
│   ├── figure2_tes_distribution.svg
│   ├── figure3_diminishing_returns.svg
│   ├── figure4_rcs_stacked.svg
│   └── figure5_quadrant.svg
├── output/                        # Generated CSVs from tes_analysis.py
│   ├── tes_pairs.csv
│   ├── benchmark_summary.csv
│   ├── rcs_table.csv
│   ├── dcm_table.csv
│   ├── diminishing_returns.csv
│   ├── tes_a_gemini.csv
│   ├── per_question_tokens.csv
│   └── quadrant_data.csv
└── README.md
```

---

## Dataset

`data/actuals_final.csv` contains 151 evaluation runs across 27 model configurations and 7 benchmarks. Each row represents one model variant on one benchmark.

### Fields

| Field | Description |
|---|---|
| `model` | Model name as listed on Artificial Analysis |
| `family` | Model family (GPT, Claude, DeepSeek, Qwen, Gemini, Grok, GLM, Gemma) |
| `variant` | Specific model size or version within the family |
| `pair_id` | Join key linking reasoning and non-reasoning variants of the same model |
| `benchmark` | One of: IFBench, MMLU-Pro, GPQA Diamond, AIME 2025, LiveCodeBench, HLE, CritPt |
| `thinking` | 1 = reasoning enabled, 0 = non-reasoning / instruct mode |
| `score` | Accuracy on the benchmark (0–1 scale) |
| `input_tokens` | Mean input tokens per benchmark run |
| `reasoning_tokens` | Mean reasoning / thinking chain tokens (0 for thinking=0 rows) |
| `output_tokens` | Mean output tokens (final response only, excluding reasoning) |
| `total_tokens` | Sum of input + reasoning + output tokens |
| `price_per_million_input` | Cloud API price per million input tokens (USD) at time of collection |
| `price_per_million_output` | Cloud API price per million output tokens (USD) at time of collection |
| `cost_input` | Computed input token cost per benchmark run (USD) |
| `cost_reasoning` | Computed reasoning token cost per benchmark run (USD) |
| `cost_output` | Computed output token cost per benchmark run (USD) |
| `cost_total` | Total cloud API cost per benchmark run (USD) |
| `on_prem_cost` | On-premises cost per benchmark run (USD); 0 if not evaluated on-prem |
| `Ratio (api_v_prem)` | DCM: cloud cost / on-prem cost |
| `deployment_context` | `cloud_api` or `on_prem` |
| `source` | `artificial_analysis` or `self_run` |

### Benchmarks

| Benchmark | Domain | Questions | Difficulty |
|---|---|---|---|
| IFBench | Instruction following | 58 | Medium |
| MMLU-Pro | Knowledge recall | 12,000 | Medium–Hard |
| GPQA Diamond | Science reasoning | 198 | Hard |
| AIME 2025 | Mathematics | 30 | Hard |
| LiveCodeBench | Code generation | Continuous | Medium–Hard |
| HLE | Expert multi-domain | 2,500 | Very Hard |
| CritPt | Research physics | 71 | Extreme |

### Model families

GPT (5.2, 5.5), Claude (Opus 4.5/4.7, Sonnet 4.5/4.6), DeepSeek (V3.2, V4 Pro), Qwen (3-32B, 3-235B-A22B, 3.5-397B-A17B, 3.6-27B), Gemini (3 Flash, 3 Pro, 3.1 Pro, 3.5 Flash), Grok (4.1 Fast, 4.2, 4.3), GLM (4.7, 5.0, 5.1), Gemma (4 31B, 4 26B-A4B), Kimi (K2.6).

On-premises evaluations were conducted on a system with 8×NVIDIA B300 GPUs (FP16) for Qwen and Gemma families. Cloud API data sourced from Artificial Analysis (https://artificialanalysis.ai/evaluations) and self-run evaluations on IFBench, GPQA Diamond, AIME 2025, and HLE.

---

## Reproducing the analysis

### Requirements

```bash
pip install pandas numpy matplotlib openpyxl
```

### Running

```bash
python analysis/tes_analysis.py --csv data/actuals_final.csv
```

All output CSVs are written to `output/`. All figures are written to `figures/` as SVG.

To skip figure generation and run metric calculations only:

```bash
python analysis/tes_analysis.py --csv data/actuals_final.csv --no-plots
```

### Key outputs

| File | Contents |
|---|---|
| `output/tes_pairs.csv` | TES-Δ for every paired model comparison |
| `output/benchmark_summary.csv` | Mean ± std TES per benchmark (Table 3 in paper) |
| `output/rcs_table.csv` | Reasoning Cost Share per reasoning model run |
| `output/dcm_table.csv` | Deployment Cost Multiplier per on-premises model |
| `output/diminishing_returns.csv` | Marginal TES across reasoning effort levels |
| `output/tes_a_gemini.csv` | TES-A values for unpaired Gemini models |
| `output/per_question_tokens.csv` | Per-question token averages for agentic analysis |
| `output/quadrant_data.csv` | Data underlying Figure 5 quadrant plot |

---

## Metrics defined

### Token Economy Score (TES)

$$\text{TES}(M_r, M_i, T) = \frac{[\text{Acc}(M_r, T) - \text{Acc}(M_i, T)] \times 100}{\text{GenTok}(M_r, T) \;/\; \text{GenTok}(M_i, T)}$$

where $\text{GenTok}(M, T) = \text{reasoning_{tokens}} + \text{output_{tokens}}$ (input tokens excluded).

- **TES-Δ**: $M_i$ is the same model architecture with reasoning disabled
- **TES-A**: $M_i$ is the best-performing non-reasoning model on benchmark $T$

**Interpretation:** TES > 1 = strong, 0 < TES ≤ 1 = marginal, TES ≤ 0 = wasteful.

### Reasoning Cost Share (RCS)

$$\text{RCS}(M_r, T) = \frac{\text{cost_reasoning}}{\text{cost_total}} \times 100$$

The percentage of total inference cost consumed by the thinking chain.

### Deployment Cost Multiplier (DCM)

$$\text{DCM}(M, T) = \frac{\text{cost_total_cloud}}{\text{cost_total_on\text{-}prem}}$$

How many times cheaper the same workload is on owned hardware versus cloud API.

---

## Key findings

1. **Task structure governs TES more strongly than difficulty.** AIME 2025 (sequential math inference) achieves mean TES of 6.27 ± 2.77; MMLU-Pro (knowledge recall) achieves 0.65 ± 0.56, despite being academically comparable in difficulty.

2. **Diminishing returns are systematic and can be negative.** Increasing reasoning effort from moderate to maximum frequently yields near-zero or negative marginal TES. DeepSeek V4 Pro on GPQA Diamond scores 1.7 percentage points *lower* at maximum effort than at high effort while consuming 2.74× more tokens.

3. **Reasoning tokens dominate inference cost.** Median RCS across reasoning models is 94.6%. Fifteen model-benchmark pairs exhibit RCS exceeding 99% — less than 1% of inference spend produces the visible answer.

4. **On-premises MoE deployment changes the economics.** DCM values of 13–26× for mixture-of-experts architectures on the 8×B300 system mean that models in the high-cost cloud quadrant (Q2) become cost-viable on-premises. Dense models show DCM of 2–4×; Qwen3.6-27B shows DCM of 20× due to cloud API pricing premiums on a recently released model.

---

## Citation

```bibtex
@inproceedings{wani2026reasoningtax,
  title     = {The Reasoning Tax: Token Economics of {LLM} Reasoning Across Task Types and Deployment Contexts},
  author    = {Wani, Sachin Gopal},
  booktitle = {Lecture Notes in Computer Science},
  publisher = {Springer Nature},
  year      = {2026},
  note      = {TPCTC 2026, co-located with VLDB 2026}
}
```

---

## License

Data and code released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Please cite the paper if you use this dataset or analysis code in your research.

---

## Contact

Sachin Gopal Wani · Lenovo Infrastructure Solutions Group  
For questions about the dataset or methodology, please open an issue in this repository.
