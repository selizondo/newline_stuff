# Mini-Project 1. Home DIY Repair Q&A Synthetic Data Generator

## 🎯 Project Goal

Build an automated pipeline that generates high-quality synthetic Q\&A data for a Home DIY Repair assistant. The pipeline must generate structured repair guidance, detect quality failures using an LLM-as-Judge, analyze failure patterns, and iteratively correct the generation prompts until the failure rate drops by **more than 80%** compared to the baseline.

**Core Challenge**: Create a system that not only generates realistic DIY repair data but also understands what makes repair guidance "good" or "bad", and then proves it works by demonstrating measurable, data-driven improvement through prompt correction.

***

## 🧠 The Problem Context

Training a reliable DIY repair assistant requires large amounts of accurate, safe, and practical Q\&A data. Manually authoring thousands of repair scenarios is expensive and slow. The solution is **synthetic data generation**: using LLMs to produce the training data automatically.

The challenge: LLMs don't always produce useful repair advice. They might give dangerously incomplete electrical guidance, recommend exotic tools a homeowner couldn't possibly own, or produce vague tips that don't actually help anyone fix anything.

Your job is to build a system that:

* **Generates** diverse DIY repair Q\&A pairs at scale
* **Validates** the structure of each generated item
* **Detects** quality failures using a second LLM acting as a judge
* **Analyzes** failure patterns to understand what's going wrong
* **Corrects** the generation prompts based on failure analysis
* **Demonstrates** measurable improvement in a final evaluation run

This mirrors a real-world MLOps workflow (generate, evaluate, diagnose, and fix) applied to a data generation pipeline.

***

## 🔁 System Architecture Overview

Your pipeline should consist of **seven sequential phases**:

**Phase 1 (Generation)**: Diverse prompt templates → LLM → Structured Q\&A items

**Phase 2 (Structural Validation)**: Pydantic schema checks → Filter malformed items

**Phase 3 (Failure Labeling, LLM-as-Judge)**: Independent LLM evaluator → Binary pass/fail per failure mode

**Phase 4 (Quality Evaluation, LLM-as-Judge)**: Score each item across all 8 quality dimensions → Compare against benchmark

**Phase 5 (Failure and Quality Analysis)**: Aggregate failure rates + quality scores → Heatmaps → Identify worst-performing areas

**Phase 6 (Prompt Correction and Re-evaluation)**: Improved prompts → Re-run full pipeline → Compare before/after

**Phase 7 (Benchmark Comparison Report)**: Run judge on benchmark sample → Calibrate → Final quality gap analysis

Each phase should be independently runnable and produce output that feeds the next phase. The system should support re-running just Phase 6 after prompt corrections without regenerating the baseline data.

**Judge Calibration Step (Phase 7)**: Before trusting your judge's scores on your generated data, you must validate the judge itself. Run your LLM-as-Judge on a random sample of at least 50 items from the benchmark dataset. If your judge fails more than 5% of benchmark items on any quality dimension, your judge criteria are miscalibrated and must be adjusted. This ensures your quality evaluation is measuring the right things.

***

## Benchmark Dataset

You are provided with a **benchmark dataset** available on Hugging Face: [dipenbhuva/home-diy-repair-qa](http
s://huggingface.co/datasets/dipenbhuva/home-diy-repair-qa). This dataset contains **5,000 high-quality DIY repair Q\&A items** produced and curated by the course instructors.

**Purpose**: This dataset serves as your quality benchmark. Your generated data will be evaluated **against this reference**, not in isolation. The benchmark defines what "good enough" looks like, and your pipeline must demonstrate that its output approaches or matches this level of quality.

**What makes the benchmark dataset the standard**:

* Every item has a substantial, narrative-style answer (typically 700-1,300 characters) that weaves together the tools, steps, safety warnings, and tips into a coherent response, not just a list of fields stitched together
* Safety information is always **specific to the hazards of the particular repair** (e.g., "Turn off the breaker before removing the outlet cover plate", not "Be careful" or "Stay safe")
* Tips provide **non-obvious, task-specific advice** that a beginner would not know (e.g., "Remove the painter's tape immediately after smoothing. If you wait until the caulk sets, the tape will tear the caulk edge")
* Tools listed are items a typical homeowner would realistically own or could purchase at a hardware store
* Steps are concrete and specific enough to follow without guessing. They include quantities, measurements, or observable indicators where relevant
* All 5 repair categories are represented with equal coverage (1,000 items each)

**How you will use it**:

* **Compare against it** after generation. Your LLM-as-Judge must evaluate your generated items against these quality standards
* **Measure the gap**. Your final report must include a quantitative comparison between your generated dataset and the benchmark

You are **not** expected to replicate this dataset. You are expected to generate data that meets the same quality standards it demonstrates.

***

## 📐 Data Quality Requirements

Your generated dataset is not just evaluated on structural correctness (valid JSON, required fields present). It must meet **semantic quality requirements** that reflect what makes repair guidance actually useful to a homeowner. These requirements are derived from the benchmark dataset and define the bar your generation pipeline must clear.

### The 8 Quality Dimensions

Every generated Q\&A item will be evaluated across these 8 dimensions. Your LLM-as-Judge must score each dimension independently.

| #  | Quality Dimension            | Requirement                                                                                                                                                                                                                                                                    | How to Measure                                                                                              |
| -- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| Q1 | **Answer Coherence**         | The `answer` field must read as a complete, natural response, not a mechanical concatenation of the other fields. It should integrate tools, steps, safety, and tips into a narrative a homeowner could follow top-to-bottom.                                                  | LLM judge evaluates whether the answer reads as a unified response vs. a disjointed list                    |
| Q2 | **Step Actionability**       | Each step must be specific enough that a person unfamiliar with the repair could execute it without guessing. Steps must include observable outcomes, quantities, or measurements where relevant (e.g., "tighten hand-tight plus a quarter turn", not "tighten until secure"). | LLM judge checks each step for vague language ("properly", "as needed", "until done") and missing specifics |
| Q3 | **Tool Realism**             | Every tool listed must be something a typical homeowner either already owns or could buy at a general hardware store for under \$50. No professional, specialty, or trade-only tools.                                                                                          | LLM judge flags tools that require professional purchase or cost over \$50                                  |
| Q4 | **Safety Specificity**       | Safety information must name **the specific hazard** of this repair and **the specific precaution** to take. Generic warnings ("be careful", "use caution", "stay safe") are failures. Safety info must be at least 80 characters long.                                        | LLM judge + character length check                                                                          |
| Q5 | **Tip Usefulness**           | Tips must provide non-obvious, task-specific advice that adds value beyond the steps. A tip that merely restates a step, or offers generic encouragement, is a failure.                                                                                                        | LLM judge evaluates whether each tip provides information not already covered in the steps                  |
| Q6 | **Problem-Answer Alignment** | The answer must directly address the specific problem described in `equipment_problem`. An answer about general maintenance when the problem is a specific symptom is a failure.                                                                                               | LLM judge checks whether the answer resolves the stated problem                                             |
| Q7 | **Appropriate Scope**        | The repair must be within realistic DIY capability. If professional help is genuinely needed (e.g., gas line work, electrical panel replacement), the answer should say so clearly rather than providing dangerous amateur instructions.                                       | LLM judge evaluates whether the repair scope matches homeowner skill level                                  |
| Q8 | **Category Accuracy**        | The `category` field must correctly match the repair domain. A plumbing repair tagged as `electrical_repair` is a failure.                                                                                                                                                     | Rule-based keyword check + LLM judge                                                                        |

### Quality Thresholds

Your generated dataset must meet these minimum quality thresholds when evaluated by your LLM-as-Judge:

| Threshold                                             | Target             |
| ----------------------------------------------------- | ------------------ |
| Answer Coherence pass rate                            | ≥ 90% of items     |
| Step Actionability pass rate                          | ≥ 85% of items     |
| Tool Realism pass rate                                | ≥ 95% of items     |
| Safety Specificity pass rate                          | ≥ 90% of items     |
| Tip Usefulness pass rate                              | ≥ 85% of items     |
| Problem-Answer Alignment pass rate                    | ≥ 95% of items     |
| Appropriate Scope pass rate                           | ≥ 95% of items     |
| Category Accuracy pass rate                           | ≥ 98% of items     |
| **Overall quality pass rate** (all 8 dimensions pass) | **≥ 80% of items** |

These thresholds are calibrated against the benchmark dataset. If your LLM-as-Judge consistently fails benchmark items on a dimension, your judge (not the benchmark) needs recalibration.

***

## 📊 Success Metrics

| Metric                                  | Target                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Baseline failure rate (initial prompts) | Establish a baseline ≥ 15% overall failure rate                                                         |
| Post-correction failure rate            | **≤ 80% of the baseline** (i.e. >80% reduction)                                                         |
| Structural validation pass rate         | ≥ 95% of all generated items pass schema validation                                                     |
| Coverage across repair categories       | All 5 repair domains must be represented in generated data                                              |
| Failure modes detected                  | All 6 defined failure modes must be measurable                                                          |
| Quality dimensions evaluated            | All 8 quality dimensions must be scored by your LLM-as-Judge                                            |
| Overall quality pass rate               | ≥ 80% of generated items pass all 8 quality dimensions                                                  |
| Benchmark comparison                    | Your judge must score a sample of benchmark items and achieve ≥ 95% pass rate (judge calibration check) |
| Minimum dataset size                    | At least 50 Q\&A pairs generated per pipeline run                                                       |

The core deliverable is the **before/after comparison**: your corrected prompts must demonstrate a statistically meaningful drop in failure rate across all 6 failure dimensions. Additionally, your final corrected dataset must meet the quality thresholds defined in the Data Quality Requirements section, as validated by your LLM-as-Judge and compared against the benchmark dataset.

***

## 🛠 Technical Requirements

### Required Technology Stack

* **Python 3.10+**: Core language
* **Pydantic**: Schema validation with detailed error reporting
* **Instructor**: Structured LLM outputs
* **LLM Provider**: Any OpenAI-compatible API (GPT-4o, Claude, Gemini, Groq, etc.)
* **Matplotlib / Seaborn**: Failure heatmap and visualization
* **No hardcoded repair answers**: all content must be LLM-generated at runtime

### Optional Enhancements

* **Pandas**: Data manipulation and aggregation
* **Braintrust**: Evaluation tracking and logging across runs
* **Logfire**: Observability and tracing for LLM calls
* **Pre-commit hooks**: Code quality (Black, Ruff, MyPy)

***

## 📏 Data Models & Interfaces

Every generated Q\&A item must conform to a structured schema. At minimum, each record must contain:

| Field               | Type            | Description                                                   |
| ------------------- | --------------- | ------------------------------------------------------------- |
| `question`          | string          | A realistic DIY repair question from a homeowner              |
| `answer`            | string          | A clear, actionable answer with step-by-step guidance         |
| `equipment_problem` | string          | The specific problem being addressed (e.g. "dripping faucet") |
| `tools_required`    | list of strings | Tools a typical homeowner would realistically own             |
| `steps`             | list of strings | Ordered, numbered repair steps                                |
| `safety_info`       | string          | Relevant safety warnings and precautions                      |
| `tips`              | list of strings | Practical tips to make the repair easier or more reliable     |

Here is what a correctly generated Q\&A item should look like with all 7 fields populated:

```
{
  "question": "How do I fix a leaky faucet?",
  "answer": "Detailed step-by-step answer explaining the repair process...",
  "equipment_problem": "Kitchen faucet with dripping water from the spout",
  "tools_required": ["adjustable wrench", "screwdriver", "plumber's tape"],
  "steps": [
    "Turn off the water supply valves under the sink",
    "Remove the faucet handle by unscrewing the decorative cap and handle screw",
    "Replace the worn washer or O-ring inside the valve seat",
    "Reassemble the handle and turn the water supply back on"
  ],
  "safety_info": "Always turn off the water supply before starting. Place a towel under the sink to catch residual water.",
  "tips": "Apply plumber's tape clockwise around threaded connections to ensure a watertight seal."
}
```

**Validation rules** (enforced at schema level):

* `question` and `answer` must be non-empty strings
* `steps` must contain at least 3 items
* `tools_required` must contain at least 1 item
* `tips` must contain at least 1 item
* `safety_info` must be present (not empty)

**Generation Logs** (tracked per item, not part of final dataset) (For Iteration Logs):

* Which prompt template was used
* Whether the item passed structural validation
* Which failure modes were flagged by the judge
* Timestamp and model used

***

## 🧪 Key Implementation Challenges

### 1. Prompt Diversity Without Repetition

Your generator must cover 5 distinct repair categories. Using a single generic prompt produces homogeneous, low-quality data. You need distinct prompt templates, each with its own expert persona, vocabulary, and context, to generate realistic variety.

The 5 required categories are:

* **Appliance Repair**: refrigerators, washing machines, dryers, dishwashers, ovens
* **Plumbing Repair**: leaks, clogs, fixture repairs, pipe problems
* **Electrical Repair**: outlet replacement, switch repair, light fixture installation (safe homeowner-level work only)
* **HVAC Maintenance**: filter changes, thermostat issues, vent cleaning, basic troubleshooting
* **General Home Repair**: drywall, doors/windows, flooring, basic carpentry

Each run should randomly select a category per item so that over 20+ samples, all 5 are naturally represented.

**Why it matters**: A bias toward one repair type creates an unbalanced dataset that won't generalize to real homeowner questions.

### 2. Detecting the 6 Failure Modes

Your LLM-as-Judge must independently evaluate each generated item for all 6 failure modes:

| Failure Mode               | Description                                                            |
| -------------------------- | ---------------------------------------------------------------------- |
| `incomplete_answer`        | Answer lacks enough detail to actually complete the repair             |
| `safety_violations`        | Missing or incorrect safety warnings for hazardous tasks               |
| `unrealistic_tools`        | Requires professional or specialized tools not found in a typical home |
| `overcomplicated_solution` | Recommends professional service for a straightforward DIY task         |
| `missing_context`          | Question or answer lacks the context needed to understand the problem  |
| `poor_quality_tips`        | Tips are vague, generic, or unhelpful ("be careful", "good luck")      |

The judge must produce a binary score (0 = pass, 1 = fail) for each failure mode independently. An item is considered "failed overall" if it has **any** failure flag.

Here is what the judge output should look like for a single evaluated item, with one score per failure mode plus a derived `overall_failure` flag:

```
{
  "trace_id": "qa_003",
  "incomplete_answer": 0,
  "safety_violations": 1,
  "unrealistic_tools": 0,
  "overcomplicated_solution": 0,
  "missing_context": 0,
  "poor_quality_tips": 1,
  "overall_failure": true,
  "quality_scores": {
    "answer_coherence": 1,
    "step_actionability": 1,
    "tool_realism": 1,
    "safety_specificity": 0,
    "tip_usefulness": 0,
    "problem_answer_alignment": 1,
    "appropriate_scope": 1,
    "category_accuracy": 1
  },
  "quality_pass": false
}
```

Note that the judge output now includes **both** the 6 failure mode flags **and** the 8 quality dimension scores (1 = pass, 0 = fail). An item has `quality_pass: true` only if all 8 quality dimensions pass. The failure modes and quality dimensions are evaluated independently. An item can pass all failure modes but still fail on quality dimensions (e.g., no safety violations detected, but the safety info is too vague to pass the Safety Specificity quality check).

**Why it matters**: Each failure mode has a different root cause and requires a different prompt correction strategy. The quality dimensions add a second layer of evaluation that ensures your data isn't just "not broken" but is genuinely useful.

### 3. Failure Pattern Analysis & Heatmap

After labeling, you must aggregate the results to understand:

* Which failure modes are most frequent?
* Are certain failure modes correlated? (e.g. do items with `missing_context` also tend to have `incomplete_answer`?)
* Which repair category (prompt template) produces the most failures?

You must produce a **failure heatmap**, a matrix visualization showing failure mode co-occurrence across items. This is your primary diagnostic tool.

**Why it matters**: Correlated failures suggest a shared root cause in the prompt, not an isolated issue.

### 4. Prompt Correction Strategy

Based on the failure analysis, you must modify your generation prompts to address the identified weaknesses. Your corrections should be targeted and traceable. You should be able to explain _why_ each change was made based on the data.

**Why it matters**: Blind prompt tweaking is not engineering. Data-driven prompt correction is the skill being practiced here.

### 5. Reliable Structured Output from LLMs

LLMs don't always return valid JSON. Your pipeline must handle malformed responses gracefully by retrying when possible, logging failures, and never crashing on a single bad output.

**Why it matters**: Production data pipelines must be robust. Fragile generators can't be trusted at scale.

***

## 📦 Deliverables

* **Working pipeline**: A Python script (or set of modules) that runs all 7 phases end-to-end
* **Generated dataset**: At least 50 validated Q\&A pairs in JSON/JSONL format from the baseline run
* **Failure analysis report**: A summary (printed or saved) showing:&#x20;
* Per-mode failure rates for baseline run
* Per-dimension quality scores for baseline run
* Failure heatmap visualization
* Quality dimension heatmap visualization
* Identified failure patterns and their suspected causes
* **Corrected prompts**: Updated prompt templates with documented changes
* **Before/after comparison**: Failure rates and quality scores from baseline vs. corrected run, showing >80% failure reduction and meeting quality thresholds
* **Benchmark comparison report**: Results from running your judge on a sample of benchmark items, demonstrating judge calibration (≥ 95% pass rate on benchmark)
* **Quality gap analysis**: A quantitative comparison between your final corrected dataset and the benchmark across all 8 quality dimensions
* **README**: Brief instructions on how to run the pipeline and interpret the results

***

## 🎨 Visualization Requirements

All charts must be generated using **Matplotlib**, **Seaborn**, or **Plotly**. Save each chart as a PNG file in a `visualizations/` directory. Your heatmap and charts must reveal actionable insights, not just display numbers:

* **Failure Mode Co-occurrence**: Which failure modes tend to appear together on the same item? (e.g., does `missing_context` correlate with `incomplete_answer`?)
* **Failure Rates by Repair Category**: Which prompt template (appliance, plumbing, electrical, HVAC, general) produces the most failures?
* **Per-Mode Failure Trend**: Before vs. after comparison for each failure mode individually, not just overall rate
* **Most Problematic Items**: Identify items with 3+ simultaneous failure flags. These are your worst cases and best diagnostic targets
* **Quality Dimension Scores**: A bar chart or radar chart showing pass rates across all 8 quality dimensions for your generated data, before and after prompt correction
* **Benchmark vs. Generated Comparison**: A side-by-side visualization comparing quality dimension pass rates between the benchmark (as scored by your judge) and your generated dataset

**Quality Standard**: All visualizations should include clear titles, labeled axes, appropriate color scales (diverging for correlation, sequential for rates), and legible font sizes. A chart that can't be read at a glance is not a finished chart.

***

## 🔄 Prompt Correction Strategy

After analyzing the failure heatmap, you must improve the generation prompts in a structured, documented way:

* **Identify the dominant failure modes**: which mode has the highest rate in the baseline?&#x20;
* **Find correlated failures**: if two modes always appear together, they likely share a root cause&#x20;
* **Pinpoint the responsible template(s)**: which repair category produces the most failures of each type?&#x20;
* **Write targeted corrections**: modify only the parts of the prompt that address the identified problem; don't rewrite everything at once&#x20;
* **Document every change**: `Change: Added explicit "include at least one safety precaution" instruction Reason: safety_violations failure rate was 42% in baseline Template affected: electrical_repair`&#x20;
* **Re-run and compare**: generate a new batch with corrected prompts, re-run the judge, compute improvement ratio&#x20;

**Success Criteria**: >80% reduction in overall failure rate. Corrections must be traceable to specific failure data, not intuition.

***

## 📋 Iteration Logs

Maintain a structured log for each prompt correction cycle or pipeline change. Use this format:

```
### Iteration N: [Brief title]
- **Date**: YYYY-MM-DD
- **Change**: What you changed from the previous iteration
- **Hypothesis**: Why you expected this change to help
- **Result**: Quantitative outcome (failure rates, quality scores)
- **Decision**: Keep / revert / modify further
- **Next step**: What to try next based on this result
```

Example entries:

| Iteration | Change                                           | Overall Failure Rate | Quality Pass Rate | Decision                             |
| --------- | ------------------------------------------------ | -------------------- | ----------------- | ------------------------------------ |
| 1         | Baseline prompts (no corrections)                | 32%                  | 61%               | Establish baseline                   |
| 2         | Added safety instructions to electrical template | 24%                  | 68%               | Keep, safety\_violations dropped 50% |
| 3         | Added context requirements to all templates      | 18%                  | 74%               | Keep, missing\_context dropped 40%   |
| 4         | Restructured steps format with numbered lists    | 12%                  | 82%               | Keep, meets target                   |

***

## 🎯 Evaluation Approach

Your system will be evaluated by running it fresh and examining the outputs. The evaluator will:

* **Run the baseline pipeline**: Generate ≥ 50 items and record per-mode failure rates and quality dimension scores
* **Check structural validation**: Confirm ≥ 95% of items pass schema validation
* **Review the heatmaps**: Confirm they correctly visualize failure co-occurrence and quality dimensions
* **Verify judge calibration**: Run the student's judge on benchmark items and confirm ≥ 95% pass rate
* **Run the corrected pipeline**: Generate another ≥ 50 items with improved prompts
* **Compute the improvement ratio**:

```
improvement = (baseline_failure_rate - corrected_failure_rate) / baseline_failure_rate
```

**Pass threshold: improvement ≥ 0.80** (i.e. failure rate drops by at least 80%)

* **Check quality thresholds**: Verify the corrected dataset meets ≥ 80% overall quality pass rate across all 8 dimensions
* **Review benchmark comparison**: Confirm the quality gap analysis is quantitative and meaningful
* **Inspect prompt corrections**: Verify that changes are data-driven and documented, not random tweaks

A pipeline that generates clean data but shows no improvement does not pass. A pipeline that achieves >80% improvement but crashes unpredictably does not pass. A pipeline whose judge fails to calibrate against the benchmark does not pass. If your judge can't recognize good data, its scores on your data are meaningless.

### Self-Evaluation Questions

* Can you explain _why_ a specific item was flagged with `safety_violations` or `incomplete_answer`?
* Do your visualizations reveal non-obvious patterns (e.g., one repair category failing much more than others)?
* Are your prompt corrections clearly linked to specific failure data, or did you just guess?
* Does your judge give consistent results if you run it twice on the same item?
* Are failure modes distributed differently across the 5 repair categories, or are they uniform?
* Does your judge pass ≥ 95% of benchmark items? If not, what does that tell you about your evaluation criteria?
* Which quality dimensions show the largest gap between your generated data and the benchmark? Why?

***

## 💡 First Principles

**Why use LLMs to generate training data?** Manual data creation is expensive. LLMs can produce hundreds of varied examples per hour at low cost. But they introduce their own failure modes, which is exactly why automated quality evaluation is necessary.

**Why LLM-as-Judge instead of rule-based checks?** Many quality issues (vague tips, overcomplicated solutions, missing safety warnings) require semantic understanding to detect. A rule that checks "does the answer contain the word 'safety'" doesn't catch the difference between a real warning and "be safe out there!"

**Why measure all 6 failure modes separately?** Aggregate quality scores hide the structure of the problem. If 80% of your failures come from one mode, that's a targeted prompt fix. If failures are evenly distributed, you have a deeper generation problem.

**Why calibrate the judge against a benchmark?** An LLM judge can be too strict or too lenient. Without a known-good reference, you have no way to tell. Running the judge on benchmark items reveals whether your evaluation criteria actually match reality. If good data fails your judge, the judge is broken.

**Why do prompt corrections need to be data-driven?** Intuitive prompt tweaks are hard to reproduce and often introduce new problems while fixing old ones. Data-driven corrections are traceable, explainable, and give you a feedback loop.

***

## 🌟 Bonus Challenges (Optional)

If you complete the core requirements and want to go further:

### 1. **Difficulty-Tiered Generation**

Add a `difficulty` field (beginner / intermediate / advanced) to each Q\&A item, and verify that more complex repairs produce proportionally more `unrealistic_tools` or `overcomplicated_solution` failures. Does the difficulty tier correlate with failure patterns?

### 2. **Auto-Correction of Individual Items**

Instead of only correcting prompts, implement a second pass where the LLM regenerates only the flagged fields of a failed item (e.g., rewrite just the `tips` field for items with `poor_quality_tips`). Track the per-item correction success rate.

### 3. **Evaluation Tracking Across Runs**

Integrate Braintrust or a simple JSON log to track failure rates across multiple pipeline runs, so you can visualize the improvement trend over several correction iterations, not just a single before/after.

### 4. **Safety-Aware Generation Rules**

Add a post-generation rule check (separate from the LLM judge): any item mentioning electrical work or gas lines must contain specific safety keywords. Compare this rule-based safety check to the LLM judge's `safety_violations` score. Where do they agree or disagree?

***

## 🚀 Getting Started Hints

### Recommended Development Order

* **Study the benchmark**: Read through 20-30 items from the benchmark dataset before writing any code. Understand what quality looks like.
* **Start with schemas**: Define Pydantic models with all validation rules
* **Build one generator**: Get a single template working end-to-end before building all five
* **Implement validation**: Ensure you can catch and categorize schema errors
* **Add the LLM judge**: Start with one failure mode, confirm it works, then add the other five. Then add the 8 quality dimensions.
* **Calibrate the judge**: Run it on benchmark items early. If it fails good data, fix the judge before proceeding.
* **Create the heatmaps**: Prove your labeling and quality scoring systems work visually
* **Write correction strategy**: Analyze heatmap output and document what to fix
* **Run corrected pipeline**: Re-generate with improved prompts and compare
* **Produce the comparison report**: Quantify the gap between your final dataset and the benchmark

### Common Pitfalls to Avoid

* **Don't hardcode prompts**: Keep templates in a dictionary or config file so you can swap them easily
* **Don't run the judge at the same temperature as the generator**: You want deterministic judgments, not creative ones
* **Don't skip rate limiting**: LLM APIs will throttle you; add a small delay between requests
* **Don't correct prompts blindly**: Every change must be justified by the failure data
* **Don't use a sample of fewer than 30 items**: Failure rates are noisy at small scale

### Storage Strategy

* Use **JSONL** for generated data (streaming-friendly, easy to append)
* Use **JSON** for summaries and before/after comparison reports
* Use **PNG** for visualizations (widely compatible)
* Save baseline and corrected outputs with separate filenames. You'll need both for comparison

***

## 📚 Key Concepts to Understand

Before building, make sure you understand these concepts:

| Concept                                  | Why It Matters Here                                                                                           |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Pydantic schema validation**           | Guarantees every item in your dataset has the right structure before it enters your pipeline                  |
| **Instructor / structured LLM output**   | Forces LLMs to return schema-compliant JSON instead of free-form text                                         |
| **LLM-as-Judge pattern**                 | Using one LLM to evaluate the output of another. This is the core quality control mechanism                   |
| **Prompt templating**                    | Producing varied generation inputs to avoid repetitive, homogeneous output                                    |
| **Binary failure scoring**               | Translates subjective quality judgments into measurable, comparable numbers                                   |
| **Failure mode correlation**             | Understanding which quality problems tend to appear together reveals shared root causes                       |
| **Reference dataset calibration**        | Validating your evaluation tool against known-good data ensures your measurements are meaningful              |
| **Multi-dimensional quality evaluation** | Measuring quality across multiple independent dimensions reveals which aspects of generation need improvement |
| **Prompt correction feedback loop**      | Closing the loop from evaluation results back to generation inputs is what makes data pipelines improvable    |
| **Rate limiting and retry logic**        | LLM APIs have limits. Production pipelines need to handle them gracefully                                     |

***

## ✅ Final Checklist

### Data Generation

* \[ ] Pipeline generates >= 50 Q\&A pairs per run without crashing
* \[ ] All generated items include all 7 required fields (`question`, `answer`, `equipment_problem`, `tools_required`, `steps`, `safety_info`, `tips`)
* \[ ] >= 95% of generated items pass Pydantic schema validation
* \[ ] All 5 repair categories are represented in the generated dataset
* \[ ] Diverse prompt templates used (at least one per repair category)
* \[ ] Pipeline handles malformed LLM responses without crashing

### LLM-as-Judge

* \[ ] LLM-as-Judge evaluates all 6 failure modes independently for every item
* \[ ] LLM-as-Judge evaluates all 8 quality dimensions independently for every item
* \[ ] Judge uses structured output (Instructor/Pydantic) for consistent scoring
* \[ ] Judge calibration validated: running judge on >= 50 benchmark items achieves >= 95% pass rate

### Failure Analysis

* \[ ] Baseline overall failure rate is >= 15% (demonstrating the pipeline can detect real problems)
* \[ ] Failure heatmap is generated and correctly shows co-occurrence patterns
* \[ ] Failure rates computed per repair category
* \[ ] Most problematic items (3+ simultaneous failures) identified

### Quality Evaluation

* \[ ] Quality dimension scores computed for all 8 dimensions
* \[ ] Quality dimension scores are visualized (bar chart, radar chart, or heatmap)
* \[ ] Benchmark comparison report included with quantitative quality gap analysis

### Prompt Correction

* \[ ] Corrected prompts are documented with explanations of what was changed and why
* \[ ] Each correction is traceable to specific failure data
* \[ ] Post-correction failure rate is \<= 80% of the baseline (>80% improvement achieved)
* \[ ] Post-correction dataset meets quality thresholds (>= 80% overall quality pass rate)
* \[ ] Before/after comparison is clearly reported with per-mode failure rates and per-dimension quality scores

### Visualizations

* \[ ] Failure mode co-occurrence heatmap generated and saved
* \[ ] Failure rates by repair category chart generated and saved
* \[ ] Per-mode failure trend (before vs. after) chart generated and saved
* \[ ] Quality dimension scores chart generated and saved
* \[ ] Benchmark vs. generated comparison chart generated and saved
* \[ ] All charts use Matplotlib, Seaborn, or Plotly
* \[ ] All charts saved as PNG in `visualizations/` directory

### Iteration Logs

* \[ ] Iteration log maintained with structured entries
* \[ ] Each prompt correction cycle documented with hypothesis, result, and decision
* \[ ] At least 4 iteration entries recorded

### Code Quality and Testing

* \[ ] Code is modular with clear phase separation (generation, validation, judging, analysis, correction)
* \[ ] Configuration management uses Pydantic models
* \[ ] Error handling for LLM API calls (rate limits, malformed responses)
* \[ ] Unit tests for schema validation logic
* \[ ] Integration test for full pipeline end-to-end

**Remember**: This isn't about following a step-by-step tutorial. It's about understanding the problem, making data-driven decisions, and proving your solution works through rigorous evaluation. The goal isn't a clean pipeline. It's a _provably better_ pipeline. Good luck!

​
