# Explainable & Robust Phishing Detection (Transformers + Reasoning LLM)

Generation-fingerprint-aware phishing detection. The project turns the original Colab notebook
(`notebooks/Phishing_Detection_Colab_Gemini_original.ipynb`) into a modular, testable pipeline with a
single orchestrator entry point.

**Idea:** three independent tools each write a short *evidence statement* about an email, and **one**
reasoning LLM (Gemini) reads them and returns a decision plus an explanation. There is no separately
trained fusion layer, so the explanation reflects what was actually used for the decision.

| Evidence | Produced by |
|---|---|
| **TEXT** | adversarially trained DistilBERT probability + top LIME phrases |
| **URL** | structural URL facts, look-alike distance to well-known brands |
| **FINGERPRINT** | GPT-2 perplexity / burstiness (is the wording LLM-like?) |

## Project layout

```
phishing-detection-llm/
├── main.py                    # CLI orchestrator entry point
├── configs/                   # default.json, quick_smoke.json
├── phishing_detection/
│   ├── pipeline.py            # stage registry, dependency resolution, Context, timing, error handling
│   ├── stages.py              # the 11 pipeline stages
│   ├── config.py              # Config dataclass (JSON file + --set overrides)
│   ├── data.py                # load / clean / split
│   ├── transformer.py         # DistilBERT, FGM adversarial training, save/load
│   ├── perturb.py             # character-level obfuscation (verify -> ver1fy)
│   ├── url_analysis.py        # URL features + look-alike detection
│   ├── fingerprint.py         # perplexity / burstiness + threshold calibration
│   ├── llm.py                 # Gemini client: cache, retries, quota handling, paraphrasing
│   ├── framework.py           # evidence statements -> reasoning LLM -> decision + explanation
│   └── utils.py               # seeding, logging, metrics, figure/table saving
├── notebooks/
│   ├── colab_launcher.ipynb   # thin Colab runner for this repo
│   └── Phishing_Detection_Colab_Gemini_original.ipynb
├── tests/                     # pytest unit tests (no GPU / API key needed)
├── data/                      # put your dataset here (git-ignored)
└── .env.example
```

## Quick start

```bash
git clone https://github.com/Seemab5767/phishing-detection-llm.git
cd phishing-detection-llm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then put your Gemini key in .env  (never commit it)
# dataset: CSV/XLSX with columns "Email Text" and "Email Type" (phishing / safe) -> data/emails.csv

python main.py run --data data/emails.csv
```

A **GPU is strongly recommended** for the DistilBERT stages (use Colab: `notebooks/colab_launcher.ipynb`).

## The orchestrator

`python main.py list-stages` shows the pipeline. Stages declare their dependencies; the orchestrator adds
what is missing, runs everything in order, logs per-stage timings and saves outputs after each stage.

```
data → baselines
     → transformers → robustness
                    → urls
     → fingerprint ─┐
       transformers ┴→ framework → faithfulness
                                 → attack
                                 → demo
report (collects everything produced so far, writes summary + zip)
```

```bash
python main.py run --data data/emails.csv --stages baselines            # just the TF-IDF baselines
python main.py run --data data/emails.csv --stages faithfulness         # + data, transformers, fingerprint, framework
python main.py run --data data/emails.csv --skip framework faithfulness attack demo   # no Gemini key needed
python main.py run --data data/emails.csv --reuse-models                # skip retraining (loads outputs/models)
python main.py run --data data/emails.csv --config configs/quick_smoke.json   # fast sanity run
python main.py run --data data/emails.csv --set epochs=3 n_eval_per_class=40
python main.py run --data data/emails.csv --continue-on-error           # keep going if a stage fails

python main.py analyze --text "Dear customer, verify your account at http://paypa1-login.xyz"
```

* If LLM stages are in the plan but no key is configured, the run **fails immediately** (before any training)
  with instructions, instead of crashing an hour in.
* The Gemini responses are cached in `outputs/cache/`, so re-running costs no extra quota.
* If Gemini is unavailable, the fingerprint stage falls back to a default threshold (`default_ppl_threshold`).

## Outputs (`outputs/`)

`tables/` (CSV results), `figures/` (PNG), `lime/` (HTML explanations), `models/` (weights, git-ignored),
`logs/` (run log, config, stage timings), `RESULTS_SUMMARY.md`, plus `outputs.zip` (weights excluded unless
`zip_models=true`).

## Configuration

All knobs live in `phishing_detection/config.py` (see `configs/default.json`). Override with `--config file.json`
or `--set KEY=VALUE`. Notable ones: `epochs`, `max_rows`, `reasoning_llm` (try `gemini-2.5-flash-lite` for higher
free-tier limits), `llm_autoselect`, `n_eval_per_class`, `n_cal`, `lime_samples`.

## Tests

```bash
pytest -q
```

## Responsible use

The LLM paraphrasing step rewrites **only emails that already exist in your dataset** to test whether the
detector survives fluent rewriting. Nothing new is generated from scratch.

## Security note

API keys are read from `GEMINI_API_KEY` (environment, `.env`, or Colab Secrets) and are never stored in code.
`.env`, datasets, outputs and caches are git-ignored.
