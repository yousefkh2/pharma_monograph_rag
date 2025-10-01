# Evaluation Toolkit

This folder now includes an end-to-end evaluation workflow for the Pharmacy Copilot stack.

## Components

- `run_evaluation.py`: general-purpose RAG evaluation (retrieval + QA metrics).
- `run_clinical_evaluation.py`: adds clinical compliance and safety scoring on top of the base evaluation.
- `llm_client.py`: async client that can call Ollama, OpenAI, or OpenRouter models.
- `report_viewer.html`: static dashboard for exploring JSON exports from either runner.
- `datasets/pharmacist_eval_v1.json`: curated pharmacist-style questions to stress the LLM output while assuming high-quality retrieval.

## Running with an LLM

The runners default to a deterministic snippet-concatenation stub. Enable live LLM answers with `--use-llm` and point at your provider:

```bash
export EVAL_LLM_PROVIDER=openrouter
export EVAL_LLM_MODEL="openrouter/anthropic/claude-3.5-sonnet"
export EVAL_LLM_API_KEY="sk-or-..."
export EVAL_LLM_REFERER="https://yourdomain.example"   # optional but recommended by OpenRouter

python eval/run_clinical_evaluation.py eval/datasets/pharmacist_eval_v1.json \
  --json-output eval/reports/pharmacist_eval.json \
  --report eval/reports/pharmacist_eval.md \
  --use-llm --timeout 60 --top-k 15
```

Open the HTML viewer to inspect results:

```bash
open eval/report_viewer.html
```

Load the generated JSON file to browse per-question context, scores, and failure tags.

## Customising Prompts

Override the system prompt and hyper-parameters via CLI flags or environment variables:

- `--llm-system-prompt` or `EVAL_LLM_SYSTEM_PROMPT`
- `--llm-temperature` / `EVAL_LLM_TEMPERATURE`
- `--llm-max-tokens` / `EVAL_LLM_MAX_TOKENS`

## Next Steps

- Populate `relevant_chunks` in the dataset after verifying retrieval accuracy so the recall metrics reflect the ground truth.
- Use the viewer to confirm retrieved evidence matches expectations before comparing models.
- Run multiple models by changing `EVAL_LLM_MODEL` between executions (OpenRouter supports a wide catalog under the same API key).
