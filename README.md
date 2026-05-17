# Startup Valuation Is Right

An autonomous multi-agent framework for a **Startup Valuation Guessing Game**.

## Concept

Instead of spotting above-market salaries, the system analyzes startup profiles and estimates valuation:

1. Show startup profile data (description, funding, traction, region)
2. Compare reported valuation vs AI-estimated valuation
3. Surface high-gap opportunities
4. Optionally send push notifications

## Agent Architecture

- **Scanner Agent**: pulls startup/news feeds and parses entries into structured profile objects
- **Specialist Agent**: calls a fine-tuned valuation model on Modal
- **Frontier Agent**: RAG + GPT valuation estimator using ChromaDB similar-profile lookup
- **Ensemble Agent**: combines specialist + frontier estimates
- **Messaging Agent**: crafts and sends concise alerts
- **Planning / Autonomous Planning Agent**: orchestrates tool-calling loop end-to-end

## Folder Layout

- `startup_valuation_is_right.py` - Gradio app entrypoint
- `startup_agent_framework.py` - orchestration + memory + Chroma setup
- `startup_valuation_service.py` - Modal deployment file for specialist model
- `agents/` - all agent implementations
- `startup_valuation_is_right.ipynb` - walkthrough notebook (same format style as original)
- `memory.json` - persisted memory of discovered opportunities

## Quick Start

From this directory:

```bash
python startup_valuation_is_right.py
```

Notebook flow:

```bash
jupyter notebook startup_valuation_is_right.ipynb
```

## Notes

- Field names in code may still use legacy names (for compatibility), but semantics are startup valuation focused.
- Configure environment variables (`OPENAI_API_KEY`, `PUSHOVER_USER`, `PUSHOVER_TOKEN`, etc.) before running live workflows.
- Inspired by Ed Donner's LLM Engineering course on Udemy
