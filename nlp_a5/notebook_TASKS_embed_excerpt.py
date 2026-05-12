"""
Excerpt of how the Kaggle notebook embeds the evaluation set (TASKS).

In `nlp_assignment5_trackA_kaggle.ipynb` the Track-A tasks live inside a huge string
named `_TASKS_EMBED`, and then the notebook executes:

    TASKS = json.loads(_TASKS_EMBED)
    assert all(t.get('track') == 'A' for t in TASKS)

The real notebook string contains 32 tasks (t001..t032). This file is an excerpt
you can cite with stable line numbers in the report while still matching the
notebook’s mechanism.
"""

from __future__ import annotations

import json

_TASKS_EMBED_EXCERPT = """
[
  {
    "id": "t001",
    "track": "A",
    "adversarial": false,
    "prompt": "Знайди 5 найновіших arXiv-робіт з темою 'probabilistic circuits' (останні 36 місяців) та для кожної дай 1 речення summary + arxiv id.",
    "rubric": {
      "quality_scale": {"0": "нема або вигадані ids", "1": "частково з інструментів", "2": "усі IDs підтверджені tool-текстом"},
      "must_use_tool_classes": ["arxiv_search"],
      "forbidden": ["DOI/arXiv без джерела з інструментів"],
      "notes": "очікуй ≥1 arxiv_search"
    }
  },
  {
    "id": "t011",
    "track": "A",
    "adversarial": false,
    "prompt": "Use fetch_mcp GET https://api.openalex.org/works?search=inductive%20biases — summarize first hit title+year без копіпаст HTML noise.",
    "rubric": {
      "quality_scale": {"0": "no fetch", "1": "partial clean", "2": "clean minimal summary"},
      "must_use_tool_classes": ["fetch"],
      "forbidden": ["using openalex_* tool"],
      "notes": "принудимо fetch-only skill check"
    }
  }
]
"""

# This is exactly how the notebook loads TASKS (but with the full 32-task JSON).
TASKS = json.loads(_TASKS_EMBED_EXCERPT)
assert all(t.get("track") == "A" for t in TASKS)
print("Loaded Track A tasks (excerpt):", len(TASKS))

