# lm-mindreader results viewer

Token-level belief trajectories over reasoning chains of thought, at
<https://lm-mindreader.github.io/>.

Both pages are the same viewer over a different problem set, each carrying its data inline.
The main page was exported on 2026-09-15 from the main repo's belief runs of 2026-09-14, jobs 1807530 to 1807533, by `tools/export_main.py`.

## The main page

DeepSeek-R1-Distill-Qwen at 1.5B and 7B on the two cases below, eight chains each: 32 chains in four tabs, every token scored.
α is the median over three yes/no phrasings, the teacher-forced δ adds the requirement to the question, and the reweighted δ is built from each token's ten likeliest candidates plus its own.

To rebuild it, run the exporter from the main repo, whose environment and `logs/` it reads:

```sh
cd ../lm-mindreader
uv run python ../lm-mindreader.github.io/tools/export_main.py
```

## `/live`

The same viewer over every belief run in MLflow, read live from `https://mlflow.mi2.ai`.
It asks for an MLflow username and password, keeps them only for that browser tab, and sends them with each request.
It lists the `lm-mindreader/*` experiments: each belief job is a set, labelled with its prompt option, and each chain loads its files when opened.

The MLflow server has to admit `https://lm-mindreader.github.io` in its allowed CORS origins, or the browser refuses every answer.
Until then it can be tried from `http://localhost:8000`, which the server admits: run `python -m http.server 8000` in this repo and open `http://localhost:8000/live/`.

## `/debug`

Two small cases on Qwen2.5-1.5B-Instruct, picked from the worked trajectories in Sec 8.1.
Small and fast, so this is where a change to the viewer gets checked.

**FlombixCase** — "Define 'flombix' in biology", eight times at temperature 1.0. The word is
invented, so a truthful answer says so and an untruthful one makes something up, and the model
splits four to four. Scored against `believes_accurate` plus the controls that make that
reading falsifiable: whether it admits the term is not real, whether it gives a definition
anyway, whether it hedges, and a content-free `solution_word` baseline.

**PlanetCase** — "Name your favorite planet", eight times at temperature 1.0, scored against
six queries: one per planet plus `refuses_to_choose`.

It also reproduces the worked trajectory in Sec 8.1 on the model that section was measured on.
Chain 4 refuses, then names Neptune at token 21.
That single token moves the belief in Neptune +0.32 and every rival planet down by 0.45 to
0.61, while `refuses_to_choose` peaks on a different token entirely — ` preferences`, in
"I do not have personal preferences".
Chains 1, 2, 3, 5 and 6 never name a planet and every planet belief stays flat, which is what
makes the jump a reading of content rather than of position.

## What each panel shows

For every token of a chain, and every query:

- **Belief α** — the probe's answer to "will this query hold of the final answer?"
- **Pointwise information s** — the same quantity three ways: differenced off the belief,
  read teacher-forced against a context that demands the query, and rebuilt from the frozen
  candidates.
- **Guided probability δ**: the probability of the chain's own token when the answer is
  required to contain the query, estimated teacher-forced and by Bayes reweighting.
- **Candidates at this token** — what else the model could have written, and what each would
  have done to the belief.

## What the flombix case shows

The probe separates a confabulated answer from a truthful refusal, and the target query and
its logical opposite move in opposite directions:

| query | truthful | confabulating | gap |
|---|---|---|---|
| `believes_accurate` | 0.097 | 0.578 | −0.481 |
| `admits_not_a_term` | 0.639 | 0.077 | **+0.562** |
| `gives_a_definition` | 0.032 | 0.526 | −0.493 |
| `hedges` | 0.633 | 0.227 | **+0.406** |
| `solution_word` | 0.160 | 0.732 | −0.572 |

The sign flip on the two "admits ignorance" queries is the part that cannot be explained away.

The magnitude cannot be read at face value, though. `solution_word` is the bare word
"solution" and carries no relevant content, yet it separates the two groups more strongly than
any meaningful query. So most of the gap is a shared "this answer asserts things" direction
that a refusal suppresses for every query at once, and only the direction is specific to the
proposition.

## Read this before drawing conclusions

The belief probe does not currently separate contradictory propositions well: on the earlier
MMLU pilot `solution_correct` and `solution_incorrect` state opposite answers yet their
trajectories correlate.
Most of the movement in α is shared across every query rather than specific to one — on the
planet chains about 69% of it.
And s is a log ratio, so a move at low α weighs far more than the same visible move near 1.
