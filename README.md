# lm-mindreader results viewer

Token-level belief trajectories over reasoning chains of thought, at
<https://lm-mindreader.github.io/>.

Both pages are the same viewer over a different problem set, built by
`src/utils/scripts/belief_viewer_export.py` in the main repo.

## The main page

DeepSeek-R1-Distill-Qwen-1.5B on MMLU-Redux-2.0 `formal_logic`, all 100 questions including
the 13 the re-annotation flagged.
The chains were generated with the full token-level record frozen at generation: the realized
token, its free log-probability, its vocabulary rank and the top-50 candidates.
Belief scoring is a pilot over the three shortest chains against the full 20-query panel,
60 (chain, query) cells.

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
- **Guided log δ** — the measure s is read off, in both of its variants.
- **Belief α implied by δ** — γ·exp(Σs) against the probed α, on a log axis. The distance
  between them is the telescoping residual.
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

The belief probe does not currently separate contradictory propositions well: on the main
page `solution_correct` and `solution_incorrect` state opposite answers yet their
trajectories correlate.
Most of the movement in α is shared across every query rather than specific to one — on the
planet chains about 69% of it.
And s is a log ratio, so a move at low α weighs far more than the same visible move near 1.
