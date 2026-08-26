# lm-mindreader results viewer

Token-level belief trajectories over reasoning chains of thought, served as a static
password-protected page at <https://lm-mindreader.github.io/>.

## What is in this run

DeepSeek-R1-Distill-Qwen-1.5B on MMLU-Redux-2.0 `formal_logic`, all 100 questions
including the 13 the re-annotation flagged.
The chains were generated with the full token-level record frozen at generation: the
realized token, its free log-probability, its vocabulary rank and the top-50 candidates.

Belief scoring is a pilot over the three shortest chains against the full 20-query panel,
60 (chain, query) cells.
For every token it carries the belief `alpha`, both estimators of the pointwise
information `s`, the cumulative products `beta` and `phi`, and a per-candidate breakdown.

## Opening it

The page asks for a password.
Ask Michał for it; it is shared, and anyone who has it can keep whatever they decrypt.

## Read this before drawing conclusions

The belief probe does not currently separate contradictory propositions.
`solution_correct` and `solution_incorrect` assert opposite answers, yet their
trajectories correlate at r = 0.98.
A single trajectory shared across all 20 queries explains about 79% of the variance in
`alpha`, leaving roughly a fifth for query identity.

So most of what a trajectory shows is how far into the chain it is, not what the query
says.
Individual curves look plausible and should not be trusted on their own until that is
resolved.

## Why the payload is encrypted

GitHub Pages has no server side, so a password prompt alone would protect nothing: the
files sit at guessable URLs in a public repository.
Each chain is a separate AES-256-GCM shard under a key derived from the password with
PBKDF2-HMAC-SHA256, and the page decrypts it in the browser with WebCrypto.
`data/keyinfo.json` holds only the KDF parameters, never the password.

No raw tables are published here for the same reason.
The parquet tables live on the cluster and are shared out of band.

## Rebuilding

From the `lm-mindreader` repository:

```
uv run python src/utils/scripts/belief_export.py --out results-fl100 \
    --run fl100-pilot-1.5b <run-dir> FL100Pilot <generation.jsonl> 60
uv run python src/utils/scripts/belief_site_export.py \
    --results results-fl100 --out results-fl100/site --run fl100-pilot-1.5b
```

Then copy `results-fl100/site/.` over this repository and push.
