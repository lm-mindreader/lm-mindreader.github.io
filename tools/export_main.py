import argparse
import base64
import importlib.util
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_SITE = Path(__file__).resolve().parents[1]
_RUNS = Path('logs/calculate_cot_belief/2026-09-14')
_SETS = [
    ('04-09-32_1807531', 'FlombixCase R1-1.5B'),
    ('04-09-32_1807533', 'FlombixCase R1-7B'),
    ('04-09-32_1807530', 'PlanetCase R1-1.5B'),
    ('04-09-32_1807532', 'PlanetCase R1-7B'),
]
_A_FLOOR = -12.0
_S_COLUMNS = {'s': 's_delta', 'sAlpha': 's_alpha', 'sReweight': 's_delta_reweighted'}
_D_COLUMNS = {'d': 'log_delta', 'dEq12': 'log_delta_reweighted'}
# A paragraph starts after a blank line, and a sentence after closing punctuation and whitespace.
_PARAGRAPH = re.compile(r'\n[ \t]*\n\s*')
_SENTENCE = re.compile(r'[.?!]["\')\]]*\s+')
_S_SERIES = [
    {
        'key': 's',
        'label': 's<sub>i</sub> = log(&delta;<sub>i</sub> / A<sub>i</sub>), teacher-forced',
    },
    {
        'key': 'sAlpha',
        'label': 's<sub>i</sub> = log(&alpha;<sub>i</sub> / &alpha;<sub>i&minus;1</sub>)',
    },
    {
        'key': 'sReweight',
        'label': 's<sub>i</sub> = log(&delta;<sub>i</sub> / A<sub>i</sub>), &delta;(v) &prop; A(v)&thinsp;&alpha;(v) over the frozen candidates',  # noqa: E501
    },
]
_D_SERIES = [
    {
        'key': 'd',
        'label': 'log &delta;<sub>i</sub> = log P(Y<sub>i</sub> | X&prime;, Y<sub>&lt;i</sub>)',
    },
    {
        'key': 'dEq12',
        'label': 'log &delta;<sub>i</sub>, &delta;(v) &prop; A(v)&thinsp;&alpha;(v) renormalised over the frozen candidates',  # noqa: E501
    },
]
_A_LABEL = 'log<sub>10</sub> &alpha;<sub>i</sub> = log<sub>10</sub>&thinsp;&gamma; + &Sigma;<sub>k&le;i</sub> s<sub>k</sub> / ln&thinsp;10'  # noqa: E501


def _main_repo(
    path: Path,
):
    """
    The main repo's upload tool, whose helpers read the runs as the stages wrote them.
    """
    sys.path.insert(0, str(path / 'src'))
    tool = path / 'src' / 'utils' / 'scripts' / 'upload_belief_to_mlflow.py'
    spec = importlib.util.spec_from_file_location('upload_belief_to_mlflow', tool)
    upload = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upload)
    return upload


def _quantised(
    values,
    lo: float,
    hi: float,
) -> str:
    scaled = np.round((np.asarray(values, dtype=np.float64) - lo) / (hi - lo) * 65535)
    codes = np.clip(scaled, 0, 65535).astype('<u2')
    text = base64.b64encode(codes.tobytes()).decode()
    return text


def _decoded(
    text: str,
    lo: float,
    hi: float,
) -> np.ndarray:
    codes = np.frombuffer(base64.b64decode(text), dtype='<u2').astype(np.float64)
    values = lo + codes * (hi - lo) / 65535
    return values


def _packed(
    values,
    dtype: str,
) -> str:
    text = base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode()
    return text


def _starts(
    text: str,
    edges: np.ndarray,
    pattern: re.Pattern,
) -> list[int]:
    found = [0]
    for match in pattern.finditer(text):
        token = int(np.searchsorted(edges, match.end(), side='right')) - 1
        if found[-1] < token < len(edges) - 1:
            found.append(token)
    return found


def _buckets(
    toks: list[str],
) -> dict[str, list[int]]:
    edges = np.concatenate([[0], np.cumsum([len(tok) for tok in toks])])
    text = ''.join(toks)
    buckets = {'sent': _starts(text, edges, _SENTENCE), 'para': _starts(text, edges, _PARAGRAPH)}
    return buckets


def _implied(
    gamma: float,
    s: np.ndarray,
) -> np.ndarray:
    running = math.log10(max(gamma, 1e-12)) + np.cumsum(np.nan_to_num(s)) / math.log(10)
    implied = np.clip(running, _A_FLOOR, 0)
    return implied


def _percentile(
    values: np.ndarray,
    q: float,
) -> float:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 1e-3

    value = float(np.percentile(finite, q))
    return value


def _candidates(
    rows: pd.DataFrame,
    order: list[str],
    ensemble: str,
    tokenizer,
) -> dict:
    """
    Per position one slice of candidates, in the order the yes/no metric read them.
    """
    unique = rows.drop_duplicates(['token_index', 'token_id'])
    firsts = unique.sort_values('token_index', kind='stable')
    alpha = rows.groupby(['query', 'token_index', 'token_id']).belief.agg(ensemble)
    pieces = {int(token): tokenizer.decode([int(token)]) for token in firsts.token_id.unique()}
    vocab = sorted(set(pieces.values()))
    slot = {piece: i for i, piece in enumerate(vocab)}
    positions = firsts.groupby('token_index', sort=True)
    realized = positions.is_realized.apply(lambda flags: int(np.flatnonzero(flags.to_numpy())[0]))
    keys = list(zip(firsts.token_index, firsts.token_id, strict=True))
    log_a = firsts.log_a.to_numpy(np.float64)
    lo, hi = float(log_a.min()), float(log_a.max())
    block = {
        'lo': lo,
        'hi': hi,
        'vocab': vocab,
        't': positions.size().index.tolist(),
        'k': _packed(positions.size().to_numpy(), 'u1'),
        'r': _packed(realized.to_numpy(), 'u1'),
        'v': _packed([slot[pieces[int(token)]] for token in firsts.token_id], '<u2'),
        'a': _quantised(log_a, lo, hi),
        'p': {q: _quantised(alpha.loc[q].reindex(keys).to_numpy(), 0.0, 1.0) for q in order},
    }
    return block


def _series(
    rows: pd.DataFrame,
    order: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    ordered = rows.sort_values(['query', 'token_index'])
    columns = {**_S_COLUMNS, **_D_COLUMNS, 'alpha': 'alpha', 'gamma': 'gamma'}
    series = {
        key: {q: ordered.loc[ordered['query'] == q, column].to_numpy(np.float64) for q in order}
        for key, column in columns.items()
    }
    return series


def _case(
    chain,
    record: dict,
    series: dict,
    texts: list[dict],
    cand: dict,
    toks: list[str],
    order: list[str],
) -> dict:
    stacked = {key: [series[key][q] for q in order] for key in _S_COLUMNS}
    magnitudes = {key: np.abs(np.concatenate(values)) for key, values in stacked.items()}
    pooled = np.concatenate(list(magnitudes.values()))
    s_cap = _percentile(pooled, 99.9)
    subject, idx = chain.uid.split('#')
    case = {
        'id': chain.uid,
        'subject': subject,
        'idx': idx,
        'short': f'CoT {idx}',
        'question': record['question'],
        'gold': record.get('expected_answer') or '',
        'guess': '',
        'correct': None,
        'finish': record['finish_reason'],
        'truncated': record['finish_reason'] == 'length',
        'n': len(toks),
        'nFull': len(toks),
        'toks': toks,
        'qtext': {text['name']: text['text'] for text in texts},
        'buckets': _buckets(toks),
        'cand': cand,
        'sCap': s_cap,
        'sCaps': {key: _percentile(values, 99.9) for key, values in magnitudes.items()},
        'sMax': float(np.nanmax(pooled)),
        'sOver': int(np.sum(pooled > s_cap)),
        'meanA': {q: float(np.mean(series['alpha'][q])) for q in order},
        'sdA': {q: float(np.std(series['alpha'][q])) for q in order},
    }
    return case


def _quantise_case(
    case: dict,
    series: dict,
    order: list[str],
    ranges: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """
    Fills the case's encoded series and returns each series' largest error once decoded.
    """
    implied = {q: _implied(series['gamma'][q][0], series['s'][q]) for q in order}
    wanted = {**{key: series[key] for key in ranges}, 'aImplied': implied}
    errors = {}
    for key, values in wanted.items():
        lo, hi = ranges.get(key, (_A_FLOOR, 0.0))
        case[key] = {q: _quantised(values[q], lo, hi) for q in order}
        gaps = [np.max(np.abs(_decoded(case[key][q], lo, hi) - values[q])) for q in order]
        errors[key] = float(max(gaps))
    return errors


def _realized_share(
    cand: dict,
    toks: list[str],
) -> float:
    k = np.frombuffer(base64.b64decode(cand['k']), dtype='u1').astype(np.int64)
    r = np.frombuffer(base64.b64decode(cand['r']), dtype='u1').astype(np.int64)
    v = np.frombuffer(base64.b64decode(cand['v']), dtype='<u2')
    offsets = np.concatenate([[0], np.cumsum(k)])
    spelled = [cand['vocab'][v[offsets[i] + r[i]]] for i in range(len(k))]
    share = sum(a == b for a, b in zip(spelled, toks, strict=True)) / len(toks)
    return share


def _run(
    run_dir: Path,
    tag: str,
    upload,
) -> dict:
    log.info(f'Reading {run_dir}')
    config = upload._config(run_dir)
    _, set_name = upload.locate_estimation(run_dir, config['cots']['set_name'])
    belief_paths = upload.paths(run_dir, set_name)
    estimation = json.loads(belief_paths.estimation_summary.read_text())
    generation = upload.paths(estimation['cots_dir'], set_name)
    belief = pd.read_parquet(belief_paths.belief)
    candidates = pd.read_parquet(belief_paths.yes_no_candidates)
    prior = pd.read_parquet(belief_paths.yes_no_prior)
    chains = upload.read_chains(generation)
    records = upload._records(generation.jsonl)
    groups = upload.queries.load_groups(config['query']['groups'])
    templates = upload.queries.select(groups, config['query']['names'])
    sources = upload.queries.load_concept_sources(config['exp']['concepts_paths'] or {})
    model = config['model']['model_name']
    tokenizer = upload.AutoTokenizer.from_pretrained(model)
    ensemble = config['exp']['ensemble']
    order = [template['name'] for template in templates]

    log.info(f'Building the {len(chains)} chains of {set_name}')
    cases, all_series = [], []
    for chain in chains:
        toks = upload.token_pieces_from_ids(chain.token_ids, tokenizer)
        series = _series(belief[belief.uid == chain.uid], order)
        concepts = upload.queries.concept_sources_for(sources, chain.uid)
        record_for_queries = {
            'question': chain.question,
            'answer': chain.answer,
            'concepts': concepts,
        }
        texts = upload._query_texts(templates, record_for_queries, prior[prior.uid == chain.uid])
        cand = _candidates(candidates[candidates.uid == chain.uid], order, ensemble, tokenizer)
        case = _case(chain, records[chain.uid], series, texts, cand, toks, order)
        share = _realized_share(cand, toks)
        log.info(f'{chain.uid}: {len(toks)} tokens, realized candidate matches at {share:.3f}')
        cases.append(case)
        all_series.append(series)

    log.info(f'Encoding {set_name}')
    s_values = np.concatenate([s[key][q] for s in all_series for key in _S_COLUMNS for q in order])
    d_values = np.concatenate([s[key][q] for s in all_series for key in _D_COLUMNS for q in order])
    s_range = (float(np.nanmin(s_values)), float(np.nanmax(s_values)))
    d_range = (float(np.nanmin(d_values)), float(np.nanmax(d_values)))
    ranges = {
        'alpha': (0.0, 1.0),
        **dict.fromkeys(_S_COLUMNS, s_range),
        **dict.fromkeys(_D_COLUMNS, d_range),
    }
    pairs = zip(cases, all_series, strict=True)
    errors = [_quantise_case(case, series, order, ranges) for case, series in pairs]
    worst = {key: max(error[key] for error in errors) for key in errors[0]}
    summary = ', '.join(f'{key} {value:.1e}' for key, value in worst.items())
    log.info(f'{tag}: largest decode error {summary}')

    n_tokens = sum(case['n'] for case in cases)
    run = {
        'name': set_name,
        'tag': tag,
        'sSeries': _S_SERIES,
        'dSeries': _D_SERIES,
        'dLo': d_range[0],
        'dHi': d_range[1],
        'aLabel': _A_LABEL,
        'aFloor': _A_FLOOR,
        'cap': '',
        'model': model,
        'sLo': s_range[0],
        'sHi': s_range[1],
        'nGen': len(cases),
        'nQueries': len(order),
        'nShown': len(cases),
        'nScored': len(cases),
        'nPos': n_tokens,
        'rows': n_tokens * len(order),
        'cellsPresent': len(cases) * len(order),
        'cellsExpected': len(cases) * len(order),
        'complete': True,
        'cases': cases,
    }
    return run


def _with_data(
    html: str,
    raw: dict,
) -> str:
    """
    The page with its one RAW line replaced; a literal </ would end the script, so it is escaped.
    """
    text = json.dumps(raw, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    lines = html.split('\n')
    found = [i for i, line in enumerate(lines) if line.startswith('const RAW=')]
    if len(found) != 1:
        raise ValueError(f'expected one RAW line in the page, found {len(found)}')

    lines[found[0]] = f'const RAW={text};'
    page = '\n'.join(lines)
    return page


def main():
    parser = argparse.ArgumentParser(
        description='Rebuilds the main page from the 2026-09-14 debug belief runs.',
    )
    parser.add_argument(
        '--main-repo', type=Path, default=Path.cwd(), help='The main repo checkout.'
    )
    parser.add_argument('--output', type=Path, default=_SITE / 'index.html', help='Where it goes.')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    main_repo = args.main_repo.resolve()

    upload = _main_repo(main_repo)
    runs = [_run(main_repo / _RUNS / name, tag, upload) for name, tag in _SETS]
    raw = {'commit': '', 'runs': runs}

    page = _with_data((_SITE / 'index.html').read_text(encoding='utf-8'), raw)
    args.output.write_text(page, encoding='utf-8')
    log.info(f'Wrote {args.output}, {len(page.encode()) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
