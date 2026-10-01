import { parquetMetadata, parquetRead } from 'hyparquet';
import { compressors } from 'hyparquet-compressors';

export const SERIES_LABELS = {
  sAlpha: 's<sub>i</sub> = log(&alpha;<sub>i</sub> / &alpha;<sub>i&minus;1</sub>)',
  s: 's<sub>i</sub> = log(&delta;<sub>i</sub> / A<sub>i</sub>), teacher-forced',
  sReweight: 's<sub>i</sub> = log(&delta;<sub>i</sub> / A<sub>i</sub>), &delta;(v) &prop; A(v)&thinsp;&alpha;(v) over the candidates',
  d: '&delta;<sub>i</sub> = P(Y<sub>i</sub> | X&prime;, Y<sub>&lt;i</sub>), teacher-forced',
  dEq12: '&delta;<sub>i</sub>, Bayes-reweighted: &delta;(v) &prop; A(v)&thinsp;&alpha;(v) renormalised over the candidates',
};

const SERIES_COLUMNS = {
  alpha: 'alpha',
  sAlpha: 's_alpha',
  s: 's_delta',
  sReweight: 's_delta_reweighted',
  d: 'log_delta',
  dEq12: 'log_delta_reweighted',
};
const S_KEYS = ['sAlpha', 's', 'sReweight'];
// The deltas are stored as logs and drawn as the probabilities themselves.
const D_KEYS = ['d', 'dEq12'];
const TRAJECTORY_COLUMNS = ['query', 'token_index', ...Object.values(SERIES_COLUMNS)];
const CANDIDATE_COLUMNS = ['query', 'token_index', 'piece', 'is_realized', 'log_a', 'alpha', 'log_delta_reweighted'];
const CORRECT = { true: true, false: false };
// A paragraph starts after a blank line, and a sentence after closing punctuation and whitespace.
const PARAGRAPH = /\n[ \t]*\n\s*/g;
const SENTENCE = /[.?!]["')\]]*\s+/g;

const fields = (list) => Object.fromEntries((list ?? []).map(({ key, value }) => [key, value]));

async function readColumns(bytes, columns) {
  const metadata = parquetMetadata(bytes);
  const rows = Number(metadata.num_rows);
  const table = Object.fromEntries(columns.map((column) => [column, new Array(rows)]));
  const place = ({ columnName, columnData, rowStart }) => {
    const target = table[columnName];
    for (let i = 0; i < columnData.length; i++) {
      target[rowStart + i] = columnData[i];
    }
  };
  await parquetRead({ file: bytes, metadata, columns, compressors, onChunk: place });
  return table;
}

function setOf(run) {
  const tags = fields(run.data.tags);
  const params = fields(run.data.params);
  const name = tags.set_name ?? run.info.run_name;
  const set = {
    runId: run.info.run_id,
    tag: tags.prompt ? `${name} · ${tags.prompt}` : name,
    model: tags.model ?? '',
    queries: (params.queries ?? '').split(', ').filter(Boolean),
    expected: Number(params.n_cots ?? 0),
    cases: [],
  };
  return set;
}

function chainOf(run) {
  const tags = fields(run.data.tags);
  const params = fields(run.data.params);
  const chain = {
    runId: run.info.run_id,
    parent: tags['mlflow.parentRunId'],
    finished: run.info.status === 'FINISHED',
    artifactUri: run.info.artifact_uri,
    id: params.uid ?? run.info.run_name,
    question: params.question ?? '',
    correct: CORRECT[params.correct] ?? null,
    truncated: params.finish_reason === 'length',
    n: Number(params.n_tokens ?? 0),
  };
  return chain;
}

export async function loadIndex(client, experimentId) {
  const [parents, children] = await Promise.all([
    client.runs(experimentId, "tags.kind = 'belief-set'"),
    client.runs(experimentId, "tags.kind = 'belief-cot'"),
  ]);
  const sets = parents.map(setOf).sort((a, b) => a.tag.localeCompare(b.tag));
  const byId = Object.fromEntries(sets.map((set) => [set.runId, set]));
  const chains = children.map(chainOf).filter((chain) => chain.finished && byId[chain.parent]);
  chains.sort((a, b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
  for (const chain of chains) {
    byId[chain.parent].cases.push(chain);
  }
  for (const set of sets) {
    set.complete = set.cases.length >= set.expected;
  }
  return sets;
}

function seriesByQuery(rows, n, order) {
  const present = [...new Set(rows.query)];
  const queries = [...order.filter((q) => present.includes(q)), ...present.filter((q) => !order.includes(q))];
  const series = {};
  for (const key of Object.keys(SERIES_COLUMNS)) {
    series[key] = Object.fromEntries(queries.map((q) => [q, new Float32Array(n).fill(NaN)]));
  }
  for (let row = 0; row < rows.query.length; row++) {
    const query = rows.query[row];
    const token = rows.token_index[row];
    for (const [key, column] of Object.entries(SERIES_COLUMNS)) {
      const value = rows[column][row] ?? NaN;
      series[key][query][token] = D_KEYS.includes(key) ? Math.min(1, Math.exp(value)) : value;
    }
  }
  return { series, queries };
}

function percentile(values, q) {
  const sorted = Float64Array.from(values.filter(Number.isFinite)).sort();
  if (!sorted.length) {
    return 1e-3;
  }

  const rank = q * (sorted.length - 1);
  const low = Math.floor(rank);
  const high = Math.min(low + 1, sorted.length - 1);
  const value = sorted[low] + (rank - low) * (sorted[high] - sorted[low]);
  return value;
}

function sScales(series, queries) {
  const sCaps = {};
  let pooled = [];
  for (const key of S_KEYS) {
    const magnitudes = queries.flatMap((q) => Array.from(series[key][q], Math.abs));
    sCaps[key] = percentile(magnitudes, 0.999);
    pooled = pooled.concat(magnitudes);
  }
  const sCap = percentile(pooled, 0.999);
  return { sCaps, sCap };
}

function tokenAt(edges, offset) {
  let low = 0;
  let high = edges.length - 2;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (edges[middle] <= offset) {
      low = middle;
      continue;
    }
    high = middle - 1;
  }
  return low;
}

function starts(text, edges, pattern) {
  const found = [0];
  for (const match of text.matchAll(pattern)) {
    const token = tokenAt(edges, match.index + match[0].length);
    if (token > found[found.length - 1] && token < edges.length - 1) {
      found.push(token);
    }
  }
  return found;
}

function segments(toks) {
  const edges = [0];
  for (const tok of toks) {
    edges.push(edges[edges.length - 1] + tok.length);
  }
  const text = toks.join('');
  const buckets = { sent: starts(text, edges, SENTENCE), para: starts(text, edges, PARAGRAPH) };
  return buckets;
}

export async function loadChain(client, chain, order) {
  const [tokens, rows, queryTexts] = await Promise.all([
    client.artifact(chain.artifactUri, 'tokens.parquet').then((bytes) => readColumns(bytes, ['piece'])),
    client.artifact(chain.artifactUri, 'trajectories.parquet').then((bytes) => readColumns(bytes, TRAJECTORY_COLUMNS)),
    client.artifactJson(chain.artifactUri, 'queries.json'),
  ]);
  const toks = tokens.piece;
  const n = toks.length;
  const { series, queries } = seriesByQuery(rows, n, order);
  const qtext = Object.fromEntries(queryTexts.map(({ name, text }) => [name, text]));
  const loaded = {
    ...chain,
    ...series,
    ...sScales(series, queries),
    n,
    toks,
    qtext,
    buckets: segments(toks),
    cand: null,
  };
  return loaded;
}

function candidateIndex(table, n) {
  const index = {};
  for (let row = 0; row < table.query.length; row++) {
    const token = table.token_index[row];
    const at = (index[table.query[row]] ??= {
      start: new Int32Array(n).fill(-1),
      end: new Int32Array(n).fill(-1),
    });
    at.start[token] = at.start[token] < 0 ? row : at.start[token];
    at.end[token] = row + 1;
  }
  return index;
}

export async function loadCandidates(client, chain) {
  const bytes = await client.artifact(chain.artifactUri, 'candidates.parquet');
  const table = await readColumns(bytes, CANDIDATE_COLUMNS);
  const candidates = { table, index: candidateIndex(table, chain.n) };
  return candidates;
}
