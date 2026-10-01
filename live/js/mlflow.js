const SERVER = 'https://mlflow.mi2.ai';
const STORE = 'lm-mindreader.mlflow';
const PREFIX = 'lm-mindreader/';
const PROXIED = 'mlflow-artifacts:/';

const REASONS = {
  401: 'The username or password was not accepted.',
  403: 'This account may not read these experiments.',
  503: 'The MLflow server is unavailable right now.',
};

export class MlflowError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

function basicAuthorization(user, password) {
  const bytes = new TextEncoder().encode(`${user}:${password}`);
  const binary = Array.from(bytes, (byte) => String.fromCharCode(byte)).join('');
  const header = `Basic ${btoa(binary)}`;
  return header;
}

function artifactUrl(artifactUri, path) {
  if (!artifactUri.startsWith(PROXIED)) {
    throw new MlflowError(`Files at ${artifactUri} are not served through MLflow.`, 0);
  }

  const url = `${SERVER}/api/2.0/mlflow-artifacts/artifacts/${artifactUri.slice(PROXIED.length)}/${path}`;
  return url;
}

export function forgetLogin() {
  sessionStorage.removeItem(STORE);
}

export class Mlflow {
  constructor(user, authorization) {
    this.user = user;
    this.authorization = authorization;
  }

  static async login(user, password) {
    const client = new Mlflow(user, basicAuthorization(user, password));
    await client.experiments();
    sessionStorage.setItem(STORE, JSON.stringify({ user, authorization: client.authorization }));
    return client;
  }

  static restore() {
    const saved = sessionStorage.getItem(STORE);
    if (!saved) {
      return null;
    }

    const { user, authorization } = JSON.parse(saved);
    const client = new Mlflow(user, authorization);
    return client;
  }

  async send(url, options = {}) {
    const headers = { ...options.headers, Authorization: this.authorization };
    let response;
    try {
      response = await fetch(url, { ...options, headers });
    } catch {
      throw new MlflowError(`${SERVER} could not be reached, or does not yet admit ${location.origin}: it has to be in the server's allowed CORS origins.`, 0);
    }
    if (!response.ok) {
      const reason = REASONS[response.status] ?? `MLflow answered ${response.status}.`;
      throw new MlflowError(reason, response.status);
    }

    return response;
  }

  async post(path, body) {
    const response = await this.send(`${SERVER}/api/2.0/mlflow/${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const answer = await response.json();
    return answer;
  }

  async experiments() {
    const answer = await this.post('experiments/search', {
      filter: `name LIKE '${PREFIX}%'`,
      max_results: 1000,
    });
    const found = (answer.experiments ?? []).map(({ experiment_id, name }) => ({ id: experiment_id, name }));
    found.sort((a, b) => a.name.localeCompare(b.name));
    return found;
  }

  async runs(experimentId, filter) {
    const runs = [];
    let pageToken;
    do {
      const answer = await this.post('runs/search', {
        experiment_ids: [experimentId],
        filter,
        max_results: 1000,
        page_token: pageToken,
      });
      runs.push(...(answer.runs ?? []));
      pageToken = answer.next_page_token;
    } while (pageToken);
    return runs;
  }

  async artifact(artifactUri, path) {
    const response = await this.send(artifactUrl(artifactUri, path));
    const bytes = await response.arrayBuffer();
    return bytes;
  }

  async artifactJson(artifactUri, path) {
    const response = await this.send(artifactUrl(artifactUri, path));
    const parsed = await response.json();
    return parsed;
  }
}
