#!/usr/bin/env node

import process from 'node:process';
import path from 'node:path';
import os from 'node:os';
import fs from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { GoogleGenerativeAI } from '@google/generative-ai';

const execFileAsync = promisify(execFile);

const GIT_TIMEOUT_MS = 15_000;
const GEMINI_TIMEOUT_MS = 60_000;
const LOG_FILE_MAX_READ_BYTES = 5 * 1024 * 1024; // 5 MB
const MAX_DIFF_CHARS = 100_000; // ~100 KB to avoid huge prompts and API timeouts

function execFileWithTimeout(file, args, options, timeoutMs = GIT_TIMEOUT_MS) {
  return Promise.race([
    execFileAsync(file, args, options),
    new Promise((_, reject) =>
      setTimeout(
        () => reject(new Error(`Command timed out after ${timeoutMs}ms`)),
        timeoutMs,
      ),
    ),
  ]);
}

class CliError extends Error {
  constructor(message, exitCode = 1) {
    super(message);
    this.name = 'CliError';
    this.exitCode = exitCode;
  }
}

function printUsage() {
  console.log(`Usage:
  udt learn flag "concept" --file path/to/file`);
}

function parseLearnFlagArgs(args) {
  const positional = [];
  let filePath;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];

    if (arg === '--file') {
      const value = args[i + 1];
      if (!value) {
        throw new CliError('Missing value for --file.');
      }
      filePath = value;
      i += 1;
    } else if (arg.startsWith('--')) {
      throw new CliError(`Unknown option: ${arg}`);
    } else {
      positional.push(arg);
    }
  }

  if (positional.length === 0) {
    throw new CliError('Missing concept string. Provide it after "flag".');
  }

  if (!filePath) {
    throw new CliError('Missing required --file path.');
  }

  const concept = positional.join(' ');

  return { concept, filePath };
}

// Collect minimal git context (root, branch, short commit) for learn-flag metadata.
async function getGitMetadata(cwd) {
  const git = {};

  const { stdout: rootOut } = await execFileWithTimeout(
    'git',
    ['rev-parse', '--show-toplevel'],
    { cwd },
  );
  git.root = rootOut.trim();

  const { stdout: branchOut } = await execFileWithTimeout(
    'git',
    ['rev-parse', '--abbrev-ref', 'HEAD'],
    { cwd },
  );
  git.branch = branchOut.trim();

  const { stdout: commitOut } = await execFileWithTimeout(
    'git',
    ['rev-parse', '--short', 'HEAD'],
    { cwd },
  );
  git.shortCommit = commitOut.trim();

  return git;
}

async function getFileDiffAndMeta(filePath, cwd) {
  try {
    const { stdout, stderr } = await execFileWithTimeout(
      'git',
      ['diff', '--', filePath],
      { cwd },
    );
    const diff = stdout;
    let gitMeta = null;

    try {
      gitMeta = await getGitMetadata(cwd);
    } catch {
      gitMeta = null;
    }

    if (!diff.trim()) {
      return { diff: '', git: gitMeta };
    }

    if (stderr && stderr.trim()) {
      throw new CliError(`git diff error: ${stderr.trim()}`);
    }

    return { diff, git: gitMeta };
  } catch (err) {
    if (err instanceof CliError) {
      throw err;
    }

    if (err.code === 'ENOENT') {
      throw new CliError('git is not installed or not found in PATH.');
    }

    if (err.message && err.message.includes('timed out')) {
      throw new CliError(`Git command timed out. Check that the repo is not on a slow drive or stuck (e.g. credential helper). ${err.message}`);
    }

    const stderr = err.stderr || '';
    if (typeof stderr === 'string' && stderr.includes('Not a git repository')) {
      throw new CliError('learn flag currently requires running inside a git repository.');
    }

    throw new CliError(`git diff failed: ${stderr || err.message}`);
  }
}

async function getProjectInfo(cwd) {
  let gitRoot = null;

  try {
    const { stdout } = await execFileWithTimeout(
      'git',
      ['rev-parse', '--show-toplevel'],
      { cwd },
    );
    gitRoot = stdout.trim();
  } catch {
    gitRoot = null;
  }

  const projectRoot = gitRoot || cwd;
  const projectName = path.basename(projectRoot);

  return { projectName, projectRoot, gitRoot };
}

function getLogFilePath() {
  const override = process.env.UDT_LEARN_LOG_PATH;
  if (override && override.trim().length > 0) {
    return override.trim();
  }

  return path.join(os.homedir(), '.udt-cli', 'learn-flags.json');
}

async function logLearnFlag(entry) {
  const logFilePath = getLogFilePath();
  const logDir = path.dirname(logFilePath);

  await fs.mkdir(logDir, { recursive: true });

  let existing = [];

  try {
    const raw = await fs.readFile(logFilePath, 'utf8');
    if (raw.length > LOG_FILE_MAX_READ_BYTES) {
      console.error(
        `Warning: learn-flags log is very large (${(raw.length / 1024 / 1024).toFixed(1)} MB). Starting a fresh log to avoid freezing.`,
      );
    } else {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) {
        existing = parsed;
      }
    }
  } catch (err) {
    if (err.code !== 'ENOENT') {
      console.error(
        'Warning: failed to read existing learn flags log, starting a fresh log file.',
      );
    }
  }

  existing.push(entry);

  await fs.writeFile(logFilePath, JSON.stringify(existing, null, 2), 'utf8');

  return logFilePath;
}

async function openLearnFlagsInBrowser() {
  const logFilePath = getLogFilePath();
  let entries = [];

  try {
    const raw = await fs.readFile(logFilePath, 'utf8');
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      entries = parsed;
    }
  } catch (err) {
    if (err.code === 'ENOENT') {
      throw new CliError(
        `No learn flags found yet. Log file does not exist at: ${logFilePath}`,
      );
    }
    throw new CliError(`Failed to read learn flags log at ${logFilePath}: ${err.message}`);
  }

  const htmlPath = path.join(path.dirname(logFilePath), 'learn-flags.html');

  const html = `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>UDT Learn Flags</title>
    <style>
      :root {
        color-scheme: dark light;
        --bg: #050816;
        --bg-elevated: #0f172a;
        --border: #1f2937;
        --accent: #22c55e;
        --accent-soft: rgba(34, 197, 94, 0.1);
        --text: #e5e7eb;
        --text-muted: #9ca3af;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #1f2937 0, #020617 55%, #000 100%);
        color: var(--text);
        min-height: 100vh;
        padding: 24px;
      }
      .shell {
        max-width: 1080px;
        margin: 0 auto;
      }
      header {
        margin-bottom: 24px;
      }
      h1 {
        font-size: 28px;
        margin: 0 0 4px;
        letter-spacing: 0.04em;
      }
      .subtitle {
        font-size: 14px;
        color: var(--text-muted);
      }
      .controls {
        display: flex;
        gap: 12px;
        margin-bottom: 16px;
        flex-wrap: wrap;
      }
      .pill {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 999px;
        border: 1px solid var(--border);
        padding: 4px 10px;
        font-size: 12px;
        color: var(--text-muted);
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }
      .pill strong {
        color: var(--accent);
        font-weight: 600;
      }
      .search {
        flex: 1 1 220px;
        position: relative;
      }
      .search input {
        width: 100%;
        padding: 8px 10px 8px 26px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(15, 23, 42, 0.9);
        color: var(--text);
        font-size: 13px;
      }
      .search input::placeholder {
        color: var(--text-muted);
      }
      .search-icon {
        position: absolute;
        left: 9px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 11px;
        color: var(--text-muted);
      }
      .list {
        display: flex;
        flex-direction: column;
        gap: 10px;
      }
      .card {
        border-radius: 14px;
        border: 1px solid var(--border);
        background:
          radial-gradient(circle at top left, rgba(34, 197, 94, 0.18), transparent 55%),
          radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.16), transparent 60%),
          rgba(15, 23, 42, 0.96);
        padding: 12px 14px;
        cursor: pointer;
        transition: transform 120ms ease-out, box-shadow 120ms ease-out, border-color 120ms;
      }
      .card:hover {
        transform: translateY(-1px);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.6);
        border-color: rgba(34, 197, 94, 0.6);
      }
      .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
      }
      .concept {
        font-weight: 600;
        font-size: 14px;
      }
      .meta {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 4px;
        font-size: 11px;
        color: var(--text-muted);
      }
      .meta span {
        padding: 2px 7px;
        border-radius: 999px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.8);
      }
      .timestamp {
        font-size: 11px;
        color: var(--text-muted);
      }
      .badge {
        padding: 3px 8px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 11px;
        border: 1px solid rgba(34, 197, 94, 0.5);
      }
      .body {
        margin-top: 10px;
        font-size: 13px;
        line-height: 1.6;
        color: #e5e7eb;
      }
      .body pre {
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
          "Courier New", monospace;
      }
      .resources {
        margin-top: 8px;
        padding-top: 6px;
        border-top: 1px dashed rgba(148, 163, 184, 0.5);
      }
      .resources h3 {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
        margin: 0 0 4px;
      }
      .resources ul {
        list-style: none;
        margin: 0;
        padding: 0;
        display: grid;
        gap: 3px;
      }
      .resources a {
        color: var(--accent);
        text-decoration: none;
        font-size: 12px;
      }
      .resources a:hover {
        text-decoration: underline;
      }
      .empty-state {
        margin-top: 40px;
        text-align: center;
        color: var(--text-muted);
        font-size: 14px;
      }
      .empty-state strong {
        color: var(--accent);
      }
      .count {
        font-size: 12px;
        color: var(--text-muted);
        margin-bottom: 6px;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <header>
        <h1>UDT Learn Flags</h1>
        <div class="subtitle">Captured explanations and resources from your coding sessions.</div>
      </header>
      <div class="controls">
        <div class="pill">
          <span>Log file:</span>
          <strong>${logFilePath.replace(/\\/g, '/')}</strong>
        </div>
        <div class="search">
          <span class="search-icon">⌕</span>
          <input id="search" type="search" placeholder="Filter by concept, file, or explanation..." />
        </div>
      </div>
      <div id="count" class="count"></div>
      <div id="list" class="list"></div>
      <div id="empty" class="empty-state" style="display:none;">
        <p>No learn flags match your filter.</p>
        <p>Try broadening your search or run <strong>udt learn flag</strong> again.</p>
      </div>
    </div>
    <script>
      window.LEARN_FLAGS = ${JSON.stringify(entries)};

      function formatDate(iso) {
        if (!iso) return '';
        const d = new Date(iso);
        if (Number.isNaN(d.getTime())) return iso;
        return d.toLocaleString();
      }

      function normalize(text) {
        return (text || '').toLowerCase();
      }

      function render(flags, query) {
        const list = document.getElementById('list');
        const empty = document.getElementById('empty');
        const count = document.getElementById('count');
        list.innerHTML = '';

        let filtered = flags;
        const q = normalize(query);
        if (q) {
          filtered = flags.filter((f) => {
            const haystack = [
              f.concept,
              f.filePath,
              f.project && f.project.projectName,
              f.explanation,
            ]
              .filter(Boolean)
              .join(' ')
              .toLowerCase();
            return haystack.includes(q);
          });
        }

        count.textContent = filtered.length
          ? filtered.length + ' flag' + (filtered.length === 1 ? '' : 's')
          : 'No flags to display';

        if (!filtered.length) {
          empty.style.display = 'block';
          return;
        }

        empty.style.display = 'none';

        filtered
          .slice()
          .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
          .forEach((entry) => {
            const card = document.createElement('article');
            card.className = 'card';

            const header = document.createElement('div');
            header.className = 'card-header';

            const left = document.createElement('div');
            const concept = document.createElement('div');
            concept.className = 'concept';
            concept.textContent = entry.concept || '(no concept)';
            left.appendChild(concept);

            const meta = document.createElement('div');
            meta.className = 'meta';
            if (entry.filePath) {
              const file = document.createElement('span');
              file.textContent = entry.filePath;
              meta.appendChild(file);
            }
            if (entry.project && entry.project.projectName) {
              const proj = document.createElement('span');
              proj.textContent = entry.project.projectName;
              meta.appendChild(proj);
            }
            if (entry.git && entry.git.branch) {
              const branch = document.createElement('span');
              branch.textContent = 'branch: ' + entry.git.branch;
              meta.appendChild(branch);
            }
            left.appendChild(meta);

            const right = document.createElement('div');
            right.style.display = 'flex';
            right.style.flexDirection = 'column';
            right.style.alignItems = 'flex-end';
            right.style.gap = '4px';

            const ts = document.createElement('div');
            ts.className = 'timestamp';
            ts.textContent = formatDate(entry.timestamp);
            right.appendChild(ts);

            const badge = document.createElement('div');
            badge.className = 'badge';
            badge.textContent = 'learn flag';
            right.appendChild(badge);

            header.appendChild(left);
            header.appendChild(right);
            card.appendChild(header);

            if (entry.explanation) {
              const body = document.createElement('div');
              body.className = 'body';
              const pre = document.createElement('pre');
              pre.textContent = entry.explanation;
              body.appendChild(pre);
              card.appendChild(body);
            }

            if (Array.isArray(entry.resources) && entry.resources.length) {
              const res = document.createElement('div');
              res.className = 'resources';
              const h3 = document.createElement('h3');
              h3.textContent = 'Resources';
              res.appendChild(h3);
              const ul = document.createElement('ul');
              entry.resources.forEach((r) => {
                if (!r || !r.url) return;
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = r.url;
                a.target = '_blank';
                a.rel = 'noreferrer';
                a.textContent = r.title || r.url;
                li.appendChild(a);
                ul.appendChild(li);
              });
              res.appendChild(ul);
              card.appendChild(res);
            }

            list.appendChild(card);
          });
      }

      (function init() {
        const input = document.getElementById('search');
        const flags = Array.isArray(window.LEARN_FLAGS) ? window.LEARN_FLAGS : [];
        render(flags, '');
        input.addEventListener('input', () => {
          render(flags, input.value || '');
        });
      })();
    </script>
  </body>
</html>`;

  await fs.writeFile(htmlPath, html, 'utf8');

  const platform = process.platform;
  const opener =
    platform === 'win32' ? 'explorer' : platform === 'darwin' ? 'open' : 'xdg-open';
  const args = [htmlPath];

  try {
    await execFileAsync(opener, args);
  } catch (err) {
    console.error(
      `Failed to automatically open browser. Open this file manually: ${htmlPath}`,
    );
  }

  console.log(`Rendered learn flags dashboard to: ${htmlPath}`);
}

function createGeminiModel() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new CliError(
      'GEMINI_API_KEY environment variable is not set. Please set it before running this command.',
    );
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  const modelId = process.env.GEMINI_MODEL || 'gemini-1.5-pro';
  return genAI.getGenerativeModel(
    { model: modelId },
    { apiVersion: 'v1' },
  );
}

async function explainConceptWithGemini(model, payload) {
  const { concept, diff, filePath, project, cwd } = payload;

  const prompt = [
    'You are helping a developer learn while coding based on their actual git diff.',
    '',
    `Project name: ${project.projectName}`,
    `Project root: ${project.projectRoot}`,
    `Current working directory: ${cwd}`,
    `File path: ${filePath}`,
    '',
    `Concept to explain: ${concept}`,
    '',
    'Here is the git diff for this file:',
    '',
    '```diff',
    diff,
    '```',
    '',
    'Explain the concept in the context of what is changing in this diff and how it relates to the surrounding code.',
    'Then provide a short list of external resources (with URLs and one-line descriptions) for further reading.',
    '',
    'Respond ONLY with minified JSON in this exact shape (no markdown fences, no extra text):',
    '{"explanation":"...markdown explanation...","resources":[{"title":"...","url":"...","type":"doc|video|article|spec|other"}]}',
  ].join('\n');

  let timeoutId;
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = setTimeout(
      () => reject(new CliError(`Gemini API timed out after ${GEMINI_TIMEOUT_MS}ms. Try again or check your network.`, 1)),
      GEMINI_TIMEOUT_MS,
    );
  });
  const result = await Promise.race([
    model.generateContent(prompt).then((res) => {
      clearTimeout(timeoutId);
      return res;
    }),
    timeoutPromise,
  ]);
  const text = result.response.text().trim();

  try {
    const parsed = JSON.parse(text);
    const explanation = typeof parsed.explanation === 'string' ? parsed.explanation : text;
    const resources = Array.isArray(parsed.resources) ? parsed.resources : [];
    return { explanation, resources };
  } catch {
    return { explanation: text, resources: [] };
  }
}

async function handleLearnFlag(args) {
  let parsed;

  try {
    parsed = parseLearnFlagArgs(args);
  } catch (err) {
    if (err instanceof CliError) {
      console.error(err.message);
      printUsage();
      process.exit(err.exitCode);
      return;
    }

    throw err;
  }

  const cwd = process.cwd();
  const relativeFilePath = parsed.filePath;
  const absoluteFilePath = path.resolve(cwd, relativeFilePath);

  const project = await getProjectInfo(cwd);

  const { diff, git } = await getFileDiffAndMeta(relativeFilePath, cwd);

  if (!diff || !diff.trim()) {
    console.log('No changes found for that file; nothing to flag.');
    return;
  }

  let diffForApi = diff;
  if (diff.length > MAX_DIFF_CHARS) {
    console.warn(
      `Warning: diff is large (${(diff.length / 1024).toFixed(0)} KB). Truncating to ${MAX_DIFF_CHARS} chars to avoid timeouts.`,
    );
    diffForApi = diff.slice(0, MAX_DIFF_CHARS) + '\n\n... (truncated)';
  }

  const model = createGeminiModel();
  const { explanation, resources } = await explainConceptWithGemini(model, {
    concept: parsed.concept,
    diff: diffForApi,
    filePath: relativeFilePath,
    project,
    cwd,
  });

  const entry = {
    timestamp: new Date().toISOString(),
    concept: parsed.concept,
    filePath: relativeFilePath,
    absoluteFilePath,
    cwd,
    project,
    git,
    diff,
    explanation,
    resources,
  };

  const logFilePath = await logLearnFlag(entry);

  console.log(
    `Saved learn flag for concept "${parsed.concept}" in project "${project.projectName}" (file: ${relativeFilePath}).`,
  );
  console.log(`Log file: ${logFilePath}`);
}

async function main() {
  const args = process.argv.slice(2);
  const [command, subcommand, ...rest] = args;

  if (command === 'learn' && subcommand === 'flag') {
    await handleLearnFlag(rest);
    return;
  }

  if (command === 'learn' && subcommand === 'open') {
    await openLearnFlagsInBrowser();
    return;
  }

  printUsage();
  process.exitCode = 1;
}

main().catch((err) => {
  if (err instanceof CliError) {
    console.error(err.message);
    process.exit(err.exitCode);
    return;
  }

  console.error('Unexpected error:', err);
  process.exit(1);
});