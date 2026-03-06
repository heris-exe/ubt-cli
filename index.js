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

function createGeminiModel() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new CliError(
      'GEMINI_API_KEY environment variable is not set. Please set it before running this command.',
    );
  }

  const genAI = new GoogleGenerativeAI(apiKey);
  return genAI.getGenerativeModel({ model: 'gemini-1.5-pro' });
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