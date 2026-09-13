import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_PC28_BACKEND_BASE = "https://api.startpops.xyz";
const DEFAULT_HISTORY_PATH = "data/history.jsonl";
const PRESSURE_SERIES_LIMIT = 1000;
const RECENT_RESULTS_LIMIT = 100;
const ROOT_DIR = path.dirname(fileURLToPath(import.meta.url));

let historyCache = null;

function normalizeBaseUrl(value) {
  const raw = String(value || DEFAULT_PC28_BACKEND_BASE).trim();
  const withProtocol = /^[a-z][a-z\d+.-]*:\/\//i.test(raw) ? raw : `https://${raw}`;
  const url = new URL(withProtocol);
  url.hash = "";
  url.search = "";
  return url.toString().replace(/\/$/, "");
}

function resolvePc28BackendBase(env = process.env) {
  return normalizeBaseUrl(env.PC28_BACKEND_BASE_URL || env.PC28_BACKEND_BASE || DEFAULT_PC28_BACKEND_BASE);
}

function resolvePc28SignalUrl(env = process.env) {
  if (env.PC28_SIGNAL_URL) {
    return String(env.PC28_SIGNAL_URL).trim();
  }
  return new URL("/api/ai-signal", resolvePc28BackendBase(env)).toString();
}

export async function fetchPc28Signal(env = process.env) {
  const url = resolvePc28SignalUrl(env);
  try {
    return await fetchRemotePc28Signal(url);
  } catch (error) {
    if (env.PC28_DISABLE_HISTORY_FALLBACK === "true") {
      throw error;
    }
    return buildHistoryFallbackSignal(env, {
      upstream: url,
      reason: error.message || String(error)
    });
  }
}

async function fetchRemotePc28Signal(url) {
  const response = await fetch(url, {
    headers: {
      "Accept": "application/json",
      "Cache-Control": "no-cache",
      "Pragma": "no-cache",
      "User-Agent": "web-pc-proxy"
    }
  });

  if (!response.ok) {
    throw new Error(`PC-28 backend returned HTTP ${response.status}`);
  }

  const payload = await response.json();
  if (!isValidPc28SignalPayload(payload)) {
    throw new Error("PC-28 backend returned invalid ai-signal payload");
  }
  return payload;
}

function isValidPc28SignalPayload(payload) {
  return Boolean(payload && payload.feature && payload.feature.latest && payload.signal);
}

function buildHistoryFallbackSignal(env = process.env, details = {}) {
  const rows = loadHistoryRows(env);
  if (!rows.length) {
    throw new Error("local PC-28 history fallback has no usable rows");
  }

  const metrics = computeHistoryMetrics(rows);
  const latest = rows[rows.length - 1];
  const pressureScore = roundNumber(metrics.currentPressureScore);
  const zone = signalZone(pressureScore);

  return {
    ok: false,
    refresh_status: "fallback_history",
    source: "web-pc-local-history",
    upstream: details.upstream,
    fallback_reason: details.reason,
    updated_at: new Date().toISOString(),
    fetched_count: rows.length,
    feature: {
      latest,
      history_count: rows.length,
      probability: {
        straight_probability: roundNumber(metrics.straightProbability, 6),
        fair_straight_odds: metrics.straightProbability > 0 ? roundNumber(1 / metrics.straightProbability, 4) : 0,
        straight_odds: 18
      },
      pressure: {
        current_non_straight_streak: metrics.currentNonStraightStreak,
        avg_non_straight_gap: roundNumber(metrics.avgNonStraightGap),
        max_non_straight_gap: metrics.maxNonStraightGap,
        pressure_ratio_to_avg: metrics.avgNonStraightGap > 0 ? roundNumber(metrics.currentNonStraightStreak / metrics.avgNonStraightGap, 4) : 0,
        pressure_ratio_to_max: metrics.maxNonStraightGap > 0 ? roundNumber(metrics.currentNonStraightStreak / metrics.maxNonStraightGap, 4) : 0
      },
      conditional_probability: {
        observed: metrics.conditionalObserved,
        straight_count: metrics.conditionalStraightCount,
        straight_probability: metrics.conditionalObserved > 0 ? roundNumber(metrics.conditionalStraightCount / metrics.conditionalObserved, 6) : 0,
        mode: "local_history_same_miss_streak"
      },
      recent_density: {
        recent_50_rate: roundNumber(metrics.recent50.rate, 6),
        recent_100_rate: roundNumber(metrics.recent100.rate, 6),
        recent_50_straight_count: metrics.recent50.straightCount,
        recent_100_straight_count: metrics.recent100.straightCount
      },
      recent_results: rows.slice(-RECENT_RESULTS_LIMIT).reverse(),
      pressure_series: metrics.pressureSeries.slice(-PRESSURE_SERIES_LIMIT)
    },
    signal: {
      event_label: latest.is_straight ? "STRAIGHT_RESULT" : "NON_STRAIGHT_RESULT",
      next_signal: zone.nextSignal,
      signal_label: zone.label,
      risk_score: pressureScore,
      reason: "Remote API unavailable; serving local history fallback.",
      confidence: 0.25
    }
  };
}

function loadHistoryRows(env = process.env) {
  const historyPath = resolveHistoryPath(env.PC28_HISTORY_PATH || DEFAULT_HISTORY_PATH);
  const stat = fs.statSync(historyPath);
  if (historyCache && historyCache.path === historyPath && historyCache.mtimeMs === stat.mtimeMs) {
    return historyCache.rows;
  }

  const rows = fs.readFileSync(historyPath, "utf-8")
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => normalizeHistoryRow(JSON.parse(line)))
    .filter(Boolean);

  historyCache = {
    path: historyPath,
    mtimeMs: stat.mtimeMs,
    rows
  };
  return rows;
}

function resolveHistoryPath(value) {
  return path.isAbsolute(value) ? value : path.resolve(ROOT_DIR, value);
}

function normalizeHistoryRow(row) {
  const digits = parseDigits(row.digits || row.numbers);
  if (digits.length !== 3) {
    return null;
  }
  const sum = Number.isFinite(Number(row.sum)) ? Number(row.sum) : digits.reduce((total, digit) => total + digit, 0);
  const isStraight = coerceBoolean(row.is_straight ?? row.outcome_is_straight ?? row.straight);
  return {
    issue: String(row.issue || "").trim(),
    time: row.time,
    digits,
    sum,
    is_straight: isStraight ?? isStraightDigits(digits),
    pattern: row.pattern || classifyPattern(sum),
    raw: row.raw || `${digits.join("+")}=${sum}`
  };
}

function parseDigits(value) {
  if (Array.isArray(value)) {
    return value.map(Number).filter((digit) => Number.isInteger(digit));
  }
  if (typeof value === "string") {
    return value.split(/[,+\s]+/).map(Number).filter((digit) => Number.isInteger(digit));
  }
  return [];
}

function coerceBoolean(value) {
  if (typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    return value !== 0;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (["true", "1", "yes", "y"].includes(normalized)) {
      return true;
    }
    if (["false", "0", "no", "n"].includes(normalized)) {
      return false;
    }
  }
  return null;
}

function isStraightDigits(digits) {
  const sorted = [...new Set(digits)].sort((a, b) => a - b);
  return sorted.length === 3 && sorted[1] === sorted[0] + 1 && sorted[2] === sorted[1] + 1;
}

function classifyPattern(sum) {
  const size = sum >= 14 ? "big" : "small";
  const parity = sum % 2 === 0 ? "even" : "odd";
  return `${size}_${parity}`;
}

function computeHistoryMetrics(rows) {
  let currentMiss = 0;
  let straightCount = 0;
  let previousPressure = 0;
  const completedGaps = [];
  const preOmissions = [];
  const pressureSeries = [];

  rows.forEach((row, index) => {
    const priorBaseRate = index ? straightCount / index : 0.055;
    const avgGap = average(completedGaps) || Math.max(1, (1 / Math.max(priorBaseRate, 0.001)) - 1);
    const pressureScore = pressureFromMiss(currentMiss, avgGap);
    pressureSeries.push({
      issue: row.issue,
      value: roundNumber(currentMiss),
      risk_score: roundNumber(pressureScore),
      pressure_score: roundNumber(pressureScore),
      pressure_delta: roundNumber(pressureScore - previousPressure),
      is_straight: row.is_straight
    });
    preOmissions.push(currentMiss);
    previousPressure = pressureScore;

    if (row.is_straight) {
      straightCount += 1;
      completedGaps.push(currentMiss);
      currentMiss = 0;
    } else {
      currentMiss += 1;
    }
  });

  const avgNonStraightGap = average(completedGaps) || 0;
  const conditional = computeConditional(rows, preOmissions, currentMiss);

  return {
    currentNonStraightStreak: currentMiss,
    avgNonStraightGap,
    maxNonStraightGap: completedGaps.length ? Math.max(...completedGaps) : 0,
    currentPressureScore: pressureFromMiss(currentMiss, avgNonStraightGap || 1),
    straightProbability: rows.length ? straightCount / rows.length : 0,
    conditionalObserved: conditional.observed,
    conditionalStraightCount: conditional.straightCount,
    recent50: recentStraightDensity(rows, 50),
    recent100: recentStraightDensity(rows, 100),
    pressureSeries
  };
}

function computeConditional(rows, preOmissions, currentMiss) {
  let observed = 0;
  let straightCount = 0;
  rows.forEach((row, index) => {
    if (preOmissions[index] === currentMiss) {
      observed += 1;
      if (row.is_straight) {
        straightCount += 1;
      }
    }
  });
  return { observed, straightCount };
}

function recentStraightDensity(rows, windowSize) {
  const sample = rows.slice(-windowSize);
  const straightCount = sample.filter((row) => row.is_straight).length;
  return {
    straightCount,
    rate: sample.length ? straightCount / sample.length : 0
  };
}

function pressureFromMiss(currentMiss, avgGap) {
  return Math.min(100, 100 * (currentMiss / Math.max(avgGap * 2, 1)));
}

function average(values) {
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : 0;
}

function roundNumber(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}

function signalZone(score) {
  if (score >= 75) {
    return { nextSignal: "red", label: "RISK" };
  }
  if (score >= 50) {
    return { nextSignal: "yellow", label: "WATCH" };
  }
  return { nextSignal: "green", label: "PLAYER_ADVANTAGE" };
}

export {
  DEFAULT_PC28_BACKEND_BASE,
  buildHistoryFallbackSignal,
  resolvePc28BackendBase,
  resolvePc28SignalUrl
};
