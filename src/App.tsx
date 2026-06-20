import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from "react";
import {
  Activity,
  AlertCircle,
  Bot,
  ChevronLeft,
  ChevronRight,
  Database,
  Gauge as GaugeIcon,
  RefreshCw,
  ShieldAlert,
  Target,
  Zap
} from "lucide-react";

type Draw = {
  issue: string;
  time?: string;
  digits: number[];
  sum: number;
  is_straight: boolean;
  raw?: string;
};

type PressurePoint = {
  issue: string;
  value: number;
  is_straight: boolean;
};

type Pc28Payload = {
  ok?: boolean;
  refresh_status?: string;
  updated_at?: string;
  fetched_count?: number;
  feature: {
    latest: Draw;
    history_count: number;
    probability: {
      straight_probability: number;
      fair_straight_odds: number;
      straight_odds: number;
    };
    pressure: {
      current_non_straight_streak: number;
      avg_non_straight_gap: number;
      max_non_straight_gap: number;
      pressure_ratio_to_avg: number;
      pressure_ratio_to_max: number;
    };
    conditional_probability: {
      observed: number;
      straight_count: number;
      straight_probability: number;
      mode: string;
    };
    recent_density: {
      recent_50_rate: number;
      recent_100_rate: number;
      recent_50_straight_count: number;
      recent_100_straight_count: number;
    };
    aftershock?: {
      current_stage?: number;
      current_stage_label?: string;
      current_stage_rate?: number;
      current_stage_observed?: number;
      current_stage_deviation?: number;
      current_stage_signal?: string;
      next_1_observed?: number;
      next_1_straight_count?: number;
      next_1_rate?: number;
      next_3_observed?: number;
      next_3_straight_count?: number;
      next_3_rate?: number;
      next_5_observed?: number;
      next_5_straight_count?: number;
      next_5_rate?: number;
    };
    recent_results: Draw[];
    pressure_series: PressurePoint[];
  };
  signal: {
    event_label: string;
    next_signal: "green" | "yellow" | "red";
    signal_label: string;
    risk_score: number;
    reason: string;
    confidence: number;
  };
  prediction?: {
    next_judgment?: {
      label: string;
      hit_rate: number;
      score: number;
      sample_size: number;
      hit_count: number;
    };
    targets?: {
      straight?: {
        score: number;
        sample_size: number;
        backtest_next_hit_rate: number;
        reason: string;
      };
      non_straight?: {
        score: number;
        sample_size: number;
        backtest_next_hit_rate: number;
        reason: string;
      };
    };
  };
};

type TrendPoint = {
  issue: string;
  value: number;
  isStraight: boolean;
  x: number;
  y: number;
  trendValue: number;
  riskScore: number;
  zoneKey: ZoneKey;
  zoneLabel: string;
};

type ChartPointer = {
  clientX: number;
  clientY: number;
};

type ZoneKey = "player" | "watch" | "risk";

type ChartSignal = {
  score: number;
  zone: ReturnType<typeof pressureZone>;
  latestRisk: number;
  weightedRisk: number;
  playerShare: number;
  watchShare: number;
  riskShare: number;
};

const API_URL = "/api/pc28-signal";
const REFRESH_MS = 10000;
const CHART_HIT_LIFT = 17;
const CHART_MISS_STEP = 1;
const ZOOM_PRESETS = [100, 200, 400, 800, 1000];
const CHART_MIN_WINDOW = 60;
const CHART_MAX_WINDOW = 1000;
const CHART_WIDTH = 1200;
const CHART_HEIGHT = 468;
const ACTUAL_RATE_WINDOWS = [1000, 800, 400];
const CHART_MISS_COLOR = "#22d3ee";
const CHART_HIT_COLOR = "#E54B4B";

function formatPercent(value: number | undefined, digits = 1) {
  return `${(Number(value || 0) * 100).toFixed(digits)}%`;
}

function formatTime(value?: string) {
  if (!value) {
    return "--";
  }
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

function formatDateTime(value?: string | Date) {
  if (!value) {
    return "--";
  }
  if (typeof value === "string") {
    const drawTime = value.match(/^(\d{2})-(\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?$/);
    if (drawTime) {
      return `${drawTime[1]}/${drawTime[2]} ${drawTime[3]}:${drawTime[4]}:${drawTime[5] || "00"}`;
    }
  }
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }
  return date.toLocaleString([], {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

function formatIssue(issue?: string) {
  return issue ? `#${issue}` : "--";
}

function eventText(draw?: Draw) {
  return draw?.is_straight ? "出顺：客户输" : "未出：客户赢";
}

function formatDrawFormula(draw?: Draw) {
  if (!draw?.digits?.length) {
    return "-- + -- + -- = --";
  }

  return `${draw.digits.map((digit) => String(digit).padStart(2, "0")).join(" + ")} = ${String(draw.sum ?? 0).padStart(2, "0")}`;
}

function DrawDigits({ draw, compact = false }: { draw?: Draw; compact?: boolean }) {
  const digits = draw?.digits?.length ? draw.digits : [0, 0, 0];
  const chipSize = compact ? "min-w-6 h-6 px-1.5 text-[10px]" : "min-w-8 h-8 px-2 text-xs";
  const operatorClass = compact ? "text-[10px] px-0" : "text-xs px-0.5";
  return (
    <div className="flex flex-wrap items-center gap-1">
      {digits.map((digit, index) => (
        <div key={`${index}-${digit}`} className="contents">
          <span
            className={`${chipSize} draw-chip-terminal inline-flex items-center justify-center font-bold border ${
              draw?.is_straight
                ? "bg-[#2a1212] border-[#E54B4B]/70 text-[#ffb4ab]"
                : "bg-[#1e2022] border-[#414754]/55 text-white"
            }`}
          >
            {String(digit).padStart(2, "0")}
          </span>
          {index < digits.length - 1 ? (
            <span className={`${operatorClass} text-[#777] font-black`}>+</span>
          ) : null}
        </div>
      ))}
      <span className={`${operatorClass} text-[#777] font-black`}> = </span>
      <span
        className={`${compact ? "min-w-7 h-6 px-1.5 text-[10px]" : "min-w-9 h-8 px-2 text-xs"} draw-chip-terminal draw-sum-terminal inline-flex items-center justify-center border border-[#4a8eff]/45 bg-[#1a1c1e] text-[#abc7ff] font-bold`}
      >
        {draw ? String(draw.sum).padStart(2, "0") : "--"}
      </span>
    </div>
  );
}

function signalClass(signal?: string) {
  if (signal === "red") {
    return "text-[#ff9e99] border-[#E54B4B]/30 bg-[#E54B4B]/10";
  }
  if (signal === "green") {
    return "text-green-300 border-green-500/20 bg-green-500/10";
  }
  return "text-amber-300 border-amber-500/20 bg-amber-500/10";
}

function robotBiasClass(label?: string) {
  if (!label) {
    return "text-[#888]";
  }
  if (label.includes("非")) {
    return "text-amber-300";
  }
  if (label.includes("顺")) {
    return "text-[#ff9e99]";
  }
  return "text-white";
}

function robotBiasBgClass(label?: string) {
  if (!label) {
    return "";
  }
  if (label.includes("非")) {
    return "robot-bias-non-straight";
  }
  if (label.includes("顺")) {
    return "robot-bias-straight";
  }
  return "";
}

function pressureZone(score: number): {
  key: ZoneKey;
  label: string;
  className: string;
  dot: string;
} {
  if (score >= 75) {
    return {
      key: "risk",
      label: "风险区",
      className: "text-[#ff9e99] border-[#E54B4B]/30 bg-[#E54B4B]/10",
      dot: "bg-[#E54B4B]"
    };
  }
  if (score >= 50) {
    return {
      key: "watch",
      label: "临界观察区",
      className: "text-amber-300 border-amber-500/30 bg-amber-500/10",
      dot: "bg-amber-500"
    };
  }
  return {
    key: "player",
    label: "玩家优势区",
    className: "text-green-300 border-green-500/25 bg-green-500/10",
    dot: "bg-green-500"
  };
}

function zoneTextClass(key: string) {
  if (key === "risk") {
    return "text-[#ff9e99]";
  }
  if (key === "watch") {
    return "text-amber-300";
  }
  return "text-green-300";
}

function zoneBgClass(key: string) {
  if (key === "risk") {
    return "bg-[#E54B4B]/10 border-[#E54B4B]/25";
  }
  if (key === "watch") {
    return "bg-amber-500/10 border-amber-500/25";
  }
  return "bg-green-500/10 border-green-500/20";
}

function zoneFill(key: ZoneKey) {
  if (key === "risk") {
    return "#E54B4B";
  }
  if (key === "watch") {
    return "#f59e0b";
  }
  return "#22c55e";
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function compoundStraightProbability(singleRate: number, periods: number) {
  const safeRate = clamp(singleRate, 0, 1);
  return 1 - Math.pow(1 - safeRate, periods);
}

function straightRateForWindow(series: PressurePoint[], windowSize: number) {
  const sample = series.slice(-Math.min(windowSize, series.length));
  const straightCount = sample.filter((point) => point.is_straight).length;
  return {
    windowSize,
    sampleSize: sample.length,
    straightCount,
    rate: sample.length ? straightCount / sample.length : 0
  };
}

function forecastHelper(count?: number, observed?: number, fallback = "") {
  const safeObserved = Number(observed || 0);
  if (safeObserved > 0) {
    return `${Number(count || 0)}/${safeObserved} 样本`;
  }
  return fallback;
}

function calibratedRiskScore(
  missValue: number,
  currentMiss: number,
  avgGap: number,
  maxGap: number,
  pressureScore: number,
  conditionalProbability: number,
  baseProbability: number
) {
  const rawFor = (value: number) => {
    const avgRatio = avgGap > 0 ? value / avgGap : 0;
    const maxRatio = maxGap > 0 ? value / maxGap : 0;
    const conditionLift = baseProbability > 0 ? conditionalProbability / baseProbability : 1;
    return avgRatio * 48 + maxRatio * 30 + clamp(conditionLift, 0.5, 1.8) * 12;
  };
  const currentRaw = rawFor(currentMiss);
  const scale = currentRaw > 0 && pressureScore > 0 ? pressureScore / currentRaw : 1;
  return Math.round(clamp(rawFor(missValue) * scale, 0, 100));
}

function buildTrend(
  series: PressurePoint[],
  context: {
    currentMiss: number;
    avgGap: number;
    maxGap: number;
    pressureScore: number;
    conditionalProbability: number;
    baseProbability: number;
  }
): TrendPoint[] {
  let trendValue = 0;
  const raw = series.map((point) => {
    if (point.is_straight) {
      trendValue += CHART_HIT_LIFT;
    } else {
      trendValue -= CHART_MISS_STEP;
    }
    return {
      issue: point.issue,
      value: point.value,
      isStraight: point.is_straight,
      riskScore: calibratedRiskScore(
        point.value,
        context.currentMiss,
        context.avgGap,
        context.maxGap,
        context.pressureScore,
        context.conditionalProbability,
        context.baseProbability
      ),
      trendValue,
      x: 0,
      y: 0
    };
  });

  if (!raw.length) {
    return [];
  }

  const values = raw.map((point) => point.trendValue);
  const max = Math.max(...values, 8);
  const min = Math.min(...values, -8);
  const range = Math.max(1, max - min);

  return raw.map((point, index) => {
    const zone = pressureZone(point.riskScore);
    return {
      ...point,
      zoneKey: zone.key,
      zoneLabel: zone.label,
      x: raw.length > 1 ? (index / (raw.length - 1)) * CHART_WIDTH : CHART_WIDTH / 2,
      y: 20 + ((max - point.trendValue) / range) * (CHART_HEIGHT - 40)
    };
  });
}

function calculateChartSignal(points: TrendPoint[], latestPoint?: TrendPoint | null): ChartSignal {
  if (!points.length) {
    const zone = pressureZone(0);
    return {
      score: 0,
      zone,
      latestRisk: 0,
      weightedRisk: 0,
      playerShare: 0,
      watchShare: 0,
      riskShare: 0
    };
  }

  const totals = points.reduce(
    (acc, point, index) => {
      const weight = 1 + (index / Math.max(1, points.length - 1)) * 1.8;
      acc.weight += weight;
      acc.weightedRisk += point.riskScore * weight;
      acc[point.zoneKey] += 1;
      return acc;
    },
    { weight: 0, weightedRisk: 0, player: 0, watch: 0, risk: 0 }
  );
  const latestRisk = latestPoint?.riskScore ?? points[points.length - 1].riskScore;
  const weightedRisk = totals.weight ? totals.weightedRisk / totals.weight : latestRisk;
  const score = Math.round(clamp(latestRisk * 0.68 + weightedRisk * 0.32, 0, 100));

  return {
    score,
    zone: pressureZone(score),
    latestRisk,
    weightedRisk,
    playerShare: totals.player / points.length,
    watchShare: totals.watch / points.length,
    riskShare: totals.risk / points.length
  };
}

function MetricCard({
  label,
  value,
  suffix,
  footer,
  hot
}: {
  label: string;
  value: string | number;
  suffix?: string;
  footer: string;
  hot?: boolean;
}) {
  return (
    <div className={`bg-[#0a0a0a] border ${hot ? "border-[#E54B4B]/30" : "border-[#1a1a1a]"} p-4 rounded-sm flex flex-col justify-between`}>
      <div className="text-[11px] text-[#888] uppercase tracking-wider font-sans flex justify-between items-center">
        <span>{label}</span>
        {hot ? <AlertCircle className="w-3.5 h-3.5 text-[#E54B4B]" /> : null}
      </div>
      <div className="mt-2 flex items-baseline gap-2">
        <span className={`text-3xl font-light font-mono tracking-tight ${hot ? "text-[#ff9e99]" : "text-white"}`}>
          {value}
        </span>
        {suffix ? <span className="text-xs text-[#666]">{suffix}</span> : null}
      </div>
      <div className="text-[10px] text-[#666] font-sans border-t border-[#151515] pt-2 mt-2">
        {footer}
      </div>
    </div>
  );
}

function HistoryTable({ results }: { results: Draw[] }) {
  return (
    <div className="history-shell bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm overflow-hidden">
      <div className="bg-[#0f0f0f] border-b border-[#1a1a1a] px-4 py-2.5 flex justify-between items-center">
        <span className="text-xs font-bold uppercase tracking-wider text-white font-sans">历史开奖 HISTORY</span>
        <span className="text-[10px] text-[#888]">最近 20 期</span>
      </div>
      <div className="history-table-view overflow-x-auto text-xs select-none">
        <table className="w-full min-w-[680px] text-left">
          <thead className="bg-[#050505] text-[#888] font-sans text-[10px] border-b border-[#1a1a1a]">
            <tr>
              <th className="p-2 py-2 pl-4">期号</th>
              <th className="p-2 py-2">开奖号</th>
              <th className="p-2 py-2 pr-4 text-right">状态</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#1a1a1a]">
            {results.slice(0, 20).map((item, index) => (
              <tr key={item.issue} className={`${index % 2 === 0 ? "bg-[#0a0a0a]" : "bg-[#070707]"} hover:bg-[#151515]`}>
                <td className="p-2 pl-4 font-mono text-white text-[11px]">{formatIssue(item.issue)}</td>
                <td className="p-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <DrawDigits draw={item} compact />
                    <span className="font-mono text-[#777] text-[10px]">{formatDrawFormula(item)}</span>
                  </div>
                </td>
                <td className="p-2 pr-4 text-right font-mono">
                  {item.is_straight ? (
                    <span className="bg-[#E54B4B]/10 text-[#ffb4ab] border border-[#E54B4B]/20 px-1.5 py-0.5 rounded-sm text-[10px] font-bold">出顺</span>
                  ) : (
                    <span className="text-zinc-600 text-[10px]">未出</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="history-mobile-list">
        {results.slice(0, 20).map((item) => (
          <div key={item.issue} className="history-mobile-item">
            <div className="history-mobile-head">
              <span>{formatIssue(item.issue)}</span>
              {item.is_straight ? <strong>出顺</strong> : <em>未出</em>}
            </div>
            <div className="history-mobile-draw">
              <DrawDigits draw={item} compact />
              <span>{formatDrawFormula(item)}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function PressureGauge({
  score,
  zone,
  avgRatio,
  maxRatio,
  conditionalRate
}: {
  score: number;
  zone: ReturnType<typeof pressureZone>;
  avgRatio: number;
  maxRatio: number;
  conditionalRate: number;
}) {
  const radius = 86;
  const centerX = 120;
  const centerY = 118;
  const circumference = Math.PI * radius;
  const progress = clamp(score, 0, 100) / 100;
  const offset = circumference * (1 - progress);
  const needleAngle = -180 + progress * 180;
  const needleRadians = (needleAngle * Math.PI) / 180;
  const needleX = centerX + Math.cos(needleRadians) * 70;
  const needleY = centerY + Math.sin(needleRadians) * 70;

  return (
    <div className="pressure-gauge-shell">
      <svg className="w-full max-w-[270px] mx-auto block" viewBox="0 0 240 150" aria-label="压力指数仪表">
        <path
          d={`M ${centerX - radius} ${centerY} A ${radius} ${radius} 0 0 1 ${centerX + radius} ${centerY}`}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="18"
          strokeLinecap="round"
        />
        <path
          d={`M ${centerX - radius} ${centerY} A ${radius} ${radius} 0 0 1 ${centerX + radius} ${centerY}`}
          fill="none"
          stroke={zoneFill(zone.key)}
          strokeWidth="18"
          strokeLinecap="round"
          strokeDasharray={`${circumference} ${circumference}`}
          strokeDashoffset={offset}
          className="pressure-gauge-progress"
        />
        {[0, 25, 50, 75, 100].map((tick) => {
          const angle = (-180 + tick * 1.8) * Math.PI / 180;
          const outerX = centerX + Math.cos(angle) * 98;
          const outerY = centerY + Math.sin(angle) * 98;
          const innerX = centerX + Math.cos(angle) * 82;
          const innerY = centerY + Math.sin(angle) * 82;
          return (
            <line
              key={tick}
              x1={innerX}
              y1={innerY}
              x2={outerX}
              y2={outerY}
              stroke="rgba(255,255,255,0.28)"
              strokeWidth={tick % 50 === 0 ? "1.8" : "1"}
            />
          );
        })}
        <line
          x1={centerX}
          y1={centerY}
          x2={needleX}
          y2={needleY}
          stroke="#ffffff"
          strokeWidth="2.4"
          strokeLinecap="round"
          className="pressure-gauge-needle"
        />
        <circle cx={centerX} cy={centerY} r="5.5" fill="#ffffff" />
        <text x="36" y="139" fill="#666" fontSize="9" fontFamily="JetBrains Mono, monospace" fontWeight="700">0</text>
        <text x="110" y="41" fill="#666" fontSize="9" fontFamily="JetBrains Mono, monospace" fontWeight="700">50</text>
        <text x="196" y="139" fill="#666" fontSize="9" fontFamily="JetBrains Mono, monospace" fontWeight="700">100</text>
      </svg>

      <div className="pressure-readout">
        <div>
          <span className="text-[10px] text-[#777] font-sans uppercase tracking-wider">PRESSURE SCORE</span>
          <div className="flex items-baseline gap-2">
            <strong className={`text-5xl font-black leading-none ${zoneTextClass(zone.key)}`}>{score}</strong>
            <span className="text-xs text-[#666]">/100</span>
          </div>
        </div>
        <div className={`border px-2.5 py-2 rounded-sm min-w-[108px] text-right ${zoneBgClass(zone.key)}`}>
          <div className="text-[9px] text-[#888] font-sans uppercase tracking-wider">OUTPUT</div>
          <div className={`text-sm font-extrabold font-sans ${zoneTextClass(zone.key)}`}>{zone.label}</div>
        </div>
      </div>

      <div className="pressure-spark-grid">
        {[
          ["AVG", avgRatio.toFixed(2)],
          ["MAX", avgRatio || maxRatio ? maxRatio.toFixed(2) : "0.00"],
          ["COND", formatPercent(conditionalRate)]
        ].map(([label, value]) => (
          <div key={label} className="pressure-mini-cell">
            <span>{label}</span>
            <strong>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [payload, setPayload] = useState<Pc28Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [zoomRange, setZoomRange] = useState(1000);
  const [viewEndIndex, setViewEndIndex] = useState<number | null>(null);
  const [hovered, setHovered] = useState<TrendPoint | null>(null);
  const [localNow, setLocalNow] = useState(() => new Date());
  const chartPointers = useRef<Map<number, ChartPointer>>(new Map());
  const chartPinch = useRef({ active: false, distance: 0, startRange: 1000 });
  const chartDrag = useRef({ active: false, startX: 0, startEndIndex: 0, moved: false });

  const loadSignal = async () => {
    try {
      const response = await fetch(API_URL, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`API HTTP ${response.status}`);
      }
      const nextPayload = await response.json();
      if (!nextPayload?.feature?.latest || !nextPayload?.signal) {
        throw new Error("Invalid PC28 payload");
      }
      setPayload(nextPayload);
      setError(null);
    } catch (err: any) {
      setError(err.message || "PC28 API unavailable");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSignal();
    const timer = window.setInterval(loadSignal, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => setLocalNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const length = payload?.feature?.pressure_series?.length || 0;
    if (!length) {
      setViewEndIndex(null);
      return;
    }
    setViewEndIndex((current) => {
      if (current === null) {
        return null;
      }
      return Math.round(clamp(current, Math.min(length, zoomRange), length));
    });
  }, [payload?.feature?.pressure_series?.length, zoomRange]);

  const feature = payload?.feature;
  const latest = feature?.latest;
  const signal = payload?.signal;
  const prediction = payload?.prediction;
  const historyCount = feature?.history_count || payload?.fetched_count || 0;
  const currentMiss = feature?.pressure.current_non_straight_streak || 0;
  const avgGap = feature?.pressure.avg_non_straight_gap || 0;
  const maxGap = feature?.pressure.max_non_straight_gap || 0;
  const pressureScore = signal?.risk_score || 0;
  const theoreticalMiss = feature?.probability.straight_probability
    ? (1 / feature.probability.straight_probability).toFixed(1)
    : "--";
  const conditional = feature?.conditional_probability;
  const zone = pressureZone(pressureScore);
  const robotBias = prediction?.next_judgment?.label || "--";
  const pressureRatioAvg = feature?.pressure.pressure_ratio_to_avg || 0;
  const pressureRatioMax = feature?.pressure.pressure_ratio_to_max || 0;
  const pressureAlgorithm = [
    {
      label: "平均压力比",
      input: `当前遗漏 ${currentMiss} / 平均 ${avgGap.toFixed(2)}`,
      value: pressureRatioAvg.toFixed(2)
    },
    {
      label: "极值压力比",
      input: `当前遗漏 ${currentMiss} / 最大 ${maxGap || "--"}`,
      value: pressureRatioMax.toFixed(2)
    },
    {
      label: "条件出顺率",
      input: `样本 ${conditional?.straight_count || 0}/${conditional?.observed || 0}`,
      value: formatPercent(conditional?.straight_probability)
    }
  ];
  const baseStraightProbability = feature?.probability.straight_probability || 0;
  const conditionalStraightProbability = conditional?.straight_probability || 0;
  const aftershock = feature?.aftershock;
  const straightForecastBase =
    prediction?.targets?.straight?.backtest_next_hit_rate ??
    conditional?.straight_probability ??
    baseStraightProbability;
  const forecastProbabilities = [
    {
      label: "下一期",
      value: aftershock?.next_1_rate ?? straightForecastBase,
      helper: forecastHelper(aftershock?.next_1_straight_count, aftershock?.next_1_observed, "机器人顺单期率"),
      emphasis: false
    },
    {
      label: "下3期出顺",
      value: aftershock?.next_3_rate ?? compoundStraightProbability(straightForecastBase, 3),
      helper: forecastHelper(aftershock?.next_3_straight_count, aftershock?.next_3_observed, `至少一次 / 单期 ${formatPercent(straightForecastBase)}`),
      emphasis: true
    },
    {
      label: "下5期出顺",
      value: aftershock?.next_5_rate ?? compoundStraightProbability(straightForecastBase, 5),
      helper: forecastHelper(aftershock?.next_5_straight_count, aftershock?.next_5_observed, `至少一次 / 单期 ${formatPercent(straightForecastBase)}`),
      emphasis: true
    }
  ];
  const actualStraightRates = ACTUAL_RATE_WINDOWS.map((windowSize) => straightRateForWindow(feature?.pressure_series || [], windowSize));

  const pageData = useMemo(() => {
    const series = feature?.pressure_series || [];
    const maxWindow = Math.min(CHART_MAX_WINDOW, Math.max(CHART_MIN_WINDOW, series.length || CHART_MAX_WINDOW));
    const safeZoom = clamp(Math.round(zoomRange), CHART_MIN_WINDOW, maxWindow);
    const end = Math.round(clamp(viewEndIndex ?? series.length, safeZoom, series.length || safeZoom));
    const start = Math.max(0, end - safeZoom);
    const visible = series.slice(start, end);
    return {
      points: buildTrend(visible, {
        currentMiss,
        avgGap,
        maxGap,
        pressureScore,
        conditionalProbability: conditionalStraightProbability,
        baseProbability: baseStraightProbability
      }),
      start,
      end,
      visibleRange: safeZoom,
      totalCount: series.length
    };
  }, [
    feature?.pressure_series,
    zoomRange,
    viewEndIndex,
    currentMiss,
    avgGap,
    maxGap,
    pressureScore,
    conditionalStraightProbability,
    baseStraightProbability
  ]);

  const pathD = pageData.points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const chartSegments = pageData.points.slice(1).map((point, index) => ({
    from: pageData.points[index],
    to: point,
    isStraight: point.isStraight
  }));
  const missPoints = pageData.points.filter((point) => !point.isStraight);
  const hitPoints = pageData.points.filter((point) => point.isStraight);
  const latestTrendPoint = pageData.points[pageData.points.length - 1];
  const zoneCounts = pageData.points.reduce<Record<ZoneKey, number>>((counts, point) => {
    counts[point.zoneKey] += 1;
    return counts;
  }, { player: 0, watch: 0, risk: 0 });
  const chartSignal = calculateChartSignal(pageData.points, latestTrendPoint);
  const chartZone = chartSignal.zone;
  const isAtLatest = !feature?.pressure_series?.length || pageData.end >= feature.pressure_series.length;

  const setChartWindow = (targetRange: number, anchorRatio = 1) => {
    const seriesLength = feature?.pressure_series?.length || CHART_MAX_WINDOW;
    const maxWindow = Math.min(CHART_MAX_WINDOW, Math.max(CHART_MIN_WINDOW, seriesLength));
    const nextRange = Math.round(clamp(targetRange, CHART_MIN_WINDOW, maxWindow));
    if (seriesLength) {
      const currentRange = pageData.visibleRange;
      const end = pageData.end || seriesLength;
      const start = Math.max(0, end - currentRange);
      const anchor = clamp(anchorRatio, 0, 1);
      const anchorIndex = start + anchor * currentRange;
      const nextStart = Math.round(clamp(anchorIndex - anchor * nextRange, 0, Math.max(0, seriesLength - nextRange)));
      const nextEnd = Math.round(clamp(nextStart + nextRange, nextRange, seriesLength));
      setViewEndIndex(nextEnd >= seriesLength - 1 ? null : nextEnd);
    }
    setZoomRange(nextRange);
    setHovered(null);
  };

  const setZoomByRatio = (ratio: number, anchorRatio = 1) => {
    setChartWindow(pageData.visibleRange * ratio, anchorRatio);
  };

  const nearestChartPoint = (clientX: number, target: SVGSVGElement) => {
    if (!pageData.points.length) {
      return null;
    }
    const rect = target.getBoundingClientRect();
    const x = Math.min(1200, Math.max(0, ((clientX - rect.left) / rect.width) * 1200));
    return pageData.points.reduce((best, point) => (
      Math.abs(point.x - x) < Math.abs(best.x - x) ? point : best
    ), pageData.points[0]);
  };

  const chartAnchorRatio = (clientX: number, target: SVGSVGElement) => {
    const rect = target.getBoundingClientRect();
    if (!rect.width) {
      return 1;
    }
    return clamp((clientX - rect.left) / rect.width, 0, 1);
  };

  const panChartByPixels = (deltaX: number, target: SVGSVGElement) => {
    const seriesLength = feature?.pressure_series?.length || 0;
    if (!seriesLength || seriesLength <= pageData.visibleRange) {
      return;
    }
    const rect = target.getBoundingClientRect();
    const pointsDelta = Math.round((deltaX / Math.max(1, rect.width)) * pageData.visibleRange);
    const nextEnd = Math.round(clamp(chartDrag.current.startEndIndex - pointsDelta, pageData.visibleRange, seriesLength));
    setViewEndIndex(nextEnd >= seriesLength - 1 ? null : nextEnd);
  };

  const shiftChartWindow = (direction: -1 | 1) => {
    const seriesLength = feature?.pressure_series?.length || 0;
    if (!seriesLength || seriesLength <= pageData.visibleRange) {
      return;
    }
    const step = Math.max(10, Math.round(pageData.visibleRange * 0.45));
    const nextEnd = Math.round(clamp(pageData.end + direction * step, pageData.visibleRange, seriesLength));
    setViewEndIndex(nextEnd >= seriesLength - 1 ? null : nextEnd);
    setHovered(null);
  };

  const updateChartPinch = () => {
    const pointers: ChartPointer[] = Array.from(chartPointers.current.values());
    if (pointers.length < 2) {
      return;
    }

    const [first, second] = pointers;
    const distance = Math.hypot(first.clientX - second.clientX, first.clientY - second.clientY);
    if (!distance) {
      return;
    }

    if (!chartPinch.current.distance) {
      chartPinch.current.distance = distance;
      chartPinch.current.startRange = zoomRange;
      return;
    }

    chartPinch.current.active = true;
    const ratio = distance / chartPinch.current.distance;
    if (ratio > 0) {
      setChartWindow(chartPinch.current.startRange / ratio, 0.5);
    }
  };

  const handleChartPointerDown = (event: PointerEvent<SVGSVGElement>) => {
    event.currentTarget.setPointerCapture?.(event.pointerId);
    chartPointers.current.set(event.pointerId, {
      clientX: event.clientX,
      clientY: event.clientY
    });
    if (chartPointers.current.size >= 2) {
      updateChartPinch();
      return;
    }
    chartDrag.current = {
      active: true,
      startX: event.clientX,
      startEndIndex: pageData.end,
      moved: false
    };
    setHovered(nearestChartPoint(event.clientX, event.currentTarget));
  };

  const handleChartPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (chartPointers.current.has(event.pointerId)) {
      chartPointers.current.set(event.pointerId, {
        clientX: event.clientX,
        clientY: event.clientY
      });
    }

    if (chartPointers.current.size >= 2) {
      updateChartPinch();
      return;
    }

    if (chartDrag.current.active) {
      const deltaX = event.clientX - chartDrag.current.startX;
      if (Math.abs(deltaX) > 3) {
        chartDrag.current.moved = true;
        panChartByPixels(deltaX, event.currentTarget);
      }
    }

    setHovered(nearestChartPoint(event.clientX, event.currentTarget));
  };

  const clearChartPointer = (event: PointerEvent<SVGSVGElement>, hideCursor = false) => {
    chartPointers.current.delete(event.pointerId);
    if (chartPointers.current.size < 2) {
      chartPinch.current.distance = 0;
      chartPinch.current.startRange = zoomRange;
      chartPinch.current.active = false;
    }
    if (hideCursor) {
      setHovered(null);
    }
    chartDrag.current.active = false;
  };

  const handleChartWheel = (event: WheelEvent<SVGSVGElement>) => {
    event.preventDefault();
    if (Math.abs(event.deltaY) < 0.5) {
      return;
    }
    const ratio = Math.exp(event.deltaY * 0.0018);
    setZoomByRatio(ratio, chartAnchorRatio(event.clientX, event.currentTarget));
  };

  return (
    <div className="min-h-screen bg-[#050505] p-4 md:p-6 text-[#e5e5e5] font-mono selection:bg-white selection:text-black relative app-shell">
      <div className="absolute inset-0 pointer-events-none terminal-grid opacity-20 z-0"></div>

      <div className="max-w-[1600px] mx-auto space-y-4 relative z-10">
        <header className="mobile-header flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 bg-[#0a0a0a] border border-[#1a1a1a] px-5 py-4 rounded-sm">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 bg-white rotate-45 shrink-0"></div>
            <span className="text-lg font-bold tracking-tighter text-white font-sans uppercase">CASINO</span>
            <span className="text-[10px] bg-[#151515] text-[#888] border border-[#1a1a1a] px-2 py-0.5 rounded tracking-wide uppercase font-sans">
              PC28 REVERSE STRAIGHT
            </span>
          </div>

          <div className="mobile-live-strip flex w-full flex-col gap-3 bg-[#050505] p-2 rounded-sm border border-[#1a1a1a] sm:w-auto sm:flex-row sm:flex-wrap sm:items-center sm:gap-4">
            <div className="text-xs font-sans text-[#888] flex items-center gap-1.5 sm:border-r sm:border-[#1a1a1a] sm:pr-4">
              <span className={`inline-block w-2 h-2 rounded-full ${error ? "bg-[#E54B4B]" : "bg-green-500"} system-status-pulse`}></span>
              当前期号 ISSUE
              <span className="text-white font-mono font-bold ml-1">{formatIssue(latest?.issue)}</span>
            </div>

            <div className="text-[10px] font-sans text-[#888] flex flex-wrap items-center gap-x-2 gap-y-1 sm:border-r sm:border-[#1a1a1a] sm:pr-4">
              <span>开奖时间 DRAW</span>
              <span className="text-white font-mono font-bold">{formatDateTime(latest?.time || payload?.updated_at)}</span>
            </div>

            <div className="text-[10px] font-sans text-[#888] flex flex-wrap items-center gap-x-2 gap-y-1 sm:border-r sm:border-[#1a1a1a] sm:pr-4">
              <span>本机时间 LOCAL</span>
              <span className="text-white font-mono font-bold">{formatDateTime(localNow)}</span>
            </div>

            <div className="flex min-w-0 flex-wrap items-center gap-2 sm:gap-3">
              <span className="text-xs text-[#888] font-sans">开奖结果 RESULT:</span>
              <DrawDigits draw={latest} />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-2 px-2.5 py-1 border rounded-sm ${signalClass(signal?.next_signal)}`}>
              <ShieldAlert className="w-3.5 h-3.5" />
              <span className="text-[10px] font-extrabold tracking-wider font-sans uppercase">
                {signal?.signal_label || (loading ? "LOADING" : "OFFLINE")}
              </span>
            </div>
            <button
              onClick={loadSignal}
              className="bg-[#111] hover:bg-[#151515] border border-[#222] text-white p-2 rounded-sm transition cursor-pointer"
              title="Refresh PC28 signal"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </header>

        {error ? (
          <div className="bg-[#E54B4B]/10 border border-[#E54B4B]/30 text-[#ffb4ab] px-4 py-3 text-xs rounded-sm">
            PC28 API LINK ERROR: {error}
          </div>
        ) : null}

        <div className="metric-grid grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <MetricCard label="历史最大遗漏" value={maxGap} suffix="期" footer="历史最长未出顺区间" hot={currentMiss >= maxGap * 0.8 && maxGap > 0} />
          <MetricCard label="当前遗漏" value={currentMiss} suffix="期" footer={`平均 ${avgGap.toFixed(1)} / 最大 ${maxGap}`} hot={currentMiss >= avgGap && avgGap > 0} />
          <MetricCard label="当前连续未出" value={currentMiss} suffix="期" footer={eventText(latest)} hot={signal?.next_signal === "red"} />
          <MetricCard label="理论遗漏" value={theoreticalMiss} suffix="期" footer={`顺子理论概率 ${formatPercent(feature?.probability.straight_probability)}`} />
          <MetricCard label="条件出顺率" value={formatPercent(conditional?.straight_probability)} footer={`${conditional?.straight_count || 0}/${conditional?.observed || 0} 样本`} />
          <MetricCard label="历史平均间隔" value={avgGap.toFixed(1)} suffix="期" footer={`完整历史口径 ${historyCount} 期`} />
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
          <div className="xl:col-span-3 space-y-4">
            <div className="chart-card bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm overflow-hidden">
              <div className="bg-[#0f0f0f] border-b border-[#1a1a1a] px-5 py-4 flex flex-wrap justify-between items-center gap-3">
                <div className="flex flex-wrap items-center gap-2 min-w-0">
                  <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5 animate-pulse">
                    <Activity className="w-4 h-4 text-white" />
                    走势分析 TREND ANALYSIS
                  </span>
                  <span className="text-[10px] text-[#888] bg-[#050505] border border-[#1a1a1a] px-2 py-0.5 rounded-sm">
                    {formatIssue(latest?.issue)} / 窗口 {pageData.visibleRange} 期
                  </span>
                </div>

                <div className="chart-controls flex flex-wrap items-center gap-2">
                  <div className="hidden md:flex items-center gap-2 bg-[#050505] border border-[#1a1a1a] px-2 py-1 rounded-sm">
                    <span className="text-[10px] text-[#666] font-sans uppercase">ZOOM</span>
                    <input
                      type="range"
                      min={CHART_MIN_WINDOW}
                      max={Math.min(CHART_MAX_WINDOW, Math.max(CHART_MIN_WINDOW, pageData.totalCount || CHART_MAX_WINDOW))}
                      value={pageData.visibleRange}
                      onChange={(event) => setChartWindow(Number(event.target.value))}
                      className="w-28 accent-white cursor-pointer"
                    />
                    <span className="text-[10px] text-white font-mono w-10 text-right">{pageData.visibleRange}</span>
                  </div>

                  <div className="chart-presets bg-[#050505] p-1 rounded-sm border border-[#1a1a1a] flex flex-wrap">
                    {ZOOM_PRESETS.map((range) => (
                      <button
                        key={range}
                        onClick={() => setChartWindow(range)}
                        className={`text-[10px] px-2.5 py-1 rounded-sm transition-all font-sans uppercase font-bold cursor-pointer ${
                          Math.abs(pageData.visibleRange - range) <= 8 ? "bg-white text-black font-extrabold" : "text-[#888] hover:text-white"
                        }`}
                      >
                        {range}期
                      </button>
                    ))}
                  </div>

                  <div className="chart-pan-controls flex gap-1">
                    <button
                      onClick={() => shiftChartWindow(-1)}
                      disabled={pageData.start <= 0}
                      className="bg-[#0f0f0f] hover:bg-[#151515] disabled:opacity-35 border border-[#1a1a1a] p-1 rounded-sm text-[#888] hover:text-white transition cursor-pointer"
                      title="向历史平移"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => shiftChartWindow(1)}
                      disabled={isAtLatest}
                      className="bg-[#0f0f0f] hover:bg-[#151515] disabled:opacity-35 border border-[#1a1a1a] p-1 rounded-sm text-[#888] hover:text-white transition cursor-pointer"
                      title="向最新平移"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => {
                        setViewEndIndex(null);
                        setHovered(null);
                      }}
                      disabled={isAtLatest}
                      className="bg-[#0f0f0f] hover:bg-[#151515] disabled:opacity-35 border border-[#1a1a1a] px-2 rounded-sm text-[10px] text-[#888] hover:text-white transition cursor-pointer font-sans font-bold"
                    >
                      最新
                    </button>
                  </div>
                </div>
              </div>

              <div className="chart-subbar bg-[#050505]/80 px-5 py-2.5 border-b border-[#1a1a1a] flex flex-col sm:flex-row sm:flex-wrap sm:justify-between sm:items-center gap-2 text-[10px] text-[#888] font-sans">
                <div className="flex flex-wrap gap-x-4 gap-y-1 min-w-0">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 inline-block"></span>
                    未出 (-1) : 客户赢走势向下
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#E54B4B] inline-block"></span>
                    出顺 (+17) : 碰顺亏损上拉
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-2 self-start sm:self-auto">
                  <span className="text-[#666]">
                    {pageData.totalCount ? `${pageData.start + 1}-${pageData.end} / ${pageData.totalCount}` : "0 / 0"}
                  </span>
                  {pageData.points.length ? (
                    <span className={`border px-1.5 py-0.5 rounded-sm font-bold ${zoneBgClass(chartZone.key)} ${zoneTextClass(chartZone.key)}`}>
                      图表信号 {chartZone.label} / {chartSignal.score}
                    </span>
                  ) : null}
                  <span className="text-[10px] text-[#666]">
                    滚轮缩放 / 拖动平移 / 双指缩放
                  </span>
                </div>
              </div>

              <div className="chart-stage-wrap p-4 bg-[#050505] relative">
                <svg
                  className="w-full h-[468px] overflow-visible cursor-crosshair z-20 relative"
                  onPointerDown={handleChartPointerDown}
                  onPointerEnter={(event) => setHovered(nearestChartPoint(event.clientX, event.currentTarget))}
                  onPointerMove={handleChartPointerMove}
                  onPointerUp={(event) => clearChartPointer(event)}
                  onPointerLeave={(event) => clearChartPointer(event, true)}
                  onPointerCancel={(event) => clearChartPointer(event, true)}
                  onWheel={handleChartWheel}
                  viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
                  preserveAspectRatio="none"
                  style={{ touchAction: "none" }}
                >
                  <rect
                    x="0"
                    y="0"
                    width="1200"
                    height={CHART_HEIGHT}
                    fill={zoneFill(chartZone.key)}
                    opacity="0.08"
                  />
                  <line
                    x1="0"
                    y1="38"
                    x2="1200"
                    y2="38"
                    stroke={zoneFill(chartZone.key)}
                    strokeOpacity="0.32"
                    strokeDasharray="7 9"
                  />
                  <text
                    x="24"
                    y="27"
                    textAnchor="start"
                    fill={zoneFill(chartZone.key)}
                    opacity="0.74"
                    fontSize="15"
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight="700"
                  >
                    {chartZone.label} SIGNAL / 最新{chartSignal.latestRisk} 加权{chartSignal.weightedRisk.toFixed(1)}
                  </text>
                  {[40, 100, 160, 220].map((y) => (
                    <line key={y} x1="0" y1={y} x2="1200" y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="4 4" />
                  ))}
                  <defs>
                    <linearGradient id="pc28ChartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ffffff" stopOpacity="0.08" />
                      <stop offset="100%" stopColor="#ffffff" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>
                  {pathD ? (
                    <>
                      <path d={`${pathD} L ${CHART_WIDTH} ${CHART_HEIGHT} L 0 ${CHART_HEIGHT} Z`} fill="url(#pc28ChartGradient)" />
                    </>
                  ) : null}
                  {chartSegments.map((segment) => (
                    <g key={`${segment.from.issue}-${segment.to.issue}`}>
                      {segment.isStraight ? (
                        <>
                          <line
                            x1={segment.from.x}
                            y1={segment.from.y}
                            x2={segment.to.x}
                            y2={segment.from.y}
                            stroke={CHART_MISS_COLOR}
                            strokeWidth="1.9"
                            strokeLinecap="round"
                            opacity="0.82"
                          />
                          <line
                            x1={segment.to.x}
                            y1={segment.from.y}
                            x2={segment.to.x}
                            y2={segment.to.y}
                            stroke={CHART_HIT_COLOR}
                            strokeWidth="2.8"
                            strokeLinecap="round"
                            opacity="0.98"
                          />
                        </>
                      ) : (
                        <line
                          x1={segment.from.x}
                          y1={segment.from.y}
                          x2={segment.to.x}
                          y2={segment.to.y}
                          stroke={CHART_MISS_COLOR}
                          strokeWidth="1.9"
                          strokeLinecap="round"
                          opacity="0.9"
                        />
                      )}
                    </g>
                  ))}
                  {missPoints.map((point) => (
                    <circle
                      key={point.issue}
                      cx={point.x}
                      cy={point.y}
                      r="2.3"
                      fill={CHART_MISS_COLOR}
                      stroke="#051316"
                      strokeWidth="0.8"
                      opacity="0.9"
                    />
                  ))}
                  {hitPoints.map((point) => (
                    <g key={point.issue}>
                      <circle
                        cx={point.x}
                        cy={point.y}
                        r="3.5"
                        fill={CHART_HIT_COLOR}
                        stroke="#FFFFFF"
                        strokeWidth="1"
                      />
                    </g>
                  ))}
                  {hovered ? (
                    <>
                      <line x1={hovered.x} y1="0" x2={hovered.x} y2={CHART_HEIGHT} stroke="#ffffff" strokeWidth="1" strokeDasharray="3 3" />
                      <line x1="0" y1={hovered.y} x2="1200" y2={hovered.y} stroke="#ffffff" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.55" />
                      <circle cx={hovered.x} cy={hovered.y} r="6" fill={hovered.isStraight ? CHART_HIT_COLOR : CHART_MISS_COLOR} opacity="0.88" />
                    </>
                  ) : null}
                </svg>

                {hovered ? (
                  <div
                    className="absolute bg-[#0a0a0a] text-xs p-3 rounded-sm border border-[#1a1a1a] shadow-2xl text-white space-y-1.5 z-30 pointer-events-none"
                    style={{ left: `${Math.min((hovered.x / 1200) * 100, 72)}%`, top: `${Math.min(hovered.y + 18, 210)}px` }}
                  >
                    <div className="font-bold border-b border-[#222] pb-1.5 flex justify-between gap-6">
                      <span className="text-[#888]">ISSUE:</span>
                      <span>{formatIssue(hovered.issue)}</span>
                    </div>
                    <div className="flex justify-between text-[10px] gap-5">
                      <span className="text-[#888]">STATUS:</span>
                      <span className={hovered.isStraight ? "text-[#ff9e99] font-bold" : "text-cyan-300 font-bold"}>
                        {hovered.isStraight ? "出顺 (+17)" : "未出 (-1)"}
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px] gap-5">
                      <span className="text-[#888]">MISS:</span>
                      <span className="text-white">{hovered.value}期</span>
                    </div>
                    <div className="flex justify-between text-[10px] gap-5">
                      <span className="text-[#888]">ZONE:</span>
                      <span className={`${zoneTextClass(hovered.zoneKey)} font-bold`}>{hovered.zoneLabel} / {hovered.riskScore}</span>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

          </div>

          <div className="space-y-4">
            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4 space-y-4">
              <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5">
                  <GaugeIcon className="w-4 h-4 text-white" />
                  压力算法 PRESSURE MODEL
                </span>
                <span className={`text-[9.5px] font-extrabold px-1.5 py-0.5 rounded-sm font-sans tracking-wide border ${zone.className}`}>
                  {zone.label}
                </span>
              </div>

              <PressureGauge
                score={pressureScore}
                zone={zone}
                avgRatio={pressureRatioAvg}
                maxRatio={pressureRatioMax}
                conditionalRate={conditional?.straight_probability || 0}
              />

              <div className="space-y-2">
                {pressureAlgorithm.map((item) => (
                  <div key={item.label} className="bg-[#050505] border border-[#1a1a1a] rounded-sm px-3 py-2">
                    <div className="flex justify-between gap-3 text-[10px] font-sans uppercase tracking-wider">
                      <span className="text-[#888]">{item.label}</span>
                      <span className="text-white font-mono font-bold">{item.value}</span>
                    </div>
                    <div className="mt-1 text-[10px] text-[#666] font-mono">{item.input}</div>
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-3 gap-1 text-[9px] font-sans uppercase tracking-wider">
                {[
                  { key: "player", label: "玩家优势", min: 0, max: 49 },
                  { key: "watch", label: "临界观察", min: 50, max: 74 },
                  { key: "risk", label: "风险", min: 75, max: 100 }
                ].map((item) => (
                  <div
                    key={item.key}
                    className={`border rounded-sm px-2 py-1.5 ${zone.key === item.key ? `${zoneBgClass(item.key)} ${zoneTextClass(item.key)}` : "border-[#1a1a1a] text-[#666]"}`}
                  >
                    <div className="font-bold">{item.label}</div>
                    <div className="text-[8px] opacity-80">{item.min}-{item.max}</div>
                  </div>
                ))}
              </div>

              <div className="text-[10px] text-[#777] font-mono border-t border-[#1a1a1a] pt-3 leading-relaxed">
                压力分只读取当前遗漏、历史平均间隔、历史最大遗漏、条件出顺率；走势图逐点背景沿用同一压力口径。
              </div>
            </div>

            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4 space-y-3">
              <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5">
                  <Bot className="w-4 h-4 text-white" />
                  机器人信号 BOT SIGNAL
                </span>
                <span className={`text-[9.5px] font-extrabold px-1.5 py-0.5 rounded-sm border ${signalClass(signal?.next_signal)}`}>
                  {signal?.signal_label || "--"}
                </span>
              </div>

              <div className="robot-signal-shell">
                <div className={`robot-data-strip robot-data-strip-strong ${robotBiasBgClass(robotBias)}`}>
                  <span>RAW JUDGMENT</span>
                  <strong className={robotBiasClass(robotBias)}>{robotBias}</strong>
                  <em>score {prediction?.next_judgment?.score ?? "--"}</em>
                </div>

                <div className="grid grid-cols-2 gap-2">
                  <div className="robot-stat-cell col-span-2 sm:col-span-1">
                    <span>回测命中</span>
                    <strong>{formatPercent(prediction?.next_judgment?.hit_rate)}</strong>
                    <em>{prediction?.next_judgment?.hit_count || 0}/{prediction?.next_judgment?.sample_size || 0} samples</em>
                  </div>
                  <div className={`robot-stat-cell col-span-2 sm:col-span-1 ${zoneBgClass(zone.key)}`}>
                    <span>风险信号</span>
                    <strong className={zoneTextClass(zone.key)}>{signal?.signal_label || "--"}</strong>
                    <em>{pressureScore}/100 pressure</em>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  <div className="robot-target-card robot-target-straight">
                    <div className="flex items-center justify-between gap-2">
                      <span className="robot-target-label text-[#ff9e99]">
                        <Target className="w-3.5 h-3.5" />
                        机器人顺
                      </span>
                      <strong>{formatPercent(prediction?.targets?.straight?.backtest_next_hit_rate)}</strong>
                    </div>
                    <div className="robot-meter">
                      <span style={{ width: `${Math.min(100, Number(prediction?.targets?.straight?.backtest_next_hit_rate || 0) * 100)}%` }} />
                    </div>
                    <div className="robot-target-meta">
                      score {prediction?.targets?.straight?.score ?? "--"} / sample {prediction?.targets?.straight?.sample_size ?? "--"}
                    </div>
                    <p>{prediction?.targets?.straight?.reason || "--"}</p>
                  </div>

                  <div className="robot-target-card robot-target-non-straight">
                    <div className="flex items-center justify-between gap-2">
                      <span className="robot-target-label text-amber-300">
                        <Target className="w-3.5 h-3.5" />
                        机器人非顺
                      </span>
                      <strong>{formatPercent(prediction?.targets?.non_straight?.backtest_next_hit_rate)}</strong>
                    </div>
                    <div className="robot-meter">
                      <span style={{ width: `${Math.min(100, Number(prediction?.targets?.non_straight?.backtest_next_hit_rate || 0) * 100)}%` }} />
                    </div>
                    <div className="robot-target-meta">
                      score {prediction?.targets?.non_straight?.score ?? "--"} / sample {prediction?.targets?.non_straight?.sample_size ?? "--"}
                    </div>
                    <p>{prediction?.targets?.non_straight?.reason || "--"}</p>
                  </div>
                </div>

                <div className="text-[10px] text-[#777] font-mono leading-relaxed border-t border-[#1a1a1a] pt-3">
                  <span className={zoneTextClass(zone.key)}>{signal?.signal_label || "--"}</span>
                  <span className="text-[#555]"> / </span>
                  {signal?.reason || "--"}
                </div>
              </div>
            </div>

            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4 space-y-3">
              <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-white" />
                  策略参考 STRATEGY REF
                </span>
                <span className={`text-[9px] font-bold ${zoneTextClass(zone.key)}`}>{zone.label}</span>
              </div>
              <div className="text-[11px] leading-relaxed text-zinc-300 font-mono space-y-3">
                <div className="bg-[#050505] border border-[#1a1a1a] rounded-sm p-3">
                  短线信号：{signal?.signal_label || "--"} / 压力 {pressureScore}/100。{signal?.reason || "--"}
                </div>
                <div className="strategy-grid strategy-probability-grid">
                  {forecastProbabilities.map((item) => (
                    <div key={item.label} className={`strategy-metric-card strategy-probability-card ${item.emphasis ? "strategy-probability-hot" : ""}`}>
                      <div className="strategy-probability-line">
                        <span>{item.label}</span>
                        <strong>{formatPercent(item.value)}</strong>
                      </div>
                      <em>{item.helper}</em>
                    </div>
                  ))}
                </div>
                <div className="strategy-grid">
                  {actualStraightRates.map((item) => (
                    <div key={item.windowSize} className="strategy-metric-card">
                      <span>{item.windowSize}期实际出顺</span>
                      <strong>{formatPercent(item.rate)}</strong>
                      <em>{item.straightCount}/{item.sampleSize} draws</em>
                    </div>
                  ))}
                </div>
                <div className="text-zinc-500">
                  走势图背景按当前窗口聚合计算，只显示主信号色；窗口内分布：玩家优势 {zoneCounts.player} / 临界观察 {zoneCounts.watch} / 风险 {zoneCounts.risk}。
                </div>
              </div>
            </div>

            <div className="robot-raw-panel bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4">
              <div className="robot-raw-block">
                <div className="text-white font-sans font-bold uppercase tracking-wider mb-1">原始机器人数据 RAW BOT DATA</div>
                <div>偏向 {robotBias} / 原始分数 {prediction?.next_judgment?.score ?? "--"} / 命中 {prediction?.next_judgment?.hit_count || 0}/{prediction?.next_judgment?.sample_size || 0} / 回测 {formatPercent(prediction?.next_judgment?.hit_rate)}</div>
                <div>顺 score {prediction?.targets?.straight?.score ?? "--"} sample {prediction?.targets?.straight?.sample_size ?? "--"} hit {formatPercent(prediction?.targets?.straight?.backtest_next_hit_rate)}</div>
                <div>非顺 score {prediction?.targets?.non_straight?.score ?? "--"} sample {prediction?.targets?.non_straight?.sample_size ?? "--"} hit {formatPercent(prediction?.targets?.non_straight?.backtest_next_hit_rate)}</div>
              </div>
            </div>

            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4 space-y-3">
              <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5 border-b border-[#1a1a1a] pb-2.5">
                <Database className="w-4 h-4 text-white" />
                统计分析 STATS ANALYSIS
              </span>
              {[
                ["近50期出顺率", feature?.recent_density.recent_50_rate, feature?.recent_density.recent_50_straight_count],
                ["近100期出顺率", feature?.recent_density.recent_100_rate, feature?.recent_density.recent_100_straight_count],
                ["条件出顺率", conditional?.straight_probability, conditional?.straight_count]
              ].map(([label, rate, count]) => (
                <div key={String(label)} className="space-y-1">
                  <div className="flex justify-between items-center text-[11px]">
                    <span className="text-zinc-300 font-sans">{label}</span>
                    <span className="text-white font-bold">{formatPercent(Number(rate || 0))}</span>
                  </div>
                  <div className="w-full bg-[#050505] h-2 rounded-sm border border-[#1a1a1a] overflow-hidden p-0.5">
                    <div className="bg-white h-full rounded-sm transition-all duration-300" style={{ width: `${Math.min(100, Number(rate || 0) * 100)}%` }} />
                  </div>
                  <div className="text-[9px] text-[#666]">出顺 {Number(count || 0)} 次</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <HistoryTable results={feature?.recent_results || []} />
      </div>
    </div>
  );
}
