import { useEffect, useMemo, useState, MouseEvent } from "react";
import {
  Activity,
  AlertCircle,
  ChevronLeft,
  ChevronRight,
  Database,
  RefreshCw,
  ShieldAlert,
  Sparkles,
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
      ev_straight: number;
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
};

const API_URL = "/api/pc28-signal";
const REFRESH_MS = 10000;
const CHART_HIT_LIFT = 17;
const CHART_MISS_STEP = 1;

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

function formatIssue(issue?: string) {
  return issue ? `#${issue}` : "--";
}

function eventText(draw?: Draw) {
  return draw?.is_straight ? "出顺：客户输" : "未出：客户赢";
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

function buildTrend(series: PressurePoint[], limit: number): TrendPoint[] {
  const visible = series.slice(-limit);
  let trendValue = 0;
  const raw = visible.map((point) => {
    if (point.is_straight) {
      trendValue += CHART_HIT_LIFT;
    } else {
      trendValue -= CHART_MISS_STEP;
    }
    return {
      issue: point.issue,
      value: point.value,
      isStraight: point.is_straight,
      trendValue,
      x: 0,
      y: 0
    };
  });

  if (!raw.length) {
    return [];
  }

  const width = 1200;
  const height = 280;
  const values = raw.map((point) => point.trendValue);
  const max = Math.max(...values, 8);
  const min = Math.min(...values, -8);
  const range = Math.max(1, max - min);

  return raw.map((point, index) => ({
    ...point,
    x: raw.length > 1 ? (index / (raw.length - 1)) * width : width / 2,
    y: 20 + ((max - point.trendValue) / range) * (height - 40)
  }));
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

export default function App() {
  const [payload, setPayload] = useState<Pc28Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [zoomRange, setZoomRange] = useState(1000);
  const [chartPage, setChartPage] = useState(0);
  const [hovered, setHovered] = useState<TrendPoint | null>(null);

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
    setChartPage(0);
    setHovered(null);
  }, [zoomRange]);

  const feature = payload?.feature;
  const latest = feature?.latest;
  const signal = payload?.signal;
  const prediction = payload?.prediction;
  const historyCount = feature?.history_count || payload?.fetched_count || 0;

  const pageData = useMemo(() => {
    const series = feature?.pressure_series || [];
    const totalPages = Math.max(1, Math.ceil(series.length / zoomRange));
    const safePage = Math.min(chartPage, totalPages - 1);
    const end = series.length - safePage * zoomRange;
    const start = Math.max(0, end - zoomRange);
    return {
      points: buildTrend(series.slice(start, end), zoomRange),
      totalPages,
      page: safePage
    };
  }, [feature?.pressure_series, zoomRange, chartPage]);

  const pathD = pageData.points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
  const hitPoints = pageData.points.filter((point) => point.isStraight);
  const currentMiss = feature?.pressure.current_non_straight_streak || 0;
  const avgGap = feature?.pressure.avg_non_straight_gap || 0;
  const maxGap = feature?.pressure.max_non_straight_gap || 0;
  const pressureScore = signal?.risk_score || 0;
  const theoreticalMiss = feature?.probability.straight_probability
    ? (1 / feature.probability.straight_probability).toFixed(1)
    : "--";
  const conditional = feature?.conditional_probability;

  const handleChartMove = (event: MouseEvent<SVGSVGElement>) => {
    if (!pageData.points.length) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 1200;
    const closest = pageData.points.reduce((best, point) => (
      Math.abs(point.x - x) < Math.abs(best.x - x) ? point : best
    ), pageData.points[0]);
    setHovered(closest);
  };

  return (
    <div className="min-h-screen bg-[#050505] p-4 md:p-6 text-[#e5e5e5] font-mono selection:bg-white selection:text-black relative">
      <div className="absolute inset-0 pointer-events-none terminal-grid opacity-20 z-0"></div>

      <div className="max-w-[1600px] mx-auto space-y-4 relative z-10">
        <header className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 bg-[#0a0a0a] border border-[#1a1a1a] px-5 py-4 rounded-sm">
          <div className="flex items-center gap-3">
            <div className="w-5 h-5 bg-white rotate-45 shrink-0"></div>
            <span className="text-lg font-bold tracking-tighter text-white font-sans uppercase">CASINO</span>
            <span className="text-[10px] bg-[#151515] text-[#888] border border-[#1a1a1a] px-2 py-0.5 rounded tracking-wide uppercase font-sans">
              PC28 REVERSE STRAIGHT
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-4 bg-[#050505] p-2 rounded-sm border border-[#1a1a1a]">
            <div className="text-xs font-sans text-[#888] flex items-center gap-1.5 border-r border-[#1a1a1a] pr-4">
              <span className={`inline-block w-2 h-2 rounded-full ${error ? "bg-[#E54B4B]" : "bg-green-500"} system-status-pulse`}></span>
              当前期号 ISSUE
              <span className="text-white font-mono font-bold ml-1">{formatIssue(latest?.issue)}</span>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs text-[#888] font-sans">开奖结果 RESULT:</span>
              <div className="flex gap-1.5">
                {(latest?.digits || [0, 0, 0]).map((digit, index) => (
                  <span
                    key={index}
                    className={`w-8 h-8 flex items-center justify-center rounded-full text-xs font-bold border ${
                      latest?.is_straight
                        ? "bg-[#E54B4B]/20 border-[#E54B4B] text-[#ff9e99]"
                        : "bg-[#111] border-[#222] text-white"
                    } shadow-inner`}
                  >
                    {String(digit).padStart(2, "0")}
                  </span>
                ))}
              </div>
              <span className="text-xs text-[#888] font-mono">SUM {latest?.sum ?? "--"}</span>
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

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <MetricCard label="当前连续未出顺子" value={currentMiss} suffix="期" footer={eventText(latest)} hot={signal?.next_signal === "red"} />
          <MetricCard label="历史平均间隔" value={avgGap.toFixed(1)} suffix="期" footer={`完整历史口径 ${historyCount} 期`} />
          <MetricCard label="历史最大间隔" value={maxGap} suffix="期" footer="历史最长未出顺区间" hot={currentMiss >= maxGap * 0.8 && maxGap > 0} />
          <MetricCard label="条件出顺率" value={formatPercent(conditional?.straight_probability)} footer={`${conditional?.straight_count || 0}/${conditional?.observed || 0} 样本`} />
          <MetricCard label="压力指数" value={(pressureScore / 100).toFixed(3)} footer={signal?.reason || "等待实时信号"} hot={pressureScore >= 75} />
          <MetricCard label="EV" value={feature?.probability.ev_straight?.toFixed(2) || "--"} footer={`赔率 ${feature?.probability.straight_odds || "--"} / 公平 ${feature?.probability.fair_straight_odds || "--"}`} />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 bg-[#0a0a0a] border border-[#1a1a1a] p-5 rounded-sm">
          <div className="flex items-center justify-between border-b md:border-b-0 md:border-r border-[#1a1a1a] pb-4 md:pb-0 pr-0 md:pr-6">
            <div className="space-y-1">
              <span className="text-xs text-[#888] tracking-wider uppercase font-sans">理论遗漏 THEORETICAL MISS</span>
              <div className="text-sm font-bold text-white font-mono">
                顺子理论概率 {formatPercent(feature?.probability.straight_probability)}
              </div>
            </div>
            <span className="text-3xl font-extrabold text-white tracking-tighter">
              {theoreticalMiss}
            </span>
          </div>

          <div className="flex items-center justify-between pl-0 md:pl-6 pt-4 md:pt-0">
            <div className="space-y-1">
              <span className="text-xs text-[#888] tracking-wider uppercase font-sans">下期信号 NEXT SIGNAL</span>
              <div className="text-xs text-[#666] font-sans">
                机器人样本 {prediction?.next_judgment?.hit_count || 0} / {prediction?.next_judgment?.sample_size || 0}
              </div>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-3xl font-extrabold tracking-tighter text-white">
                {signal?.signal_label || "--"}
              </span>
              <span className="text-xs text-[#888]">机器人偏向 {prediction?.next_judgment?.label || "--"} / {formatPercent(prediction?.next_judgment?.hit_rate, 1)}</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-4 gap-4">
          <div className="xl:col-span-3 space-y-4">
            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm overflow-hidden">
              <div className="bg-[#0f0f0f] border-b border-[#1a1a1a] px-5 py-4 flex flex-wrap justify-between items-center gap-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5 animate-pulse">
                    <Activity className="w-4 h-4 text-white" />
                    走势分析 TREND ANALYSIS
                  </span>
                  <span className="text-[10px] text-[#888] bg-[#050505] border border-[#1a1a1a] px-2 py-0.5 rounded-sm">
                    {formatIssue(latest?.issue)} / 最近 {zoomRange} 期
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <div className="bg-[#050505] p-1 rounded-sm border border-[#1a1a1a] flex">
                    {[400, 800, 1000].map((range) => (
                      <button
                        key={range}
                        onClick={() => setZoomRange(range)}
                        className={`text-[10px] px-2.5 py-1 rounded-sm transition-all font-sans uppercase font-bold cursor-pointer ${
                          zoomRange === range ? "bg-white text-black font-extrabold" : "text-[#888] hover:text-white"
                        }`}
                      >
                        {range}期
                      </button>
                    ))}
                  </div>

                  <div className="flex gap-1">
                    <button
                      onClick={() => setChartPage((page) => Math.min(page + 1, pageData.totalPages - 1))}
                      disabled={pageData.page >= pageData.totalPages - 1}
                      className="bg-[#0f0f0f] hover:bg-[#151515] disabled:opacity-35 border border-[#1a1a1a] p-1 rounded-sm text-[#888] hover:text-white transition cursor-pointer"
                    >
                      <ChevronLeft className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => setChartPage((page) => Math.max(0, page - 1))}
                      disabled={pageData.page <= 0}
                      className="bg-[#0f0f0f] hover:bg-[#151515] disabled:opacity-35 border border-[#1a1a1a] p-1 rounded-sm text-[#888] hover:text-white transition cursor-pointer"
                    >
                      <ChevronRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>

              <div className="bg-[#050505]/80 px-5 py-2.5 border-b border-[#1a1a1a] flex flex-wrap justify-between items-center gap-2 text-[10px] text-[#888] font-sans">
                <div className="flex gap-4">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block"></span>
                    未出 (-1) : 客户赢走势向下
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-full bg-[#E54B4B] inline-block"></span>
                    出顺 (+17) : 碰顺亏损上拉
                  </span>
                </div>
                <div className="text-xs font-mono font-semibold text-[#888]">
                  第 <span className="text-white">{pageData.page + 1}</span> / {pageData.totalPages} 页
                </div>
              </div>

              <div className="p-4 bg-[#050505] relative">
                <svg
                  className="w-full h-[280px] overflow-visible cursor-crosshair z-20 relative"
                  onMouseMove={handleChartMove}
                  onMouseLeave={() => setHovered(null)}
                  viewBox="0 0 1200 280"
                  preserveAspectRatio="none"
                >
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
                      <path d={`${pathD} L 1200 280 L 0 280 Z`} fill="url(#pc28ChartGradient)" />
                      <path d={pathD} fill="none" stroke="#ffffff" strokeWidth="1.8" strokeLinecap="round" />
                    </>
                  ) : null}
                  {hitPoints.map((point) => (
                    <circle
                      key={point.issue}
                      cx={point.x}
                      cy={point.y}
                      r="3.5"
                      fill="#E54B4B"
                      stroke="#FFFFFF"
                      strokeWidth="1"
                    />
                  ))}
                  {hovered ? (
                    <>
                      <line x1={hovered.x} y1="0" x2={hovered.x} y2="280" stroke="#ffffff" strokeWidth="1" strokeDasharray="3 3" />
                      <line x1="0" y1={hovered.y} x2="1200" y2={hovered.y} stroke="#ffffff" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.55" />
                      <circle cx={hovered.x} cy={hovered.y} r="6" fill={hovered.isStraight ? "#E54B4B" : "#ffffff"} opacity="0.8" />
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
                      <span className={hovered.isStraight ? "text-[#ff9e99] font-bold" : "text-[#888]"}>
                        {hovered.isStraight ? "出顺 (+17)" : "未出 (-1)"}
                      </span>
                    </div>
                    <div className="flex justify-between text-[10px] gap-5">
                      <span className="text-[#888]">MISS:</span>
                      <span className="text-white">{hovered.value}期</span>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>

            <div className="bg-[#050505] border border-[#1a1a1a] rounded-sm p-4 h-[210px] select-text relative">
              <div className="overflow-y-auto space-y-1 text-[11px] font-mono text-zinc-300 opacity-90 pr-1 select-text h-full">
                <div className="text-zinc-500 text-[10px] border-b border-[#1a1a1a] pb-1.5 mb-2.5 font-sans flex justify-between">
                  <span>PC28 LIVE DATA BRIDGE</span>
                  <span>{payload?.refresh_status || "WAITING"}</span>
                </div>
                <div><span className="text-zinc-600 mr-1.5">[00]</span>读取接口 /api/pc28-signal</div>
                <div><span className="text-zinc-600 mr-1.5">[01]</span>当前期号 {formatIssue(latest?.issue)} / {latest?.raw || "--"}</div>
                <div><span className="text-zinc-600 mr-1.5">[02]</span>本期事件 {signal?.event_label || "--"}</div>
                <div><span className="text-zinc-600 mr-1.5">[03]</span>压力指数 {pressureScore}/100 / 下期信号 {signal?.signal_label || "--"}</div>
                <div><span className="text-zinc-600 mr-1.5">[04]</span>条件出顺率 {formatPercent(conditional?.straight_probability)} / 样本 {conditional?.observed || 0}</div>
                <div><span className="text-zinc-600 mr-1.5">[05]</span>更新时间 {formatTime(payload?.updated_at)} / 本机 {formatTime(new Date().toISOString())}</div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm flex flex-col h-[280px]">
              <div className="bg-[#0f0f0f] border-b border-[#1a1a1a] px-4 py-2.5 flex justify-between items-center shrink-0">
                <span className="text-xs font-bold uppercase tracking-wider text-white font-sans">历史开奖 HISTORY</span>
                <span className="text-[10px] text-[#888]">最近 20 期</span>
              </div>
              <div className="overflow-y-auto flex-1 text-xs select-none">
                <table className="w-full text-left">
                  <thead className="bg-[#050505] text-[#888] font-sans text-[10px] sticky top-0 border-b border-[#1a1a1a] z-10">
                    <tr>
                      <th className="p-2 py-2 pl-4">期号</th>
                      <th className="p-2 py-2">开奖号</th>
                      <th className="p-2 py-2 pr-4 text-right">状态</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#1a1a1a]">
                    {(feature?.recent_results || []).slice(0, 20).map((item, index) => (
                      <tr key={item.issue} className={`${index % 2 === 0 ? "bg-[#0a0a0a]" : "bg-[#070707]"} hover:bg-[#151515]`}>
                        <td className="p-2 pl-4 font-mono text-white text-[11px]">{formatIssue(item.issue)}</td>
                        <td className="p-2">
                          <div className="flex gap-1">
                            {item.digits.map((digit, i) => (
                              <span key={i} className="w-5 h-5 flex items-center justify-center rounded-full text-[10px] font-bold bg-[#0f0f0f] border border-zinc-900 text-[#ddd]">
                                {digit}
                              </span>
                            ))}
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
            </div>

            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4 space-y-4">
              <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-white font-sans">风险评估 RISK ASSESSMENT</span>
                <span className={`text-[9.5px] font-extrabold px-1.5 py-0.5 rounded-sm font-sans tracking-wide border ${signalClass(signal?.next_signal)}`}>
                  {signal?.signal_label || "--"}
                </span>
              </div>
              <div className="flex flex-col items-center py-2">
                <div className="relative w-40 h-28 overflow-hidden">
                  <svg className="w-full h-full" viewBox="0 0 100 70">
                    <path d="M 15 55 A 40 40 0 0 1 85 55" fill="none" stroke="#1a1a1a" strokeWidth="8" strokeLinecap="round" />
                    <path
                      d="M 15 55 A 40 40 0 0 1 85 55"
                      fill="none"
                      stroke={signal?.next_signal === "red" ? "#E54B4B" : signal?.next_signal === "yellow" ? "#f59e0b" : "#ffffff"}
                      strokeWidth="8"
                      strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 - (pressureScore / 100) * 110}
                      className="transition-all duration-300"
                    />
                  </svg>
                  <div className="absolute inset-x-0 bottom-1 flex flex-col items-center">
                    <span className="text-3xl font-extrabold font-mono tracking-tighter text-white">{pressureScore}%</span>
                    <span className="text-[10px] text-[#888] font-sans font-medium uppercase tracking-wider">压力指数</span>
                  </div>
                </div>
                <div className="w-full grid grid-cols-2 gap-3 border-t border-[#1a1a1a] pt-3 mt-2 text-[11px] font-mono">
                  <div><span className="text-[#888] block text-[10px] font-sans uppercase">当前遗漏</span><span className="text-white font-bold">{currentMiss}</span></div>
                  <div className="text-right"><span className="text-[#888] block text-[10px] font-sans uppercase">平均间隔</span><span className="text-white font-bold">{avgGap.toFixed(2)}</span></div>
                  <div><span className="text-[#888] block text-[10px] font-sans uppercase">最大间隔</span><span className="text-white font-bold">{maxGap}</span></div>
                  <div className="text-right"><span className="text-[#888] block text-[10px] font-sans uppercase">下期信号</span><span className="text-white font-bold">{signal?.signal_label || "--"}</span></div>
                </div>
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

            <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-sm p-4 space-y-3">
              <div className="flex justify-between items-center border-b border-[#1a1a1a] pb-2.5">
                <span className="text-xs font-bold uppercase tracking-wider text-white font-sans flex items-center gap-1.5">
                  <Sparkles className="w-4 h-4 text-white" />
                  策略参考 STRATEGY REF
                </span>
                <Zap className="w-4 h-4 text-white" />
              </div>
              <div className="bg-[#050505] p-3 rounded-sm border border-[#1a1a1a] text-[11px] leading-relaxed text-zinc-300 font-mono min-h-[120px] space-y-2">
                <p><span className="text-[#888]">下期信号：</span>{signal?.signal_label || "--"} / {signal?.reason || "--"}</p>
                <p><span className="text-[#888]">机器人顺：</span>{formatPercent(prediction?.targets?.straight?.backtest_next_hit_rate)} / 分数 {prediction?.targets?.straight?.score ?? "--"}</p>
                <p><span className="text-[#888]">机器人非顺：</span>{formatPercent(prediction?.targets?.non_straight?.backtest_next_hit_rate)} / 分数 {prediction?.targets?.non_straight?.score ?? "--"}</p>
                <p className="text-zinc-500">只做风险参考，不输出金额，不执行任何自动操作。</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
