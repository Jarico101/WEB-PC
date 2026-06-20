import express from "express";
import path from "path";
import dotenv from "dotenv";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";

// Load environment variables from .env file
dotenv.config();

let aiClient: GoogleGenAI | null = null;
const PC28_SIGNAL_URL = process.env.PC28_SIGNAL_URL || "https://pc28-ai-board-gray.vercel.app/api/ai-signal";

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY is not defined in the environment variables. Please configure it in your Secrets settings.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        }
      }
    });
  }
  return aiClient;
}

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT || 3000);

  // Middleware for parsing JSON requests
  app.use(express.json());

  // API Route for health checks
  app.get("/api/health", (req, res) => {
    res.json({ status: "ok", time: new Date().toISOString() });
  });

  app.get("/api/pc28-signal", async (req: express.Request, res: express.Response) => {
    try {
      const response = await fetch(PC28_SIGNAL_URL, {
        headers: {
          "Accept": "application/json",
          "Cache-Control": "no-cache",
          "Pragma": "no-cache",
          "User-Agent": "nexus-terminal-pc28-viewer"
        }
      });

      if (!response.ok) {
        throw new Error(`PC28 API returned HTTP ${response.status}`);
      }

      const payload = await response.json();
      res.status(200).json(payload);
    } catch (error: any) {
      console.error("PC28 Signal Proxy Failure:", error);
      res.status(502).json({
        ok: false,
        error: error.message || "Unable to read PC28 signal payload."
      });
    }
  });

  // API Route for Gemini quantitative analysis
  app.post("/api/analyze", async (req: express.Request, res: express.Response) => {
    try {
      const {
        currentIssue,
        currentNumbers,
        theoryAvgMiss,
        maxMiss,
        currentMiss,
        missChain,
        histAvg,
        pressureIdx,
        actualHitRate,
        deviation,
        recentHistoryText
      } = req.body;

      const ai = getGeminiClient();

      const prompt = `You are an expert quantitative casino analyst, specialized in draw pattern analysis, probability theory, and risk modeling.
Analyze the following live draw statistics from a high-density monitoring terminal:

Current Issue: #${currentIssue || '884211'}
Current Numbers Drawn: ${currentNumbers?.join(', ') || '04, 09, 12, 21, 33'}
Metrics:
- Theory Average Miss: ${theoryAvgMiss || '16.6'}
- Max Miss Streak: ${maxMiss || '104'}
- Current Miss Streak: ${currentMiss || '22'}
- Current Miss Chain (unbroken sequences of missing targets): ${missChain || '4'}
- Historical Avg Interval: ${histAvg || '16.4'}
- Current Pressure Index: ${pressureIdx || '0.784'} (Alert threshold: >0.75)
- Actual Hit Rate (last 1000 periods): ${actualHitRate || '4.8%'}
- Deviation vs Theoretical Baseline (6% expected, calculated variance): ${deviation || '-1.2%'}

Recent Hit History (most recent first):
${recentHistoryText || 'Issue #884210: HIT, Issue #884209: MISS, Issue #884208: MISS'}

Provide a deep mathematical and structural pattern diagnosis. Keep the response high-density, technical, precise, and highly concise (under 250 words), using a clean Markdown structure with bullet points. Speak with the objective, calm, and authoritative tone of a terminal system. Do not use flowery descriptors or self-reference 'as an AI'. Use technical terms such as:
- Reversion to the mean expectancy
- Poisson deviation distribution validation
- Vector direction of current miss interval force
- Concrete risk-adjusted strategy recommend (e.g., safe hedge coverage, active alert trigger, or standard monitoring standby)

Focus purely on mathematical probability risk assessment. Summarize in clear, compact sections.`;

      const response = await ai.models.generateContent({
        model: "gemini-3.5-flash",
        contents: prompt,
      });

      res.status(200).json({
        analysis: response.text,
      });
    } catch (error: any) {
      console.error("Gemini Analysis Failure:", error);
      res.status(500).json({
        error: error.message || "Unknown server-side error occurred while communicating with Gemini.",
      });
    }
  });

  // Handle Vite middleware of standard client assets
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req: express.Request, res: express.Response) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`[Nexus Console Server] Server running on http://0.0.0.0:${PORT} in ${process.env.NODE_ENV || 'development'} mode.`);
  });
}

startServer();
