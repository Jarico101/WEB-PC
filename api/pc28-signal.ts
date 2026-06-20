type ApiResponse = {
  setHeader(name: string, value: string): void;
  status(code: number): {
    json(payload: unknown): void;
  };
};

const PC28_SIGNAL_URL = process.env.PC28_SIGNAL_URL || "https://pc28-ai-board-gray.vercel.app/api/ai-signal";

export default async function handler(_req: unknown, res: ApiResponse) {
  try {
    const response = await fetch(PC28_SIGNAL_URL, {
      headers: {
        Accept: "application/json",
        "Cache-Control": "no-cache",
        Pragma: "no-cache",
        "User-Agent": "nexus-terminal-pc28-viewer"
      }
    });

    if (!response.ok) {
      throw new Error(`PC28 API returned HTTP ${response.status}`);
    }

    const payload = await response.json();
    res.setHeader("Cache-Control", "no-store, max-age=0");
    res.status(200).json(payload);
  } catch (error: any) {
    res.status(502).json({
      ok: false,
      error: error.message || "Unable to read PC28 signal payload."
    });
  }
}
