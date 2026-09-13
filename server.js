import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { fetchPc28Signal, resolvePc28SignalUrl } from "./scraper.js";

const ROOT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DIST_DIR = path.join(ROOT_DIR, "dist");
const MIME_TYPES = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon"
};

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Cache-Control,Pragma",
    "Content-Length": Buffer.byteLength(body)
  });
  res.end(body);
}

function sendStatic(req, res) {
  const parsed = new URL(req.url, "http://localhost");
  const pathname = decodeURIComponent(parsed.pathname);
  const safePath = path.normalize(pathname).replace(/^(\.\.[/\\])+/, "");
  const requested = path.join(DIST_DIR, safePath === "/" ? "index.html" : safePath);
  const filePath = requested.startsWith(DIST_DIR) && fs.existsSync(requested) && fs.statSync(requested).isFile()
    ? requested
    : path.join(DIST_DIR, "index.html");

  fs.readFile(filePath, (error, data) => {
    if (error) {
      sendJson(res, 404, { ok: false, error: "not_found" });
      return;
    }

    const ext = path.extname(filePath);
    res.writeHead(200, {
      "Content-Type": MIME_TYPES[ext] || "application/octet-stream",
      "Cache-Control": ext === ".html" ? "no-store" : "public, max-age=31536000, immutable"
    });
    res.end(data);
  });
}

async function handleRequest(req, res) {
  if (req.method === "OPTIONS") {
    sendJson(res, 204, {});
    return;
  }

  const parsed = new URL(req.url, "http://localhost");
  if (req.method === "GET" && parsed.pathname === "/api/health") {
    sendJson(res, 200, {
      ok: true,
      source: "web-pc-proxy",
      upstream: resolvePc28SignalUrl(),
      time: new Date().toISOString()
    });
    return;
  }

  if (req.method === "GET" && parsed.pathname === "/api/ai-signal") {
    try {
      sendJson(res, 200, await fetchPc28Signal());
    } catch (error) {
      sendJson(res, 502, {
        ok: false,
        error: error.message || "Unable to read PC-28 backend ai-signal payload",
        upstream: resolvePc28SignalUrl()
      });
    }
    return;
  }

  if (req.method === "GET" && ["/api/combo-signal", "/api/combo-sync", "/api/combo-history", "/api/combo/metrics"].includes(parsed.pathname)) {
    const comboBackend = process.env.COMBO_BACKEND_BASE_URL;
    if (!comboBackend) {
      sendJson(res, 503, { ok: false, dry_run: true, error: "combo_simulation_backend_not_configured" });
      return;
    }
    try {
      const target = new URL(parsed.pathname.replace(/^\/api/, ""), comboBackend);
      target.search = parsed.search;
      const upstream = await fetch(target);
      const body = await upstream.text();
      res.writeHead(upstream.status, {
        "Content-Type": upstream.headers.get("content-type") || "application/json; charset=utf-8",
        "Cache-Control": "no-store",
        "Access-Control-Allow-Origin": "*",
        "Content-Length": Buffer.byteLength(body)
      });
      res.end(body);
    } catch (error) {
      sendJson(res, 502, { ok: false, dry_run: true, error: error.message || "combo_backend_unavailable" });
    }
    return;
  }

  if (req.method !== "GET") {
    sendJson(res, 405, { ok: false, error: "method_not_allowed" });
    return;
  }

  sendStatic(req, res);
}

function startServer() {
  const port = Number(process.env.PORT || 3000);
  const host = process.env.HOST || "0.0.0.0";
  const server = http.createServer((req, res) => {
    handleRequest(req, res).catch((error) => {
      sendJson(res, 500, { ok: false, error: error.message || "internal_server_error" });
    });
  });

  server.listen(port, host, () => {
    console.log(`WEB-PC proxy listening on http://${host}:${port}`);
    console.log(`PC-28 upstream: ${resolvePc28SignalUrl()}`);
  });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  startServer();
}

export {
  handleRequest,
  startServer
};
