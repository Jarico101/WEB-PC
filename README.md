# WEB-PC Static Frontend

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies:

   ```sh
   npm install
   ```

2. Configure the PC-28 backend source if needed:

   ```sh
   cp .env.example .env.local
   ```

   WEB-PC now reads through its own backend:

   ```text
   pc28.ai/ajax.php -> PC-28 backend -> WEB-PC backend -> WEB-PC frontend
   ```

   `server.js` / `scraper.js` read `PC28_BACKEND_BASE_URL` and request:

   ```text
   ${PC28_BACKEND_BASE_URL}/api/ai-signal
   ```

   If no local value is provided, WEB-PC uses `https://api.startpops.xyz` as the PC-28 backend.

   The browser defaults to same-origin:

   ```text
   /api/ai-signal
   ```

   `server.js` tries the live upstream first. If the upstream is unavailable, it serves a
   `fallback_history` payload from `data/history.jsonl` so the UI and agent can keep
   reading the expected schema.

3. Start Vite:

   ```sh
   npm run dev
   ```

## Build

```sh
npm run lint
npm run build
```

The production output is written to `dist/` and can be served by any static host.
