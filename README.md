# WEB-PC Static Frontend

## Run Locally

**Prerequisites:** Node.js

1. Install dependencies:

   ```sh
   npm install
   ```

2. Configure the PC28 API base if needed:

   ```sh
   cp .env.example .env.local
   ```

   The static frontend reads `VITE_PC28_API_BASE` and requests:

   ```text
   ${VITE_PC28_API_BASE}/api/ai-signal
   ```

   If no local value is provided, it defaults to `http://66.42.54.206:3001`.

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
