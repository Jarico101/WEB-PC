/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_PC28_API_BASE?: string;
  readonly VITE_PC28_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
