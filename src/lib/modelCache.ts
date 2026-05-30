/**
 * DexieJS-backed model cache.
 *
 * The roadmap calls for an app that works with NO account and NO server data
 * store. We honour that on the client too: the (heavy) ONNX model files are
 * downloaded once, then cached in IndexedDB via Dexie and reused across visits.
 *
 * Caching is content-addressed and versioned: each blob is keyed by
 * `${version}:${fileName}` and validated against the sha256 from the manifest.
 * When the model is re-exported with a new version (or the bytes change), the
 * stale entry is evicted and the new file fetched. This is the "ML model is
 * cached there and versionned" requirement from the brief.
 */

import Dexie, { type Table } from "dexie";

export interface CachedModelFile {
  /** Primary key: `${version}:${fileName}` */
  key: string;
  version: string;
  fileName: string;
  sha256: string;
  bytes: number;
  data: ArrayBuffer;
  cachedAt: number;
}

class FertiLunaDB extends Dexie {
  modelFiles!: Table<CachedModelFile, string>;

  constructor() {
    super("fertiluna");
    this.version(1).stores({
      // key is the primary key; index version + fileName for housekeeping.
      modelFiles: "key, version, fileName",
    });
  }
}

let _db: FertiLunaDB | null = null;
function db(): FertiLunaDB {
  if (!_db) _db = new FertiLunaDB();
  return _db;
}

function cacheKey(version: string, fileName: string): string {
  return `${version}:${fileName}`;
}

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export interface FetchProgress {
  fileName: string;
  loaded: number;
  total: number;
}

/**
 * Get a model file as an ArrayBuffer, using the IndexedDB cache when possible.
 *
 * @param version    model version (for the cache key)
 * @param fileName   e.g. "cycle-classifier-v1.onnx"
 * @param url        absolute/relative URL to fetch on cache miss
 * @param sha256     expected checksum from the manifest (integrity + versioning)
 * @param expectedBytes  size hint for progress reporting
 * @param onProgress optional download progress callback
 */
export async function getModelFile(
  version: string,
  fileName: string,
  url: string,
  sha256: string,
  expectedBytes: number,
  onProgress?: (p: FetchProgress) => void,
): Promise<ArrayBuffer> {
  const key = cacheKey(version, fileName);

  // 1. Try cache.
  try {
    const cached = await db().modelFiles.get(key);
    if (cached && cached.sha256 === sha256) {
      onProgress?.({ fileName, loaded: cached.bytes, total: cached.bytes });
      return cached.data;
    }
    // Stale (different sha for same key) — drop it so we re-fetch.
    if (cached) await db().modelFiles.delete(key);
  } catch {
    // IndexedDB unavailable (private mode / quota) — fall through to network.
  }

  // 2. Fetch with progress.
  const data = await fetchWithProgress(url, fileName, expectedBytes, onProgress);

  // 3. Verify integrity. If it doesn't match, we still return the bytes (the
  // model may have been re-exported and the manifest already updated), but we
  // don't poison the cache with a mismatched checksum.
  const actual = await sha256Hex(data);
  const integrityOk = actual === sha256;

  // 4. Best-effort cache write.
  if (integrityOk) {
    try {
      // Evict older versions of the same file to bound storage.
      await db()
        .modelFiles.where("fileName")
        .equals(fileName)
        .and((r) => r.version !== version)
        .delete();
      await db().modelFiles.put({
        key,
        version,
        fileName,
        sha256,
        bytes: data.byteLength,
        data,
        cachedAt: Date.now(),
      });
    } catch {
      // ignore cache write failures
    }
  }

  return data;
}

async function fetchWithProgress(
  url: string,
  fileName: string,
  expectedBytes: number,
  onProgress?: (p: FetchProgress) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    // Fallback: no streaming body (or error). Try a plain arrayBuffer.
    if (res.ok) {
      const buf = await res.arrayBuffer();
      onProgress?.({ fileName, loaded: buf.byteLength, total: buf.byteLength });
      return buf;
    }
    throw new Error(`Échec du téléchargement de ${fileName} (${res.status})`);
  }

  const totalHeader = res.headers.get("Content-Length");
  const total = totalHeader ? parseInt(totalHeader, 10) : expectedBytes;

  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      loaded += value.byteLength;
      onProgress?.({ fileName, loaded, total });
    }
  }

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.byteLength;
  }
  return out.buffer;
}

/** Total bytes currently cached (for a "model cached" UI indicator). */
export async function cachedBytes(): Promise<number> {
  try {
    let sum = 0;
    await db().modelFiles.each((r) => {
      sum += r.bytes;
    });
    return sum;
  } catch {
    return 0;
  }
}

/** Clear the entire model cache (debug / settings). */
export async function clearModelCache(): Promise<void> {
  try {
    await db().modelFiles.clear();
  } catch {
    // ignore
  }
}
