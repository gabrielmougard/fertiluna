/**
 * Run history — a private, on-device log of past cycle analyses (DexieJS).
 *
 * The brief: let a user come back and find their previous images / results,
 * and clear them if they want. We honour the privacy contract ("data never
 * leaves your device") by storing everything in IndexedDB on the client — no
 * account, no server. This is a separate database from the model cache
 * (`fertiluna` in modelCache.ts) so the two evolve independently and clearing
 * history never evicts the heavy cached model.
 *
 * What we store per run:
 *   - a small JPEG thumbnail of the imported screenshot (downscaled, optional),
 *   - the rendered chart SVG (so a run re-opens instantly, offline),
 *   - the raw per-day inputs (temps / LH) so the user can reload + re-analyse,
 *   - the verdict summary (title, tone, confidence) for the list view.
 *
 * Storage is bounded: thumbnails are capped in size and we keep at most
 * MAX_RUNS entries (oldest evicted first).
 */

import Dexie, { type Table } from "dexie";

export interface CycleRun {
  /** Auto-incrementing primary key. */
  id?: number;
  createdAt: number;
  locale: string;
  /** Verdict for the list view. */
  title: string;
  summary: string;
  tone: string;
  confidencePct: number;
  /** Downscaled JPEG data URL of the source screenshot, if one was imported. */
  thumbnail: string | null;
  /** Rendered chart SVG markup (re-display without recomputation). */
  chartSvg: string;
  /** Raw inputs so a run can be reloaded into the editable table. */
  temps: (number | null)[];
  lh: (number | null)[];
  /** The analyzeCycle() result, so a run re-renders offline with no recompute.
   *  Stored as opaque JSON (structured-clone safe). */
  analysis: unknown;
}

/** Lightweight projection for the list view (no heavy payload). */
export type CycleRunSummary = Omit<
  CycleRun,
  "chartSvg" | "temps" | "lh" | "analysis"
> & {
  id: number;
};

const MAX_RUNS = 30;

class RunHistoryDB extends Dexie {
  runs!: Table<CycleRun, number>;

  constructor() {
    super("fertiluna-runs");
    this.version(1).stores({
      // primary key ++id (auto), index createdAt for chronological listing.
      runs: "++id, createdAt",
    });
  }
}

let _db: RunHistoryDB | null = null;
function db(): RunHistoryDB {
  if (!_db) _db = new RunHistoryDB();
  return _db;
}

/** Persist a run, evicting the oldest beyond MAX_RUNS. Returns the new id. */
export async function saveRun(run: Omit<CycleRun, "id">): Promise<number | null> {
  try {
    const id = await db().runs.add(run);
    // Bound storage: drop the oldest rows past the cap.
    const count = await db().runs.count();
    if (count > MAX_RUNS) {
      const overflow = count - MAX_RUNS;
      const oldest = await db()
        .runs.orderBy("createdAt")
        .limit(overflow)
        .primaryKeys();
      await db().runs.bulkDelete(oldest);
    }
    return id as number;
  } catch {
    return null; // IndexedDB unavailable (private mode / quota) — silently skip.
  }
}

/** List runs newest-first, without the heavy SVG / inputs fields. */
export async function listRuns(): Promise<CycleRunSummary[]> {
  try {
    const rows = await db().runs.orderBy("createdAt").reverse().toArray();
    return rows.map(
      ({ chartSvg: _svg, temps: _t, lh: _l, analysis: _a, ...rest }) =>
        rest as CycleRunSummary,
    );
  } catch {
    return [];
  }
}

/** Fetch a single full run (for re-opening). */
export async function getRun(id: number): Promise<CycleRun | undefined> {
  try {
    return await db().runs.get(id);
  } catch {
    return undefined;
  }
}

/** Delete one run by id. */
export async function deleteRun(id: number): Promise<void> {
  try {
    await db().runs.delete(id);
  } catch {
    /* ignore */
  }
}

/** Wipe the entire history ("clean my runs"). */
export async function clearRuns(): Promise<void> {
  try {
    await db().runs.clear();
  } catch {
    /* ignore */
  }
}

/** How many runs are stored (for showing/hiding the history panel). */
export async function runCount(): Promise<number> {
  try {
    return await db().runs.count();
  } catch {
    return 0;
  }
}
