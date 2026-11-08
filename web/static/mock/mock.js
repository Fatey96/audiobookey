/* mock backend for offline frontend testing. Loaded ONLY when the app is
   opened with ?mock in the URL; app.js dynamic-imports it and installs a
   fetch shim. Never served from the real app path.

   Usage:
     python -m http.server 8900 --directory web/static
     open http://127.0.0.1:8900/?mock
   On Python 3.13+ prefer: python mock/serve.py  (fixes the .js MIME type) */

const BOOKS = [
  {
    id: "d4c1", title: "The Wind in the Moonless Hills", pdf_path: "books/wind.pdf",
    m4b_path: "out/wind.m4b", pages: 412, duration_s: 41115,
    status: "ready", ready_pages: 412, duration_ready_s: 41115,
    progress: { position_s: 5233, page: 52, updated_at: "2026-11-02T21:14:00Z" },
    bookmarks: [{ id: "bm1", position_s: 5233, page: 52, label: "The crossing", created_at: "2026-11-02T21:20:00Z" }],
    resumable: false,
  },
  {
    id: "a9f0", title: "Letters to a Young Cartographer", pdf_path: "books/letters.pdf",
    m4b_path: "", pages: 188, duration_s: 0,
    status: "preparing", ready_pages: 12, duration_ready_s: 1248,
    progress: { position_s: 18990, page: 184, updated_at: "2026-10-28T08:02:00Z" },
    bookmarks: [],
    resumable: false,
    job_spec: JSON.stringify({ voice: "em_alex", quality: "medium", gap: 0.2, lexicon_text: "", ocr: false }),
  },
  {
    id: "b77e", title: "Salt and Circuitry", pdf_path: "books/salt.pdf",
    m4b_path: "", pages: 96, duration_s: 0,
    status: "error", ready_pages: 30, duration_ready_s: 2820,
    progress: null,
    bookmarks: [],
    resumable: true,
    job_spec: JSON.stringify({ voice: "em_alex", quality: "medium", gap: 0.2, lexicon_text: "", ocr: false }),
  },
  /* P9 seeds: one paused, one queued and one cancelled book so every matrix
     state is demoable at ?mock */
  {
    id: "c31a", title: "The Tariff of Bees", pdf_path: "books/tariff.pdf",
    m4b_path: "", pages: 231, duration_s: 0,
    status: "paused", ready_pages: 64, duration_ready_s: 4273,
    progress: null, bookmarks: [],
    resumable: true,
    job_spec: JSON.stringify({ voice: "narrator", quality: "high", gap: 0.25, lexicon_text: "AI: A I", ocr: true }),
  },
  {
    id: "q77b", title: "A Register of Late Ferries", pdf_path: "books/ferries.pdf",
    m4b_path: "", pages: 156, duration_s: 0,
    status: "preparing", ready_pages: 0, duration_ready_s: 0,
    progress: null, bookmarks: [],
    resumable: false,
    job_spec: JSON.stringify({ voice: "em_alex", quality: "medium", gap: 0.2, lexicon_text: "", ocr: false }),
  },
  {
    id: "x07c", title: "Glass Gardens", pdf_path: "books/glass.pdf",
    m4b_path: "", pages: 96, duration_s: 0,
    status: "cancelled", ready_pages: 28, duration_ready_s: 1680,
    progress: null, bookmarks: [],
    /* legacy book: resumable, but no stored job_spec. Resume opens the
       prefilled convert form instead of a one-click requeue */
    resumable: true,
  },
];

const VOICES = [
  { name: "af_heart", quality: "high", engine: "kokoro", has_clone: false },
  { name: "em_alex", quality: "medium", engine: "kokoro", has_clone: false },
  { name: "narrator", quality: "high", engine: "chatterbox-v3", has_clone: true },
];

let nextJobId = 7;
let nextBookmarkId = 100;
const jobs = new Map();

/* books.job_spec (P9): JSON written at every submit and requeue. Legacy rows
   carry "" and cannot requeue with one click. */
function parseSpec(book) {
  try {
    const spec = typeof book.job_spec === "string" ? JSON.parse(book.job_spec) : book.job_spec;
    return spec && typeof spec === "object" ? spec : null;
  } catch (_) {
    return null;
  }
}

/* P9 seeds: a running job for the preparing book, one paused and one queued
   job, so every endpoint branch is demoable at ?mock. */
const seededRun = {
  id: "job3", status: "running", book_id: "a9f0", book_ids: ["a9f0"],
  message: "converting", page_count: 188, page_done: 12,
  chunk_done: 1, chunk_total: 60, cache_hits: 3,
  files: ["Letters to a Young Cartographer"],
};
seededRun._tick = () => {
  if (seededRun.status !== "running") return;
  seededRun.page_done = Math.min(seededRun.page_count, seededRun.page_done + 5);
  seededRun.chunk_done = Math.min(
    seededRun.chunk_total,
    Math.round((seededRun.page_done / seededRun.page_count) * seededRun.chunk_total)
  );
  seededRun.cache_hits = Math.max(seededRun.cache_hits, Math.floor(seededRun.page_done / 9));
  const book = BOOKS.find((b) => b.id === "a9f0");
  if (book) {
    if (seededRun.page_done > (book.ready_pages || 0)) {
      book.ready_pages = seededRun.page_done;
      book.duration_ready_s = Number((seededRun.page_done * perPage(book)).toFixed(1));
    }
    if (seededRun.page_done >= seededRun.page_count) {
      seededRun.status = "done";
      seededRun.message = "done";
      book.status = "ready";
      book.ready_pages = book.pages;
      book.duration_s = Number((book.pages * perPage(book)).toFixed(1));
      book.duration_ready_s = book.duration_s;
    }
  }
};
jobs.set(seededRun.id, seededRun);
jobs.set("job4", {
  id: "job4", status: "paused", book_id: "c31a", book_ids: ["c31a"],
  message: "paused by user", page_count: 231, page_done: 64,
  chunk_done: 205, chunk_total: 739, cache_hits: 0,
  files: ["The Tariff of Bees"],
});
jobs.set("job5", {
  id: "job5", status: "queued", book_id: "q77b", book_ids: ["q77b"],
  message: "waiting for the worker", page_count: 156, page_done: 0,
  chunk_done: 0, chunk_total: 500, cache_hits: 0,
  files: ["A Register of Late Ferries"],
});

function silentWav(seconds) {
  const rate = 8000;
  const samples = rate * seconds;
  const buf = new ArrayBuffer(44 + samples * 2);
  const view = new DataView(buf);
  const ascii = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
  ascii(0, "RIFF"); view.setUint32(4, 36 + samples * 2, true); ascii(8, "WAVE");
  ascii(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  ascii(36, "data"); view.setUint32(40, samples * 2, true);
  return new Uint8Array(buf);
}

const AUDIO = silentWav(600);
const PAGE_AUDIO = silentWav(30); /* one mock page: 30 silent seconds */

/* seconds per page for a book, derived from whatever duration it has */
function perPage(book) {
  const total = book.pages || 1;
  const totalDur = book.duration_s || book.duration_ready_s || total * 100;
  return Math.max(1, Math.round((totalDur / total) * 10) / 10);
}

function readyCount(book) {
  if (book.ready_pages != null) return book.ready_pages;
  return book.status === "ready" ? book.pages || 0 : 0;
}

/* GET /api/books/:id/pages, contract P7 shape. A preparing book creeps one
   page forward per poll (and flips to ready at the end) so the live player
   path can be watched end to end in mock mode. */
function pagesPayload(book) {
  const total = book.pages || 0;
  let ready = readyCount(book);
  if (book.status === "preparing" && ready < total) {
    ready = Math.min(total, ready + 1);
    book.ready_pages = ready;
    book.duration_ready_s = Number((ready * perPage(book)).toFixed(1));
    if (ready >= total) {
      book.status = "ready";
      book.duration_s = Number((total * perPage(book)).toFixed(1));
      book.duration_ready_s = book.duration_s;
    }
  }
  const pages = [];
  for (let n = 1; n <= ready; n++) {
    pages.push({
      page: n,
      duration_s: perPage(book),
      url: `/api/books/${book.id}/pages/${n}`,
    });
  }
  return {
    book_id: book.id,
    status: book.status,
    total_pages: total,
    ready_count: ready,
    duration_ready_s: Number((ready * perPage(book)).toFixed(1)),
    pages,
  };
}

function json(data, status, delay = 80) {
  return new Promise((resolve) => {
    setTimeout(() => resolve(new Response(JSON.stringify(data), {
      status, headers: { "Content-Type": "application/json" },
    })), delay);
  });
}

function coverSvg(title) {
  const esc = title.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240"><rect width="320" height="240" fill="#efe6d4"/><text x="32" y="66" font-family="Georgia, serif" font-size="26" fill="#292118">${esc}</text><rect x="32" y="88" width="80" height="4" fill="#7c3a12"/></svg>`;
  return new Response(svg, { status: 200, headers: { "Content-Type": "image/svg+xml" } });
}

export function installMock() {
  console.info("[mock] fetch shim active, no real backend calls will be made");
  const realFetch = window.fetch.bind(window);

  window.fetch = async (input, init = {}) => {
    const url = String(input instanceof Request ? input.url : input);
    if (!url.includes("/api/")) return realFetch(input, init);

    const method = (init.method || (input instanceof Request ? input.method : "GET")).toUpperCase();
    const path = url.slice(url.indexOf("/api/"));

    if (path === "/api/books" && method === "GET") return json(BOOKS);

    let m = path.match(/^\/api\/books\/([^/]+)$/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      return book ? json(book) : json({ error: "no such book" }, 404);
    }
    if (m && method === "DELETE") {
      const i = BOOKS.findIndex((b) => b.id === m[1]);
      if (i === -1) return json({ error: "no such book" }, 404);
      BOOKS.splice(i, 1);
      return json({ ok: true });
    }

    m = path.match(/^\/api\/books\/([^/]+)\/cover$/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      return book ? coverSvg(book.title) : new Response("no such book", { status: 404 });
    }

    m = path.match(/^\/api\/books\/([^/]+)\/audio/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return new Response("no such book", { status: 404 });
      return new Response(AUDIO.slice(0), {
        status: 200,
        headers: { "Content-Type": "audio/wav", "Accept-Ranges": "none" },
      });
    }

    m = path.match(/^\/api\/books\/([^/]+)\/pages\/(\d+)$/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return new Response("no such book", { status: 404 });
      const n = Number(m[2]);
      if (!Number.isInteger(n) || n < 1 || n > readyCount(book)) {
        return new Response("page not ready", { status: 404 });
      }
      return new Response(PAGE_AUDIO.slice(0), {
        status: 200,
        headers: { "Content-Type": "audio/wav", "Accept-Ranges": "none" },
      });
    }

    m = path.match(/^\/api\/books\/([^/]+)\/pages$/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return json({ error: "no such book" }, 404);
      return json(pagesPayload(book));
    }

    m = path.match(/^\/api\/books\/([^/]+)\/progress$/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      return book ? json(book.progress) : json({ error: "no such book" }, 404);
    }
    if (m && method === "PUT") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return json({ error: "no such book" }, 404);
      book.progress = { ...(book.progress || {}), ...JSON.parse(init.body || "{}") };
      return json(book.progress);
    }

    m = path.match(/^\/api\/books\/([^/]+)\/bookmarks$/);
    if (m && method === "GET") {
      const book = BOOKS.find((b) => b.id === m[1]);
      return book ? json(book.bookmarks || []) : json({ error: "no such book" }, 404);
    }
    if (m && method === "POST") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return json({ error: "no such book" }, 404);
      const bm = { id: `bm${nextBookmarkId++}`, ...JSON.parse(init.body || "{}"), created_at: new Date().toISOString() };
      book.bookmarks.push(bm);
      return json(bm);
    }

    m = path.match(/^\/api\/bookmarks\/([^/]+)$/);
    if (m && method === "DELETE") {
      for (const book of BOOKS) {
        const i = (book.bookmarks || []).findIndex((b) => b.id === m[1]);
        if (i !== -1) { book.bookmarks.splice(i, 1); return json({ ok: true }); }
      }
      return json({ error: "no such bookmark" }, 404);
    }

    /* P9 job controls. Requeue goes through the same FIFO as a fresh convert
       and reuses the same book row; pause and cancel are kill-and-mark. */
    m = path.match(/^\/api\/books\/([^/]+)\/convert$/);
    if (m && method === "POST") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return json({ error: "no such book" }, 404);
      const live = [...jobs.values()].find((j) =>
        (j.status === "queued" || j.status === "running") &&
        (j.book_ids || []).includes(book.id));
      if (live) return json({ error: "this book already has a job in the queue" }, 409);
      let sent = {};
      try {
        const parsed = JSON.parse(init.body || "{}");
        if (parsed && typeof parsed === "object") sent = parsed;
      } catch (_) { sent = {}; }
      const stored = parseSpec(book);
      if (!Object.keys(sent).length && !stored) {
        return json({ error: "no stored conversion parameters: send voice and quality" }, 400);
      }
      const spec = Object.keys(sent).length ? sent : stored;
      if (spec.quality && !["draft", "final", "low", "medium", "high"].includes(spec.quality)) {
        return json({ error: "quality must be draft, final, low, medium or high" }, 400);
      }
      book.job_spec = JSON.stringify(spec);
      book.status = "preparing";
      book.resumable = false;
      const jobId = `job${nextJobId++}`;
      /* pages made earlier still count: a requeue resumes from cache, so the
         counters start at whatever the book had ready */
      const cached = readyCount(book);
      const job = {
        id: jobId, status: "queued", book_id: book.id, book_ids: [book.id],
        message: "queued", page_count: book.pages || 0, page_done: cached,
        chunk_done: Math.round(cached * 3.2), chunk_total: Math.round((book.pages || 0) * 3.2),
        cache_hits: Math.round(cached * 3.2),
        files: [book.title],
      };
      job._tick = () => {
        if (job.status === "queued") { job.status = "running"; job.message = "converting"; return; }
        if (job.status !== "running") return;
        job.page_done = Math.min(job.page_count, job.page_done + 7);
        job.chunk_done = Math.min(
          job.chunk_total,
          Math.round((job.page_done / (job.page_count || 1)) * job.chunk_total)
        );
        job.cache_hits = Math.max(job.cache_hits, Math.round(cached * 3.2));
        if (job.page_done > (book.ready_pages || 0)) {
          book.ready_pages = job.page_done;
          book.duration_ready_s = Number((job.page_done * perPage(book)).toFixed(1));
        }
        if (job.page_done >= job.page_count) {
          job.status = "done";
          job.message = "done";
          book.status = "ready";
          book.ready_pages = book.pages;
          book.duration_s = Number((book.pages * perPage(book)).toFixed(1));
          book.duration_ready_s = book.duration_s;
          book.resumable = false;
        }
      };
      jobs.set(jobId, job);
      return json({ ...job }, 201);
    }

    m = path.match(/^\/api\/books\/([^/]+)\/pause$/);
    if (m && method === "POST") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return json({ error: "no such book" }, 404);
      const job = [...jobs.values()].find((j) =>
        (j.status === "queued" || j.status === "running") &&
        (j.book_ids || []).includes(book.id));
      if (!job) return json({ error: "no queued or running job on this book" }, 409);
      job.status = "paused";
      job.message = "paused by user";
      book.status = "paused";
      book.resumable = true;
      return json({ job_id: job.id, status: "paused" }, 202);
    }

    m = path.match(/^\/api\/books\/([^/]+)\/cancel$/);
    if (m && method === "POST") {
      const book = BOOKS.find((b) => b.id === m[1]);
      if (!book) return json({ error: "no such book" }, 404);
      /* the shelf matrix offers Cancel on paused books too, so the mock
         accepts paused jobs; the partial audio stays as it is */
      const job = [...jobs.values()].find((j) =>
        ["queued", "running", "paused"].includes(j.status) &&
        (j.book_ids || []).includes(book.id));
      if (!job) return json({ error: "no queued or running job on this book" }, 409);
      job.status = "cancelled";
      job.message = "cancelled by user";
      book.status = "cancelled";
      book.resumable = true;
      return json({ job_id: job.id, status: "cancelled" }, 202);
    }

    if (path === "/api/voices") return json(VOICES);

    if (path === "/api/convert" && method === "POST") {
      const jobId = `job${nextJobId++}`;
      jobs.set(jobId, {
        id: jobId, status: "running", book_id: null, message: "converting",
        page_count: 97 + Math.floor(Math.random() * 200), page_done: 0,
        chunk_done: 0, chunk_total: 12, cache_hits: 0,
      });
      const tickSource = jobs.get(jobId);
      tickSource._tick = () => {
        tickSource.page_done = Math.min(tickSource.page_count, tickSource.page_done + 7);
        tickSource.chunk_done = Math.min(tickSource.chunk_total, Math.floor((tickSource.page_done / tickSource.page_count) * 12));
        tickSource.cache_hits = Math.floor(tickSource.page_done / 9);
        if (tickSource.page_done >= tickSource.page_count) {
          tickSource.status = "done";
          const book = {
            id: `new${jobId}`, title: "A Field Guide to Quiet Places",
            pdf_path: "books/new.pdf", m4b_path: `out/${jobId}.m4b`,
            pages: tickSource.page_count, duration_s: tickSource.page_count * 92,
            status: "ready", ready_pages: tickSource.page_count,
            duration_ready_s: tickSource.page_count * 92,
            progress: null, bookmarks: [],
          };
          BOOKS.unshift(book);
          tickSource.book_id = book.id;
          tickSource.book_ids = [book.id];
          tickSource.message = "done";
        }
      };
      return json({ ...jobs.get(jobId) });
    }

    m = path.match(/^\/api\/jobs\/([^/]+)$/);
    if (m && method === "GET") {
      const job = jobs.get(m[1]);
      if (!job) return json({ error: "no such job" }, 404);
      if (job._tick) job._tick();
      return json({ ...job });
    }

    if (path === "/api/jobs") return json([...jobs.values()]);

    return json({ error: `mock does not handle ${method} ${path}` }, 404);
  };
}
