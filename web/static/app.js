/* audiobookey reader frontend. Vanilla ES module, no framework, works offline.
   Code talks only to the routes in BRIEF-P3-CONTRACT.md. With ?mock in the
   URL it dynamic-imports mock/mock.js, a fetch shim; the mock never ships in
   the real app path. The audio element is the source of truth. */

const API = "/api";

/* ---------- tiny helpers ---------- */

const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

/* The only way code hides a node. Assigning to the hidden property of an
   SVG element creates a JS expando, not the attribute, so the node never
   hides (the "keeps buffering" bug). The attribute also survives
   re-attachments and works on SVG, where the property is not reliably
   reflected. */
const setHidden = (node, on) => {
  if (!node) return;
  if (on) node.setAttribute("hidden", "");
  else node.removeAttribute("hidden");
};
const icon = (name) => {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "icon");
  svg.setAttribute("width", "24");
  svg.setAttribute("height", "24");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.appendChild(use);
  return svg;
};

const fmtTime = (s) => {
  const t = Math.max(0, Math.round(s || 0));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  return `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
};

const fmtDur = (s) => {
  if (!s) return "";
  const min = Math.round(s / 60);
  if (min < 60) return `${min} min`;
  return `${Math.floor(min / 60)} h ${min % 60} min`;
};

/* ---------- api layer ---------- */

async function api(path, options) {
  const res = await fetch(`${API}${path}`, options);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body && body.error) msg = body.error;
    } catch (_) { /* non-JSON error body keeps the status line */ }
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

const getBooks = () => api("/books");
const getBook = (id) => api(`/books/${id}`);
const deleteBook = (id) => api(`/books/${id}`, { method: "DELETE" });
const getProgress = (id) => api(`/books/${id}/progress`);
const putProgress = (id, progress) =>
  api(`/books/${id}/progress`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(progress),
  });
const getBookmarks = (id) => api(`/books/${id}/bookmarks`);
const addBookmark = (id, b) =>
  api(`/books/${id}/bookmarks`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b),
  });
const deleteBookmark = (bmId) => api(`/bookmarks/${bmId}`, { method: "DELETE" });
const getVoices = () => api("/voices");
const getJob = (jobId) => api(`/jobs/${jobId}`);
/* P9 job controls (BRIEF-P9-JOB-CONTROLS.md): pause and cancel act on the
   book's live job; convert with an empty body requeues the stored job_spec,
   and with a params object requeues a legacy book that has none stored. */
const pauseBook = (id) => api(`/books/${id}/pause`, { method: "POST" });
const cancelBook = (id) => api(`/books/${id}/cancel`, { method: "POST" });
const requeueBook = (id, params) =>
  api(`/books/${id}/convert`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params || {}),
  });
const coverUrl = (id) => `${API}/books/${id}/cover`;
const audioUrl = (id) => `${API}/books/${id}/audio?format=m4b`;

/* ---------- state ---------- */

const state = {
  theme: localStorage.getItem("ab-theme") || null,
  books: [],
  jobs: null, /* last GET /api/jobs result: array or null (unknown); the one source for which books run, sit in the queue or rest paused */
  pollers: new Map(),
  sentences: [],
  activeIdx: -1,
};

/* ---------- theme toggle ---------- */

function initTheme() {
  const root = document.documentElement;
  if (state.theme) root.dataset.theme = state.theme;
  const btn = $("#theme-toggle");
  /* presentation only: the state source of truth stays root.dataset.theme
     + localStorage "ab-theme"; aria-pressed mirrors the resolved dark mode */
  const syncSwitch = () => {
    const prefersDark = matchMedia("(prefers-color-scheme: dark)").matches;
    const current = root.dataset.theme || (prefersDark ? "dark" : "light");
    btn.setAttribute("aria-pressed", String(current === "dark"));
  };
  syncSwitch();
  btn.addEventListener("click", () => {
    const prefersDark = matchMedia("(prefers-color-scheme: dark)").matches;
    const current = root.dataset.theme || (prefersDark ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    state.theme = next;
    localStorage.setItem("ab-theme", next);
    syncSwitch();
  });
}

/* ---------- fallback cover ---------- */

function coverEl(bookId, alt) {
  const img = el("img");
  img.alt = alt;
  img.loading = "lazy";
  img.src = coverUrl(bookId);
  img.addEventListener("error", () => {
    const fb = el("div", "cover-fallback");
    fb.setAttribute("role", "img");
    fb.setAttribute("aria-label", alt);
    fb.textContent = bookId.slice(0, 2).toUpperCase();
    img.replaceWith(fb);
  });
  return img;
}

/* ---------- library view ---------- */

async function renderLibrary(opts = {}) {
  /* no stopAllPollers here: trackJob adopts the existing panel and keeps one
     poller per job id, so blindly clearing intervals only orphans running
     conversions and made re-attachment spawn a second card */
  /* a job finishing while the user listens to that very book must refresh the
     shelf data without yanking them out of the player; the router still owns
     view changes for #/book/* navigation */
  const inPlayer = /^#\/book\//.test(location.hash);
  if (opts.show !== false && !inPlayer) showView("library");
  setHidden($("#shelf-loading"), false);
  setHidden($("#shelf-empty"), true);
  setHidden($("#shelf-error"), true);
  try {
    /* books and their live jobs arrive together: a preparing book alone
       cannot say whether it is queued or already running, and a just-paused
       book still reads preparing until this refetch lands */
    const [books, jobs] = await Promise.all([
      getBooks(),
      api("/jobs").catch(() => null),
    ]);
    state.books = books;
    state.jobs = Array.isArray(jobs) ? jobs : null;
    setHidden($("#shelf-loading"), true);
    renderContinue();
    renderShelf();
    void renderVoices();
    reattachRunningJobs(state.jobs);
  } catch (err) {
    setHidden($("#shelf-loading"), true);
    setHidden($("#shelf-error"), false);
  }
}

function renderContinue() {
  const slot = $("#continue-body");
  slot.replaceChildren();
  const inProgress = state.books
    .filter((b) => b.progress)
    .sort((a, b) => new Date(b.progress.updated_at) - new Date(a.progress.updated_at));
  if (inProgress.length === 0) {
    const empty = el("p", "empty-copy", "Nothing in progress yet. Pick up a book from the shelf, or add one below.");
    slot.appendChild(empty);
    return;
  }
  const book = inProgress[0];
  const card = el("article", "continue card");
  const cover = coverEl(book.id, `Cover of ${book.title}`);
  cover.className = "continue-cover";

  const meta = el("div", "continue-text");
  const title = el("h3", "continue-title", book.title);
  const pos = el("p", "continue-meta");
  /* a preparing book has duration_s = 0; its playable length so far is
     duration_ready_s, so the "of" figure never renders as an empty string */
  const totalDur = book.duration_s || book.duration_ready_s || 0;
  const pct = totalDur ? Math.round((book.progress.position_s / totalDur) * 100) : 0;
  pos.textContent = totalDur
    ? `Resume at ${fmtTime(book.progress.position_s)} of ${fmtDur(totalDur)}, ${pct}%`
    : `Resume at ${fmtTime(book.progress.position_s)}`;

  const play = el("button", "btn btn--accent");
  play.type = "button";
  play.appendChild(icon("play"));
  play.appendChild(el("span", "btn-label", "Continue"));
  play.setAttribute("aria-label", `Continue ${book.title} at ${fmtTime(book.progress.position_s)}`);
  play.addEventListener("click", () => { location.hash = `#/book/${book.id}`; });

  meta.append(title, pos);
  card.append(cover, meta, play);
  slot.appendChild(card);
}

/* ---------- shelf: the P9 state matrix ---------- */

/* The live job for a book, or null across GET /api/jobs by job.book_ids. A
   requeue shares its book with the older paused job, so the most active phase
   wins (running beats queued beats paused) and the higher job id breaks ties. */
function findLiveJob(bookId) {
  if (!Array.isArray(state.jobs)) return null;
  const want = String(bookId);
  const rank = (job) =>
    job.status === "running" ? 3 : job.status === "queued" ? 2 : job.status === "paused" ? 1 : 0;
  const order = (job) => {
    const n = Number(String(job.id).replace(/\D/g, ""));
    return Number.isFinite(n) ? n : 0;
  };
  let best = null;
  let bestKey = "";
  for (const job of state.jobs) {
    if (!job || !rank(job)) continue;
    const ids = Array.isArray(job.book_ids) ? job.book_ids : [job.book_id];
    if (!ids.map(String).includes(want)) continue;
    const key = `${rank(job)}:${order(job)}`;
    if (key > bestKey) { best = job; bestKey = key; }
  }
  return best;
}

/* book row plus live job decide the matrix row (BRIEF-P9): badge vocabulary,
   buttons and the resume hint all hang off this key */
function tileKey(book) {
  const status = book.status || "ready";
  if (status === "ready") return "ready";
  if (status === "error") return "error";
  if (status === "cancelled") return "cancelled";
  if (status === "paused") return "paused";
  /* preparing: the job dict owns the phase. A queued book shows QUEUED, and a
     pause the book row has not picked up yet already shows PAUSED; without a
     job list the old PREPARING reading stays the honest fallback */
  const job = findLiveJob(book.id);
  if (job && job.status === "queued") return "queued";
  if (job && job.status === "paused") return "paused";
  return "running";
}

/* badge copy per matrix state: QUEUED carries no counts, nothing measurable
   has happened until the worker picks the job up */
function tileBadge(key, readyPages, totalPages) {
  const label = {
    running: "PREPARING",
    queued: "QUEUED",
    paused: "PAUSED",
    error: "INTERRUPTED",
    cancelled: "CANCELLED",
  }[key];
  if (!label) return "";
  return key === "queued" ? label : `${label} ${readyPages}/${totalPages}`;
}

function tileBadgeTitle(key, readyPages, totalPages) {
  const ofPages = `${readyPages} of ${totalPages} pages ready`;
  if (key === "queued") return `Waiting in the conversion queue, ${ofPages}`;
  if (key === "paused") return `Paused with ${ofPages}, nothing is lost`;
  if (key === "cancelled") return `Cancelled, storage wiped — requeue to re-synthesize from scratch`;
  return ofPages;
}

/* inline job-control failure: the house .form-error, never a toast */
function paintTileError(tile, message) {
  const old = $(".form-error", tile);
  if (old) old.remove();
  const alert = el("p", "form-error");
  alert.setAttribute("role", "alert");
  alert.appendChild(icon("alert"));
  alert.appendChild(el("span", null, message));
  tile.appendChild(alert);
}

/* Pause, cancel and resume are one POST each. The pressed button takes
   aria-busy, every job button on the tile disables: the state they read is
   mid-change, and a racing second command only earns a confusing 409. A
   failure re-enables everything and paints the inline tile error. */
function runTileAction(tile, buttons, btn, book, kind) {
  return async () => {
    btn.setAttribute("aria-busy", "true");
    for (const b of buttons) b.disabled = true;
    try {
      if (kind === "resume") {
        if (!bookSpec(book)) { await openRequeueForm(book); return; }
        const job = await requeueBook(book.id, {});
        trackJob(job.id, { count: 1, file: book.title });
      } else if (kind === "pause") {
        await pauseBook(book.id);
      } else {
        await cancelBook(book.id);
      }
      await refreshShelfData();
      /* the refresh replaced the pressed button: a keyboard user just lost
         their focus target, so hand focus back to the same tile's first job
         control of the new state (script focus under a pointer click does not
         paint a ring, so pointer users notice nothing) */
      const freshTile = document.querySelector(`#shelf .tile[data-book-id="${String(book.id)}"]`);
      const next = freshTile && freshTile.querySelector(".tile-actions .icon-btn[data-kind]");
      if (next && document.activeElement === document.body) next.focus();
    } catch (err) {
      paintTileError(tile, `Could not ${kind} this book: ${err.message}`);
    } finally {
      btn.removeAttribute("aria-busy");
      for (const b of buttons) b.disabled = false;
    }
  };
}

/* the job buttons a state offers (P9 matrix). Resume honors the one-click
   (stored job_spec) versus confirm-form (legacy book) split at click time. */
function buildJobButtons(tile, book, key, resumable) {
  const CANCEL_TITLE = "Stop this conversion now. The book stays, finished pages stay playable.";
  const RESUME_TITLE = "Continue where it left off — nothing is lost";
  const mk = (kind, glyph, label, title) => {
    const b = el("button", "icon-btn");
    b.type = "button";
    b.dataset.kind = kind;
    b.setAttribute("aria-label", label);
    b.title = title;
    b.appendChild(icon(glyph));
    return b;
  };
  const buttons = [];
  if (key === "running") {
    buttons.push(
      mk("pause", "pause", `Pause ${book.title}`,
        "Pause the conversion. Nothing is lost, resume continues from the cached pages."),
      mk("cancel", "x", `Cancel ${book.title}`, CANCEL_TITLE)
    );
  } else if (key === "queued") {
    buttons.push(mk("cancel", "x", `Cancel ${book.title}`, CANCEL_TITLE));
  } else if (key === "paused") {
    buttons.push(
      mk("resume", "requeue", `Resume ${book.title}`, RESUME_TITLE),
      mk("cancel", "x", `Cancel ${book.title}`, CANCEL_TITLE)
    );
  } else if ((key === "error" || key === "cancelled") && resumable) {
    buttons.push(
      key === "error"
        ? mk("resume", "requeue", `Resume ${book.title}`, RESUME_TITLE)
        : mk("resume", "requeue", `Requeue ${book.title}`,
          "Queue this book again. Nothing is lost, the cached pages are reused.")
    );
  }
  for (const b of buttons) {
    b.addEventListener("click", runTileAction(tile, buttons, b, book, b.dataset.kind));
  }
  return buttons;
}

function renderShelf() {
  const shelf = $("#shelf");
  shelf.querySelectorAll(".tile").forEach((t) => t.remove());
  /* setHidden(node, on): "on" HIDES. Empty copy shows only when nothing is on
     the shelf — the previous inversion painted it above a full shelf. */
  setHidden($("#shelf-empty"), state.books.length > 0);
  for (const book of state.books) {
    const tile = el("article", "tile");
    tile.dataset.bookId = String(book.id); /* lets prepare-card logic find the fresh book */

    const cover = coverEl(book.id, `Cover of ${book.title}`);
    cover.className = "tile-cover";

    const title = el("h3", "tile-title", book.title);

    /* unfinished books: the duration slot shows a state badge instead
       (fmtDur(0) is "", which would leave the slot empty) */
    const key = tileKey(book);
    const preparing = key !== "ready";
    const readyPages = Number(
      book.ready_pages != null ? book.ready_pages : preparing ? 0 : book.pages || 0
    );
    const totalPages = Number(book.pages || 0);
    const resumable = book.resumable !== false &&
      (key === "paused" || key === "error" || key === "cancelled");

    const row = el("div", "tile-row");
    /* the tile actions wrap as one unit: a narrow tile drops the group to its
       own line instead of splitting the listen button off the rest */
    const actions = el("div", "tile-actions");
    const lead = preparing ? el("span", "tile-badge") : el("span", "time", fmtDur(book.duration_s));
    if (preparing) {
      lead.textContent = tileBadge(key, readyPages, totalPages);
      /* the tile is narrow: keep the readable form on hover for assistive tech
         and for anyone who wants the long version */
      lead.title = tileBadgeTitle(key, readyPages, totalPages);
    }
    const open = el("button", "icon-btn");
    open.type = "button";
    open.setAttribute("aria-label", `Listen to ${book.title}`);
    open.appendChild(icon("play"));
    if (preparing && readyPages === 0) {
      open.disabled = true;
      open.setAttribute("aria-disabled", "true");
      open.title = "No pages ready yet";
    }
    open.addEventListener("click", () => { location.hash = `#/book/${book.id}`; });

    /* Remove appears only where the matrix offers it: a converting book cannot
       be removed, the server refuses it (409) because deleting the row
       orphans the running job and its audio becomes unreachable */
    let remove = null;
    if (key === "ready" || key === "error" || key === "cancelled") {
      remove = el("button", "icon-btn btn--danger");
      remove.type = "button";
      remove.setAttribute("aria-label", `Remove ${book.title} from the library`);
      remove.appendChild(icon("trash"));
      remove.addEventListener("click", async () => {
        remove.setAttribute("aria-busy", "true");
        try {
          await deleteBook(book.id);
          state.books = state.books.filter((b) => b.id !== book.id);
          tile.remove();
          renderContinue();
        } catch (err) {
          remove.removeAttribute("aria-busy");
        }
      });
    }

    actions.append(open);
    for (const b of buildJobButtons(tile, book, key, resumable)) actions.appendChild(b);
    if (remove) actions.appendChild(remove);
    row.append(lead, actions);
    tile.append(cover, title, row);
    if (resumable) {
      /* the resume promise spelled out under the badge, for the states that
         can go back through the queue */
      tile.appendChild(el("p", "tile-hint",
        key === "cancelled"
          ? "Requeue continues from the cached pages. Nothing is lost."
          : "Resume continues from the cached pages. Nothing is lost."
      ));
    }
    if (book.progress && book.duration_s) {
      const bar = el("div", "progress");
      bar.setAttribute("role", "progressbar");
      bar.setAttribute("aria-label", `${book.title} progress`);
      bar.setAttribute("aria-valuemin", "0");
      bar.setAttribute("aria-valuemax", "100");
      const pct = Math.min(100, (book.progress.position_s / book.duration_s) * 100);
      bar.setAttribute("aria-valuenow", String(Math.round(pct)));
      const fill = el("div", "progress-fill");
      fill.style.width = `${pct}%`;
      bar.appendChild(fill);
      tile.appendChild(bar);
    }
    shelf.appendChild(tile);
  }
}

/* ---------- convert: upload + job polling ---------- */

let voiceOptions = [];
let qualityOptions = [
  { value: "low", label: "Low (fast)" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High (best)" },
];

/* quality tier -> the voice engine it needs (mirrors cli.QUALITY_ENGINE_ALIASES) */
const TIER_ENGINE = { low: "draft", medium: "final", high: "final" };

/* voice pills only offer voices the selected quality tier can actually use:
   the Chatterbox final engine rejects bare draft voice ids at job time, so
   offering them under medium/high just converts into a guaranteed error. */
function renderVoicePills() {
  const wrap = $("#conv-voices");
  const tier = pillValue($("#conv-quality")) || "medium";
  const wantEngine = TIER_ENGINE[tier];
  const compatible = voiceOptions.filter(
    (v) => typeof v === "string" || !v.engine || v.engine === wantEngine
  );
  buildPills(wrap, compatible.length ? compatible : voiceOptions, compatible[0] || voiceOptions[0], { preview: true });
}

async function renderVoices() {
  const wrap = $("#conv-voices");
  wrap.replaceChildren();
  try {
    voiceOptions = (await getVoices()).map((v) => ({
      value: v.name,
      engine: typeof v.engine === "string" ? v.engine : null,
      label: typeof v.label === "string" && v.label
        ? v.label
        : v.name.replace(/_/g, " "),
    }));
  } catch (err) {
    voiceOptions = [];
  }
  if (voiceOptions.length === 0) voiceOptions = ["default"];
  buildPills($("#conv-quality"), qualityOptions, "medium");
  /* P6B: one-line hint under the quality pills so users see where "draft"
     went. Re-created on each renderVoices() call, never duplicated. */
  const hint = el("p", "field-hint pills-hint",
    "Low is the fastest draft voice; medium and high use the natural " +
    "clone-grade voice (high also encodes at best quality)."
  );
  hint.dataset.for = "conv-quality";
  const parent = wrap.parentElement;
  const stale = parent.querySelector(".pills-hint");
  if (stale) stale.remove();
  wrap.insertAdjacentElement("afterend", hint);
  renderVoicePills();
}

function buildPills(wrap, options, selected, opts = {}) {
  wrap.replaceChildren();
  for (const opt of options) {
    const value = typeof opt === "string" ? opt : opt.value;
    const label = typeof opt === "string" ? opt : opt.label;
    const btn = el("button", "pill-btn", label);
    btn.type = "button";
    btn.dataset.value = value;
    btn.setAttribute("aria-pressed", String(value === selected));
    btn.addEventListener("click", () => {
      wrap.querySelectorAll(".pill-btn").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      if (wrap.id === "conv-quality") renderVoicePills();
    });
    wrap.appendChild(btn);
    if (opts.preview) {
      const play = el("button", "icon-btn pill-preview");
      play.type = "button";
      play.title = "Preview — first play synthesizes the sample (up to a minute on the natural voice), after that it plays instantly from cache";
      play.setAttribute("aria-label", `Preview the ${label} voice`);
      const playIcon = icon("play");
      const spinner = icon("spinner");
      spinner.classList.add("spin");
      setHidden(spinner, true);
      play.append(playIcon, spinner);
      play.addEventListener("click", () => previewVoice(value, playIcon, spinner, play));
      wrap.appendChild(play);
    }
  }
  /* a pill group always keeps exactly one selection: submit-time code reads
     the pressed pill, and an all-unselected group silently falls back to the
     server default (which for voices may not fit the chosen quality tier) */
  if (!wrap.querySelector('.pill-btn[aria-pressed="true"]')) {
    const first = wrap.querySelector(".pill-btn");
    if (first) first.setAttribute("aria-pressed", "true");
  }
}

/* POST /api/voices/{name}/preview (contract P5): a sample WAV synthesized with
   the voice's own engine. One shared <audio> keeps previews exclusive: a new
   click stops whatever is playing, and only the latest click owns playback.
   A per-voice blob cache (LRU, ~8 voices) makes repeat previews instant with
   no network round trip; evicted objectURLs are revoked. */
const previewAudio = document.createElement("audio");
const previewCache = new Map(); /* voice name -> objectURL, most recent last */
const PREVIEW_CACHE_LIMIT = 8;
let previewSeq = 0; /* bumps per click; stale responses never own playback */
let playingPreview = null; /* { seq, playBtn, playIcon, spinner } currently audible */

/* One pill, one visible icon at all times: default shows the play icon,
   data-state="loading" shows only the spinner, data-playing shows neither
   (the accent background is the playing state). resetPreviewPill is the single
   way back to default, so a superseded or finished pill can never keep a
   spinner alive. */
const resetPreviewPill = (pill) => {
  setHidden(pill.playIcon, false);
  setHidden(pill.spinner, true);
  pill.playBtn.removeAttribute("data-playing");
  pill.playBtn.removeAttribute("data-state");
};

const stopPreviewAudio = () => {
  previewAudio.pause();
  if (playingPreview) resetPreviewPill(playingPreview);
  playingPreview = null;
};

previewAudio.addEventListener("ended", stopPreviewAudio);
previewAudio.addEventListener("error", stopPreviewAudio);

function cachedPreviewUrl(name) {
  const url = previewCache.get(name);
  if (url === undefined) return null;
  previewCache.delete(name);
  previewCache.set(name, url); /* re-insert as most recently used */
  return url;
}

function storePreviewUrl(name, url) {
  previewCache.set(name, url);
  while (previewCache.size > PREVIEW_CACHE_LIMIT) {
    const oldest = previewCache.keys().next().value;
    const oldestUrl = previewCache.get(oldest);
    previewCache.delete(oldest);
    if (oldestUrl === previewAudio.src) stopPreviewAudio();
    URL.revokeObjectURL(oldestUrl);
  }
  return url;
}

let pendingPreview = null; /* { seq, playBtn, playIcon, spinner } synthesizing */

async function previewVoice(name, playIcon, spinner, playBtn) {
  if (playIcon.hasAttribute("hidden")) return; /* already loading or playing */
  setHidden(playIcon, true);
  setHidden(spinner, false);
  playBtn.setAttribute("data-state", "loading"); /* spinner only, per CSS */
  const seq = ++previewSeq;
  stopPreviewAudio(); /* this click is now the only candidate for playback */
  /* superseding an in-flight preview resets ITS pill right away: two visible
     spinners read as "buffering forever", and the stale request can never
     own playback anyway */
  if (pendingPreview && pendingPreview.seq !== seq) resetPreviewPill(pendingPreview);
  pendingPreview = { seq, playBtn, playIcon, spinner };
  try {
    let url = cachedPreviewUrl(name);
    if (!url) {
      const res = await fetch(`${API}/voices/${encodeURIComponent(name)}/preview`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      url = storePreviewUrl(name, URL.createObjectURL(await res.blob()));
    }
    if (seq !== previewSeq) return; /* superseded: discard the sound silently */
    if (pendingPreview && pendingPreview.seq === seq) pendingPreview = null;
    playingPreview = { seq, playBtn, playIcon, spinner };
    /* synth done: the pill shows the playing state, spinner and play icon
       both hidden so exactly one icon (here: none) is ever visible */
    setHidden(spinner, true);
    playBtn.setAttribute("data-playing", "true");
    playBtn.setAttribute("data-state", "playing");
    previewAudio.src = url;
    await previewAudio.play();
  } catch (err) {
    /* preview is best-effort; a failure does not block converting */
    if (pendingPreview && pendingPreview.seq === seq) pendingPreview = null;
    if (playingPreview && playingPreview.seq === seq) {
      resetPreviewPill(playingPreview);
      playingPreview = null;
    }
  } finally {
    /* restore this pill unless it owns the live audio (ended/supersede resets it) */
    if (!playingPreview || playingPreview.seq !== seq) {
      resetPreviewPill({ playBtn, playIcon, spinner });
    }
  }
}

const pillValue = (wrap) => {
  const active = wrap.querySelector('.pill-btn[aria-pressed="true"]');
  return active ? active.dataset.value : "";
};

/* ---------- resume and requeue (P9) ---------- */

/* A resumable book with a stored job_spec requeues with one click. A legacy
   book (resumable, but submitted before job_spec existed) cannot: its Resume
   opens the convert form prefilled with that book, the user confirms voice and
   quality, and the submit targets POST /api/books/{id}/convert so the same
   book row goes back through the queue, never a duplicate row. */
let convertTarget = null; /* { book, spec } while the form is in requeue mode */

/* pre-P6B spec qualities live on as tiers: draft is low, final is medium */
const QUALITY_ALIAS = { draft: "low", final: "medium" };

/* the stored job_spec: JSON text from the book row (parsed defensively), an
   already-parsed object, or null for a legacy book */
const bookSpec = (book) => {
  const raw = book && book.job_spec;
  if (!raw) return null;
  if (typeof raw === "object") return raw;
  try {
    const spec = JSON.parse(raw);
    return spec && typeof spec === "object" ? spec : null;
  } catch (err) {
    return null;
  }
};

const selectPill = (wrap, value) => {
  const pills = [...wrap.querySelectorAll(".pill-btn")];
  const pill = pills.find((b) => b.dataset.value === value);
  if (!pill) return;
  pills.forEach((b) => b.setAttribute("aria-pressed", String(b === pill)));
};

const setSubmitLabel = (text) => {
  const label = $("#conv-submit .btn-label");
  if (label) label.textContent = text;
};

function clearRequeueMode() {
  convertTarget = null;
  $("#convert-form").classList.remove("convert-form--requeue");
  setHidden($("#conv-requeue-note"), true);
  setSubmitLabel("Convert");
}

async function openRequeueForm(book) {
  const spec = bookSpec(book) || {};
  convertTarget = { book, spec };
  const form = $("#convert-form");
  form.classList.add("convert-form--requeue");
  $("#conv-requeue-title").textContent = `Requeue ${book.title}`;
  const quality = QUALITY_ALIAS[spec.quality] ||
    (["low", "medium", "high"].includes(spec.quality) ? spec.quality : "medium");
  if (voiceOptions.length === 0) await renderVoices(); /* the pills need voices */
  buildPills($("#conv-quality"), qualityOptions, quality);
  renderVoicePills();
  if (spec.voice) selectPill($("#conv-voices"), spec.voice);
  $("#conv-gap").value = spec.gap != null && spec.gap !== "" ? spec.gap : "0.2";
  $("#conv-lexicon").value = spec.lexicon_text || "";
  $("#conv-ocr").checked = spec.ocr === true || spec.ocr === "true";
  setSubmitLabel("Requeue");
  setHidden($("#conv-error"), true);
  setHidden($("#conv-requeue-note"), false);
  setHidden(form, false);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  form.scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
  const firstQuality = $("#conv-quality .pill-btn");
  if (firstQuality) firstQuality.focus(); /* the file picker is hidden in this mode */
}

/* requeue submit: no file, JSON params, the book's own endpoint */
async function onRequeueSubmit() {
  const { book, spec } = convertTarget;
  const submit = $("#conv-submit");
  const errorSlot = $("#conv-error");
  submit.classList.add("is-loading");
  submit.setAttribute("aria-busy", "true");
  setHidden(errorSlot, true);
  const body = {};
  const voice = pillValue($("#conv-voices"));
  if (voice) body.voice = voice;
  body.quality = pillValue($("#conv-quality")) || "medium";
  const gap = parseFloat($("#conv-gap").value);
  if (!Number.isNaN(gap)) body.gap = gap;
  const lex = $("#conv-lexicon").value.trim();
  if (lex) body.lexicon_text = lex;
  body.ocr = $("#conv-ocr").checked;
  if (spec && spec.ocr_lang) body.ocr_lang = spec.ocr_lang; /* not editable, carried over */
  let job;
  try {
    job = await requeueBook(book.id, body);
  } catch (err) {
    submit.classList.remove("is-loading");
    submit.removeAttribute("aria-busy");
    errorSlot.textContent = `Requeue could not start: ${err.message}`;
    setHidden(errorSlot, false);
    return;
  }
  submit.classList.remove("is-loading");
  submit.removeAttribute("aria-busy");
  setHidden($("#convert-form"), true);
  clearRequeueMode();
  trackJob(job.id, { count: 1, file: book.title });
  void refreshShelfData();
}

function initConvertForm() {
  const form = $("#convert-form");
  $("#convert-open").addEventListener("click", () => {
    if (convertTarget) clearRequeueMode(); /* a fresh upload replaces the requeue intent */
    setHidden(form, !form.hidden);
    if (!form.hidden) $("#conv-pdf").focus();
  });
  $("#conv-cancel").addEventListener("click", () => {
    setHidden(form, true);
    clearRequeueMode();
  });
  const fileInput = $("#conv-pdf");
  fileInput.addEventListener("change", () => {
    const names = fileInput.files ? [...fileInput.files].map((f) => f.name) : [];
    $("#conv-pdf-name").textContent = names.length ? names.join(", ") : "No files chosen";
  });
  form.addEventListener("submit", onConvertSubmit);
}

async function onConvertSubmit(event) {
  event.preventDefault();
  if (convertTarget) { await onRequeueSubmit(); return; } /* requeue mode: no file */
  const fileInput = $("#conv-pdf");
  const files = fileInput.files ? [...fileInput.files] : [];
  const errorSlot = $("#conv-error");

  fileInput.classList.toggle("is-invalid", files.length === 0);
  setHidden($("#conv-pdf-error"), files.length > 0);
  if (files.length === 0) return;

  const submit = $("#conv-submit");
  submit.classList.add("is-loading");
  submit.setAttribute("aria-busy", "true");
  setHidden(errorSlot, true);

  const body = new FormData();
  body.append("ocr", $("#conv-ocr").checked ? "true" : "false");
  const voice = pillValue($("#conv-voices"));
  if (voice) {
    /* omit when no voice pill is selected; the server default applies.
       Sending a literal "default" makes the CLI reject the job. */
    body.append("voice", voice);
  }
  body.append("quality", pillValue($("#conv-quality")) || "draft");
  body.append("gap", $("#conv-gap").value || "0.2");
  const lex = $("#conv-lexicon").value.trim();
  if (lex) body.append("lexicon_text", lex);

  let job;
  try {
    if (files.length > 1) {
      // batch mode: repeated pdf fields, one aggregate Job for all files
      for (const f of files) body.append("pdf", f);
      job = await api("/convert/batch", { method: "POST", body });
    } else {
      body.append("pdf", files[0]);
      job = await api("/convert", { method: "POST", body });
    }
  } catch (err) {
    submit.classList.remove("is-loading");
    submit.removeAttribute("aria-busy");
    errorSlot.textContent = `Conversion could not start: ${err.message}`;
    setHidden(errorSlot, false);
    return;
  }
  submit.classList.remove("is-loading");
  submit.removeAttribute("aria-busy");
  fileInput.value = "";
  setHidden($("#convert-form"), true); /* not `form`: onConvertSubmit never declares it */
  trackJob(job.id, { count: files.length, file: files.map((f) => f.name).join(", ") });
}

/* Refresh survival: jobs live server-side, but the preparing card was
   page-local, so a reload made in-flight conversions vanish. Re-attach a card
   to every job still worth watching: running and queued ones own the FIFO, a
   paused one rests on its card until its requeue lands as a new job id. */
async function reattachRunningJobs(jobs) {
  if (jobs === undefined) {
    try {
      jobs = await api("/jobs");
    } catch (err) {
      return; /* list unavailable: skip silently, pollers re-attach next load */
    }
  }
  if (!Array.isArray(jobs)) return;
  for (const job of jobs) {
    const st = job && job.status;
    if (st !== "running" && st !== "queued" && st !== "paused") continue;
    const files = Array.isArray(job.files) ? job.files : [];
    const meta = {
      count: files.length || 1,
      file: files.join(", ") || jobTitle(job),
    };
    if (st === "paused") meta.subject = "Your audiobook is paused";
    trackJob(job.id, meta);
  }
}

/* the book name reattached cards show when the job carries no file list */
function jobTitle(job) {
  const ids = Array.isArray(job.book_ids) ? job.book_ids : job.book_id != null ? [job.book_id] : [];
  const book = state.books.find((b) => ids.map(String).includes(String(b.id)));
  return book ? book.title : "";
}

/* One preparing card per job id, forever. If a panel for this job already
   exists in the DOM (reload, back-navigation, a second renderLibrary), it is
   adopted in place instead of spawning a duplicate, and the old interval is
   cleared before a new one is installed so exactly one poller polls per id. */
function trackJob(jobId, meta = {}) {
  if (state.pollers.has(jobId)) return; /* already tracked, panel already live */
  const existing = document.querySelector(
    `#convert-jobs .job[data-job-id="${CSS.escape(String(jobId))}"]`
  );
  if (existing) {
    /* a panel with no live poller is a settled job (done, error, paused or
       cancelled): it stays as history. Polling it again would spin a poller on
       a phase only a requeue can leave, and a requeue is a NEW job id. */
    if (
      !existing.classList.contains("is-done") &&
      !existing.classList.contains("is-error") &&
      !existing.classList.contains("is-paused") &&
      !existing.classList.contains("is-cancelled")
    ) {
      startJobPoller(jobId, existing);
    }
    return;
  }
  const panel = buildJobPanel(jobId, meta);
  $("#convert-jobs").prepend(panel);
  startJobPoller(jobId, panel);
}

function buildJobPanel(jobId, meta) {
  const subject = (meta && meta.subject) ||
    ((meta && meta.count) > 1
      ? `Your ${meta.count} audiobooks are preparing`
      : "Your audiobook is preparing");
  const panel = el("article", "job job--prepare");
  panel.dataset.jobId = String(jobId);
  panel.setAttribute("aria-busy", "true");
  /* counts stay empty until the first fetched payload paints them: seeding
     "0 / 0" would flash stale numbers on re-attach after a reload */
  panel.innerHTML = `
    <div class="job-head">
      <span class="job-status">
        <svg class="icon spin" width="24" height="24" aria-hidden="true"><use href="#i-spinner"/></svg>
        <span class="job-status-text">Queued</span>
      </span>
      <span class="job-cacheline"></span>
    </div>
    <div class="progress"><div class="progress-fill" style="width:0%"></div></div>
    <dl class="job-counts">
      <dt>Pages</dt><dd class="job-pages"></dd>
      <dt>Chunks</dt><dd class="job-chunks"></dd>
      <dt>Cache hits</dt><dd class="job-cache"></dd>
    </dl>
    <p class="job-message" hidden></p>`;
  /* heading carries live file names -> DOM textContent, never innerHTML */
  const heading = el("h3", "job-title", subject);
  if (meta.file) heading.appendChild(el("span", "job-title-files", meta.file));
  panel.insertBefore(heading, $(".progress", panel));
  return panel;
}

function startJobPoller(jobId, panel) {
  const poll = async () => {
    let job;
    try {
      job = await getJob(jobId);
    } catch (err) {
      paintJobError(panel, `Job status unavailable: ${err.message}`);
      stopPoller(jobId);
      return;
    }
    paintJob(panel, job);
    if (job.status === "done") {
      stopPoller(jobId);
      panel.setAttribute("aria-busy", "false");
      const ids = Array.isArray(job.book_ids) && job.book_ids.length
        ? job.book_ids
        : job.book_id ? [job.book_id] : [];
      void (async () => {
        await renderLibrary();
        revealFreshBooks(ids);
        paintJobDone(panel, ids);
      })();
    } else if (job.status === "error") {
      stopPoller(jobId);
      paintJobError(panel, job.message || "Conversion failed.");
    } else if (job.status === "paused" || job.status === "cancelled") {
      /* kill-and-mark outcomes (P9): the FIFO moves on, this panel settles.
         A resume or requeue enqueues a NEW job id and with it a new panel. */
      stopPoller(jobId);
      settleJobPanel(panel, job.status);
      void refreshShelfData();
    }
  };
  const previous = state.pollers.get(jobId);
  if (previous) clearInterval(previous); /* never two intervals per job id */
  state.pollers.set(jobId, setInterval(poll, 1000));
  void poll();
}

/* The one shared shelf refresh: books re-render the tiles, the job list
   re-resolves every book's phase (queued, running, paused) and adoption turns
   any foreign job into a panel, deduped by job id. */
async function refreshShelfData() {
  const [books, jobs] = await Promise.all([
    getBooks(),
    api("/jobs").catch(() => null),
  ]);
  state.books = books;
  state.jobs = Array.isArray(jobs) ? jobs : null;
  renderShelf();
  renderContinue();
  reattachRunningJobs(state.jobs);
}

/* A running job publishes page after page; the shelf tile is the only entry
   point the user has to start listening early, so it must not stay frozen at
   "0 / 12 pages" until the very end. Refresh the shelf (cheap GET /books plus
   GET /jobs) on a throttle while any job is producing pages. */
let shelfRefreshAt = 0;
let shelfRefreshBusy = false;
async function refreshShelfWhilePreparing() {
  if (shelfRefreshBusy) return;
  const now = Date.now();
  if (now - shelfRefreshAt < 2500) return;
  shelfRefreshAt = now;
  shelfRefreshBusy = true;
  try {
    await refreshShelfData();
  } catch (err) {
    /* transient: the next tick retries */
  } finally {
    shelfRefreshBusy = false;
  }
}

function paintJob(panel, job) {
  const statusText = $(".job-status-text", panel);
  statusText.textContent = job.status === "running" ? "preparing" : job.status;
  const pct = job.page_count ? (job.page_done / job.page_count) * 100 : 0;
  $(".progress-fill", panel).style.width = `${Math.min(100, pct)}%`;
  $(".job-pages", panel).textContent = `${job.page_done} / ${job.page_count}`;
  $(".job-chunks", panel).textContent = `${job.chunk_done} / ${job.chunk_total}`;
  $(".job-cache", panel).textContent = String(job.cache_hits);
  const msg = $(".job-message", panel);
  if (job.message) { setHidden(msg, false); msg.textContent = job.message; }
  if (job.status === "running" && job.page_done > 0) {
    void refreshShelfWhilePreparing();
  }
}

/* the book landed: collapse the prepare card into a compact done state */
function paintJobDone(panel, bookIds) {
  panel.classList.add("is-done");
  const spinner = $(".job-status .icon.spin", panel);
  if (spinner) {
    const check = icon("check");
    spinner.replaceWith(check);
  }
  $(".job-status-text", panel).textContent = "Ready";
  const fresh = bookIds
    .map((id) => state.books.find((b) => b.id === id))
    .filter((b) => b);
  const links = $(".job-cacheline", panel);
  links.replaceChildren();
  for (const book of fresh) {
    const a = el("a", "link", `Open ${book.title}`);
    a.href = `#/book/${book.id}`;
    links.appendChild(a);
  }
}

/* a paused or cancelled panel settles in place: the spinner gives way to the
   phase glyph, aria-busy clears, the numbers stay for reference */
function settleJobPanel(panel, status) {
  panel.setAttribute("aria-busy", "false");
  panel.classList.add(status === "paused" ? "is-paused" : "is-cancelled");
  $(".job-status-text", panel).textContent = status === "paused" ? "Paused" : "Cancelled";
  const spinner = $(".job-status .icon.spin", panel);
  if (spinner) spinner.replaceWith(icon(status === "paused" ? "pause" : "x"));
}

/* scroll to and outline the books that just finished converting */
function revealFreshBooks(bookIds) {
  let first = null;
  for (const id of bookIds) {
    const tile = document.querySelector(`#shelf .tile[data-book-id="${id}"]`);
    if (!tile) continue;
    tile.classList.add("tile--fresh");
    first = first || tile;
  }
  if (first) {
    const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
    first.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
  }
}

function paintJobError(panel, message) {
  panel.classList.add("is-error");
  panel.setAttribute("aria-busy", "false");
  const statusText = $(".job-status-text", panel);
  statusText.textContent = "Error";
  const alert = el("p", "form-error");
  const alertIcon = icon("alert");
  alert.appendChild(alertIcon);
  alert.appendChild(el("span", null, message));
  const old = $(".form-error", panel);
  if (old) old.replaceWith(alert); else panel.appendChild(alert);
}

function stopPoller(jobId) {
  const timer = state.pollers.get(jobId);
  if (timer) { clearInterval(timer); state.pollers.delete(jobId); }
}

/* ---------- player view ---------- */

const audio = () => $("#audio");
const player = {
  book: null,
  bookmarks: [],
  progressTimer: null,
  sleepTimer: null,
  sleepEndsAt: 0,
  speed: 1,
  live: null, /* { pages, offsets, duration, index, status, totalPages } while playing a preparing book */
  pendingSeek: null, /* seconds inside the just-loaded page/track */
  autoAdvance: false, /* start playing as soon as the next track's metadata lands */
};

/* increments on every open/close; async player work captures the value and
   bails if the view moved on, so a pending openPlayer can never touch a
   player that closePlayer already tore down (the "null duration_s" crash) */
let viewSeq = 0;

async function openPlayer(bookId) {
  const seq = ++viewSeq;
  showView("player");
  $("#player-error-text").textContent = "This book is not available.";
  setHidden($("#player-error"), true);
  setHidden($("#player-body"), false);
  let book;
  try {
    book = await getBook(bookId);
  } catch (err) {
    if (seq !== viewSeq) return;
    setHidden($("#player-error"), false);
    setHidden($("#player-body"), true);
    return;
  }
  if (seq !== viewSeq) return;
  player.book = book;
  await loadPlayer(seq);
}

async function loadPlayer(seq = viewSeq) {
  if (seq !== viewSeq) return;
  const book = player.book;
  if (!book) return;
  $("#pl-title").textContent = book.title;
  const cover = $("#pl-cover");
  cover.src = coverUrl(book.id);
  cover.alt = `Cover of ${book.title}`;
  cover.addEventListener("error", () => { setHidden(cover, true); }, { once: true });

  let progress = book.progress;
  if (progress === undefined) {
    try { progress = await getProgress(book.id); } catch (err) { progress = null; }
    if (seq !== viewSeq) return;
  }
  player.resumeAt = progress ? progress.position_s : 0;
  player.page = progress && progress.page ? progress.page : 1;

  try {
    player.bookmarks = Array.isArray(book.bookmarks) ? book.bookmarks : await getBookmarks(book.id);
  } catch (err) {
    player.bookmarks = [];
  }
  if (seq !== viewSeq) return;
  renderBookmarks();

  const a = audio();
  a.playbackRate = player.speed;

  const seek = $("#pl-seek");

  /* live mode: the book is still preparing (or only partly published), so
     playback rides the per-page WAVs and grows as pages land */
  const readyPages = Number(book.ready_pages != null ? book.ready_pages : 0);
  const needsLive = book.status !== "ready" || readyPages < Number(book.pages || 0);
  if (needsLive) {
    let data = null;
    try {
      data = await api(`/books/${book.id}/pages`);
    } catch (err) {
      data = null;
    }
    if (seq !== viewSeq) return;
    if (data && Array.isArray(data.pages) && data.pages.length > 0) {
      setupLive(book, data, player.resumeAt);
      await initReadingPane(book.id, seq, { live: true });
      if (seq !== viewSeq) return;
      setHidden($("#player-error"), true);
      setHidden($("#player-body"), false);
      if (player.live.status === "preparing") {
        startLivePolling();
      }
      updateLiveNote();
      void loadPdfDocument(book.id, seq);
      return;
    }
    /* nothing published yet (or the pages route is unreachable): the honest
       empty state, no crash, no autoplay of a dead src */
    if (book.status !== "ready") {
      stopLivePolling();
      player.live = null;
      showPlayerEmpty();
      return;
    }
  }

  /* finished book: one m4b, as before */
  player.live = null;
  setHidden($("#pl-live-note"), true);
  a.src = audioUrl(book.id);

  seek.disabled = true;
  seek.max = Math.max(1, Math.round(book.duration_s || 0));

  setChapterLabel();
  void loadPdfDocument(book.id, seq);
  await initReadingPane(book.id, seq);
  if (seq !== viewSeq) return;
}

/* the book exists but nothing is playable yet */
function showPlayerEmpty() {
  const a = audio();
  a.pause();
  a.removeAttribute("src");
  a.load();
  $("#player-error-text").textContent = "This book has no pages ready yet.";
  setHidden($("#player-error"), false);
  setHidden($("#player-body"), true);
}

/* Build the live playlist from GET /pages. offsets[i] is the global start of
   page i; duration is the playable total so far. The resume position picks
   the page, then becomes page-local for onLoadedMetadata. */
function setupLive(book, data, resumeAt) {
  const pages = data.pages.map((p) => ({ ...p }));
  const offsets = [];
  let acc = 0;
  for (const p of pages) {
    offsets.push(acc);
    acc += Number(p.duration_s || 0);
  }
  player.live = {
    pages,
    offsets,
    duration: Number(data.duration_ready_s) || acc,
    index: 0,
    status: data.status || book.status || "preparing",
    totalPages: Number(data.total_pages) || Number(book.pages) || 0,
  };
  const target = Math.min(Math.max(0, resumeAt || 0), player.live.duration);
  let idx = 0;
  for (let i = 0; i < offsets.length; i++) {
    if (offsets[i] <= target + 0.001) idx = i;
    else break;
  }
  player.live.index = idx;
  player.resumeAt = target - offsets[idx];
  const a = audio();
  a.src = pages[idx].url;
  const seek = $("#pl-seek");
  seek.disabled = true;
  seek.max = Math.max(1, Math.round(player.live.duration || 1));
  setLivePageSentences();
  setChapterLabel();
}

/* global playback position: the one number progress, bookmarks and the seek
   bar use, identical in unit to a position inside the finished m4b */
function globalPosition() {
  const live = player.live;
  if (!live || !live.pages.length) return audio().currentTime || 0;
  return (live.offsets[live.index] || 0) + (audio().currentTime || 0);
}

function currentPageNo() {
  const live = player.live;
  if (live && live.pages[live.index]) return live.pages[live.index].page;
  return computePage(globalPosition());
}

function setChapterLabel() {
  const book = player.book;
  if (!book) return;
  const live = player.live;
  if (live && live.pages.length) {
    const page = live.pages[Math.min(live.index, live.pages.length - 1)].page;
    const total = live.totalPages || book.pages || page;
    $("#pl-chapter").textContent = `Page ${page} of ${total} · ${live.pages.length} ready`;
    return;
  }
  const pages = Math.max(1, book.pages || 1);
  const page = Math.min(Math.max(1, player.page || 1), pages);
  $("#pl-chapter").textContent = `Page ${page} of ${pages}`;
}

/* the live status line: what is playable right now, never a spinner */
function updateLiveNote() {
  const note = $("#pl-live-note");
  const live = player.live;
  if (!note) return;
  if (!live || !live.pages.length) {
    setHidden(note, true);
    return;
  }
  const total = live.totalPages || 0;
  const ready = live.pages.length;
  setHidden(note, false);
  if (live.status === "ready" || (total && ready >= total)) {
    note.textContent = "Ready · full book";
  } else if (live.status === "error") {
    note.textContent = `Interrupted · pages 1–${ready} playable`;
  } else if (live.status === "paused") {
    note.textContent = `Paused · pages 1–${ready} playable, resume from the library`;
  } else if (live.status === "cancelled") {
    note.textContent = `Cancelled · pages 1–${ready} playable, requeue from the library`;
  } else if (live.index >= ready - 1) {
    note.textContent = `Page ${ready + 1} is still preparing…`;
  } else {
    note.textContent = `Still preparing · pages 1–${ready} playable`;
  }
}

/* poll for newly published pages while the book is preparing; the playlist
   only ever grows (never rewinds the current page) */
let livePollTimer = null;
let livePollBusy = false;

function startLivePolling() {
  if (livePollTimer) return;
  livePollTimer = setInterval(() => { void pollLivePages(); }, 5000);
}

function stopLivePolling() {
  if (livePollTimer) clearInterval(livePollTimer);
  livePollTimer = null;
}

async function pollLivePages() {
  const book = player.book;
  const live = player.live;
  if (!book || !live || livePollBusy) return;
  livePollBusy = true;
  let data = null;
  try {
    data = await api(`/books/${book.id}/pages`);
  } catch (err) {
    data = null; /* transient: try again on the next tick */
  }
  livePollBusy = false;
  if (!player.book || player.live !== live) return;
  if (data && Array.isArray(data.pages)) {
    for (const p of data.pages.slice(live.pages.length)) {
      live.offsets.push(live.duration);
      live.duration += Number(p.duration_s || 0);
      live.pages.push(p);
    }
    live.status = data.status || live.status;
    if (data.total_pages) live.totalPages = Number(data.total_pages);
    const seek = $("#pl-seek");
    if (!seek.disabled) seek.max = Math.max(1, Math.round(live.duration || 1));
    setChapterLabel();
    updateLiveNote();
    followPdfPage(); /* a newly available page may be the one being played */
  }
  if (
    player.live === live &&
    (live.status === "ready" || (live.totalPages && live.pages.length >= live.totalPages))
  ) {
    stopLivePolling();
    await switchToFinalBook();
  }
}

/* the m4b landed: swap the page playlist for the finished book at the same
   global position (pages became chapters with near-identical durations) */
let switchingToFinal = false;
async function switchToFinalBook() {
  if (switchingToFinal) return;
  switchingToFinal = true;
  try {
    await switchToFinalBookInner();
  } finally {
    switchingToFinal = false;
  }
}

async function switchToFinalBookInner() {
  const book = player.book;
  if (!book) return;
  const live = player.live;
  const pos = live ? globalPosition() : 0;
  const wasPlaying = !audio().paused;
  let fresh = null;
  try {
    fresh = await getBook(book.id);
  } catch (err) {
    fresh = null;
  }
  if (!player.book || player.book.id !== book.id) return;
  if (fresh) player.book = fresh;
  /* Never swap to a final file that is not actually published. When finalize
     fails (Windows lock on a page WAV, aborted encode) the m4b route 404s and
     the player dies mid-sentence — keep playing the pages we already have. */
  if (!fresh || fresh.status !== "ready") {
    if (fresh && fresh.status === "error") {
      stopLivePolling();
      if (player.live) player.live.status = "error";
      updateLiveNote();
    }
    return; /* still finalizing: the poller retries */
  }
  player.live = null;
  const a = audio();
  player.resumeAt = Math.max(0, pos);
  player.autoAdvance = wasPlaying;
  a.src = audioUrl(player.book.id);
  /* the page WAVs the element was riding just got deleted by finalize; without
     an explicit load() the element stays in its failed state and the fresh m4b
     src never resolves ("no supported sources", playback dead on arrival) */
  a.load();
  a.playbackRate = player.speed;
  setHidden($("#pl-live-note"), true);
  setChapterLabel();
  followPdfPage();
  void initReadingPane(player.book.id, viewSeq);
}

/* map a global timestamp onto the page playlist: {index, local} */
function seekToGlobal(t) {
  const live = player.live;
  if (!live || !live.pages.length) return;
  const target = Math.min(Math.max(0, t), live.duration || 0);
  let idx = 0;
  for (let i = 0; i < live.offsets.length; i++) {
    if (live.offsets[i] <= target + 0.001) idx = i;
    else break;
  }
  const a = audio();
  const local = target - live.offsets[idx];
  if (live.index !== idx) {
    live.index = idx;
    player.autoAdvance = !a.paused; /* keep the transport state across the swap */
    player.resumeAt = 0;
    player.pendingSeek = local;
    a.src = live.pages[idx].url;
    a.load();
    setLivePageSentences();
    setChapterLabel();
    followPdfPage();
  } else {
    a.currentTime = local;
  }
}

function showView(name) {
  setHidden($("#view-library"), name !== "library");
  setHidden($("#view-player"), name !== "player");
}

/* ---------- caption mode: only the sentence being read is on screen ---------- */

/* captions for the current page in live mode: the page's own sentences, with
   page-local times, so audio.currentTime maps directly. No sentences yet ->
   the pane stays hidden. */
function setLivePageSentences() {
  const live = player.live;
  const page = live && live.pages[live.index];
  const sentences = page && Array.isArray(page.sentences) ? page.sentences : [];
  state.sentences = sentences;
  state.activeIdx = -1;
  const pane = $("#reading-pane");
  const caption = $("#caption");
  if (!sentences.length) {
    if (caption) caption.textContent = "";
    setHidden(pane, true);
    return;
  }
  setHidden(pane, false);
  updateCaption(true);
}

async function initReadingPane(bookId, seq, opts = {}) {
  const pane = $("#reading-pane");
  const caption = $("#caption");
  state.sentences = [];
  state.activeIdx = -1;
  caption.textContent = "";

  /* live mode gets its captions per page from the /pages payload; the
     whole-book manifest belongs to the finished book */
  if (opts.live) {
    setLivePageSentences();
    return;
  }

  let sentences;
  try {
    sentences = await api(`/books/${bookId}/sentences`);
  } catch (err) {
    /* 404 "no manifest" (or any failure): the feature stays out of the way */
    if (seq !== undefined && seq !== viewSeq) return;
    setHidden(pane, true);
    return;
  }
  if (seq !== undefined && seq !== viewSeq) return;
  if (!Array.isArray(sentences) || sentences.length === 0) {
    setHidden(pane, true);
    return;
  }
  state.sentences = sentences;
  setHidden(pane, false);
  updateCaption(true);
}

/* index within state.sentences holding start_s <= t < end_s, else -1 */
function sentenceAt(t) {
  let lo = 0;
  let hi = state.sentences.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const s = state.sentences[mid];
    if (t < s.start_s) hi = mid - 1;
    else if (t >= s.end_s) lo = mid + 1;
    else return mid;
  }
  return -1;
}

function updateCaption(force) {
  const caption = $("#caption");
  if (!caption || state.sentences.length === 0) return;
  const idx = sentenceAt(audio().currentTime);
  if (idx === state.activeIdx && !force) return;
  state.activeIdx = idx;
  const s = idx >= 0 ? state.sentences[idx] : null;
  setHidden(caption, !s);
  caption.textContent = s ? (s.text || "") : "";
}

function clearReadingPane() {
  state.activeIdx = -1;
  const caption = $("#caption");
  if (caption) { setHidden(caption, true); caption.textContent = ""; }
}

/* ---------- source PDF pane (P8) ---------- */

/* The reader pane beside the player: the original PDF page, following the
   audio. One document per open book, one render at a time; stale renders are
   dropped via pdfPane.seq and the viewSeq generation token. */

const pdfPane = {
  doc: null, /* PDFDocumentProxy for the open book */
  bookId: null, /* book the document belongs to */
  renderTask: null, /* in-flight page render task */
  seq: 0, /* bumps per render; stale results never paint */
  page: 0, /* 1-based PDF page rendered or being rendered */
};

function pdfjsAvailable() {
  return typeof window.pdfjsLib !== "undefined";
}

function setPdfLoading(on) {
  setHidden($("#pdf-spinner"), !on);
}

function setPdfLabel(page, total) {
  $("#pdf-page-label").textContent =
    page && total ? `Page ${page} of ${total}` : "Page — of —";
}

function pdfShowError() {
  setPdfLoading(false);
  setPdfLabel(0, 0);
  setHidden($("#pdf-canvas"), true);
  setHidden($("#pdf-error"), false);
}

function pdfResetPane() {
  setPdfLoading(true);
  setPdfLabel(0, 0);
  setHidden($("#pdf-canvas"), true);
  setHidden($("#pdf-error"), false);
}

/* the 1-based PDF page the audio is on right now, per mode */
function currentPdfPage() {
  const live = player.live;
  if (live && live.pages[live.index]) return live.pages[live.index].page;
  return computePage(audio().currentTime);
}

/* cheap enough for every timeupdate: one number compare when nothing changed */
function followPdfPage() {
  if (!pdfPane.doc) return;
  const page = currentPdfPage();
  if (!page || page === pdfPane.page) return;
  void renderPdfPage(page, viewSeq);
}

function isMissingPdf(err) {
  if (!err) return false;
  if (err.name === "MissingPDFException") return true;
  if (err.name === "UnexpectedResponseException" && err.status === 404) return true;
  return false;
}

/* the canvas is fitted to the pane at render time, so a column-width change
   (window resize, or crossing the 64rem breakpoint) must re-fit the page --
   otherwise the old wide canvas sits clipped inside a narrower pane */
let pdfResizeObserver = null;

function watchPdfWidth() {
  if (pdfResizeObserver || typeof ResizeObserver === "undefined") return;
  const body = $("#pdf-pane-body");
  if (!body) return;
  let last = body.clientWidth;
  pdfResizeObserver = new ResizeObserver(() => {
    const width = body.clientWidth;
    if (!width || Math.abs(width - last) < 2) return;
    last = width;
    if (!pdfPane.doc || !pdfPane.page) return;
    clearTimeout(pdfResizeObserver.timer);
    pdfResizeObserver.timer = setTimeout(() => {
      if (pdfPane.doc) void renderPdfPage(pdfPane.page, viewSeq);
    }, 120);
  });
  pdfResizeObserver.observe(body);
}

async function loadPdfDocument(bookId, seq) {
  if (!pdfjsAvailable()) {
    /* no pdf.js at all: same as a book without a PDF, no pane */
    setHidden($("#pdf-pane"), true);
    return;
  }
  if (pdfPane.bookId === bookId && pdfPane.doc) {
    setHidden($("#pdf-pane"), false);
    followPdfPage();
    return;
  }
  pdfResetPane();
  setHidden($("#pdf-pane"), false);
  watchPdfWidth();
  pdfjsLib.GlobalWorkerOptions.workerSrc = "./vendor/pdfjs/pdf.worker.min.js";
  try {
    const doc = await pdfjsLib.getDocument({
      url: `/api/books/${bookId}/pdf`,
      rangeChunkSize: 65536,
    }).promise;
    if (seq !== viewSeq) {
      doc.destroy(); /* the view moved on; this document is never shown */
      return;
    }
    if (pdfPane.doc) pdfPane.doc.destroy();
    pdfPane.doc = doc;
    pdfPane.bookId = bookId;
    pdfPane.page = 0;
    followPdfPage();
  } catch (err) {
    if (seq !== viewSeq) return;
    if (isMissingPdf(err)) {
      /* no PDF for this book: no pane at all, no empty box */
      pdfPane.bookId = null;
      setHidden($("#pdf-pane"), true);
      return;
    }
    pdfShowError();
  }
}

function pdfTotalPages(doc) {
  return (player.live && player.live.totalPages)
    || (player.book && player.book.pages)
    || doc.numPages;
}

async function renderPdfPage(pageNumber, seq) {
  const doc = pdfPane.doc;
  if (!doc) return;
  const target = Math.min(Math.max(1, pageNumber), doc.numPages);
  pdfPane.page = target;
  pdfPane.seq += 1;
  const mySeq = pdfPane.seq;
  setPdfLabel(target, pdfTotalPages(doc));
  setPdfLoading(true);
  setHidden($("#pdf-error"), true);
  try {
    const page = await doc.getPage(target);
    if (seq !== viewSeq || mySeq !== pdfPane.seq) return;
    const body = $("#pdf-pane-body");
    const canvas = $("#pdf-canvas");
    const style = getComputedStyle(body);
    const cssWidth = Math.max(
      1,
      body.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight)
    );
    const dpr = window.devicePixelRatio || 1;
    const base = page.getViewport({ scale: 1 });
    const viewport = page.getViewport({ scale: (cssWidth / base.width) * dpr });
    canvas.width = Math.floor(viewport.width);
    canvas.height = Math.floor(viewport.height);
    canvas.style.width = `${Math.floor(viewport.width / dpr)}px`;
    canvas.style.height = `${Math.floor(viewport.height / dpr)}px`;
    if (pdfPane.renderTask) {
      try { pdfPane.renderTask.cancel(); } catch (_) { /* already settled */ }
    }
    const task = page.render({
      canvasContext: canvas.getContext("2d"),
      viewport,
    });
    pdfPane.renderTask = task;
    await task.promise;
    if (seq !== viewSeq || mySeq !== pdfPane.seq) return;
    pdfPane.renderTask = null;
    setPdfLoading(false);
    setHidden(canvas, false);
  } catch (err) {
    if (err && err.name === "RenderingCancelledException") return;
    if (seq !== viewSeq || mySeq !== pdfPane.seq) return;
    pdfPane.renderTask = null;
    pdfShowError();
  }
}

/* closing the player: cancel pending work, destroy the document; a view opened
   after this close is never touched (viewSeq discipline in the callers) */
function destroyPdfPane() {
  pdfPane.seq += 1;
  if (pdfPane.renderTask) {
    try { pdfPane.renderTask.cancel(); } catch (_) { /* already settled */ }
    pdfPane.renderTask = null;
  }
  if (pdfPane.doc) {
    pdfPane.doc.destroy();
    pdfPane.doc = null;
  }
  pdfPane.bookId = null;
  pdfPane.page = 0;
  const canvas = $("#pdf-canvas");
  if (canvas) {
    canvas.width = 0;
    canvas.height = 0;
    canvas.style.width = "";
    canvas.style.height = "";
    setHidden(canvas, true);
  }
  setHidden($("#pdf-error"), true);
  setHidden($("#pdf-spinner"), true);
  setPdfLabel(0, 0);
  setHidden($("#pdf-pane"), true);
}

/* ---------- player: transport, seek, speed, sleep, bookmarks ---------- */

function computePage(pos) {
  const book = player.book;
  if (!book || !book.pages || !book.duration_s) return player.page || 1;
  return Math.min(book.pages, Math.max(1, Math.floor((pos / book.duration_s) * book.pages) + 1));
}

function onLoadedMetadata() {
  const a = audio();
  const book = player.book;
  if (!book) return; /* closed while the metadata was loading */
  const seek = $("#pl-seek");
  const live = player.live;
  if (live && live.pages.length) {
    /* the seek bar spans the whole playable-so-far timeline, not one page */
    seek.max = Math.max(1, Math.round(live.duration || 1));
  } else {
    seek.max = Math.round(a.duration || book.duration_s || 1);
  }
  if (player.resumeAt > 0) a.currentTime = player.resumeAt;
  player.resumeAt = 0;
  if (player.pendingSeek != null) {
    a.currentTime = player.pendingSeek;
    player.pendingSeek = null;
  }
  seek.disabled = false;
  updateSeek();
  updateCaption(true);
  followPdfPage();
  if (player.autoAdvance) {
    player.autoAdvance = false;
    void a.play().catch(() => { /* autoplay refusal: the button stays ready */ });
  }
}

function updateSeek() {
  const a = audio();
  const book = player.book;
  if (!book) return; /* closed mid-update */
  const seek = $("#pl-seek");
  const live = player.live;
  if (live && live.pages.length) {
    const pos = globalPosition();
    if (!document.activeElement || document.activeElement !== seek) {
      seek.value = Math.round(pos);
    }
    const pct = live.duration ? Math.round((pos / live.duration) * 100) : 0;
    seek.style.setProperty("--seek-pos", `${pct}%`);
    $("#pl-time-cur").textContent = fmtTime(pos);
    $("#pl-time-total").textContent = fmtTime(live.duration);
    setChapterLabel();
    return;
  }
  if (!document.activeElement || document.activeElement !== seek) {
    seek.value = Math.round(a.currentTime);
  }
  const pos = Math.round((a.currentTime / (a.duration || 1)) * 100);
  seek.style.setProperty("--seek-pos", `${pos}%`);
  $("#pl-time-cur").textContent = fmtTime(a.currentTime);
  const total = a.duration || book.duration_s || 0;
  $("#pl-time-total").textContent = fmtTime(total);
  player.page = computePage(a.currentTime);
  setChapterLabel();
}

let lastPut = 0;
function maybePutProgress(force) {
  const a = audio();
  if (!player.book || !a.src) return;
  /* progress is always the GLOBAL position, so it survives the live -> m4b
     switch; in live mode a.duration is one page, never the book */
  if (!player.live && !a.duration) return;
  const now = Date.now();
  if (!force && now - lastPut < 5000) return;
  lastPut = now;
  const progress = {
    position_s: Math.round(globalPosition()),
    page: currentPageNo(),
    updated_at: new Date().toISOString(),
  };
  player.page = progress.page;
  void putProgress(player.book.id, progress).catch(() => { /* offline PUTs retry next tick */ });
}

async function togglePlay() {
  const a = audio();
  const btn = $("#pl-play");
  if (a.paused) {
    try { await a.play(); } catch (err) {
      const msg = $("#player-error-text");
      msg.textContent = "Playback failed. Try reloading the page.";
      setHidden($("#player-error"), false);
    }
  } else {
    a.pause();
  }
  btn.setAttribute("aria-pressed", a.paused ? "false" : "true");
}

function skip(delta) {
  const a = audio();
  a.currentTime = Math.min(Math.max(0, a.currentTime + delta), a.duration || a.currentTime);
}

function stepChapter(delta) {
  const book = player.book;
  const a = audio();
  if (!book) return;
  const live = player.live;
  if (live && live.pages.length) {
    /* page steps ride the real playlist: previous/next published page */
    const target = Math.min(Math.max(0, live.index + delta), live.pages.length - 1);
    seekToGlobal((live.offsets[target] || 0) + 0.05);
    return;
  }
  if (!book.duration_s || !book.pages) return;
  // page chapters are evenly distributed over the runtime
  const pageDur = book.duration_s / book.pages;
  const page = computePage(a.currentTime);
  const targetPage = Math.min(Math.max(1, page + delta), book.pages);
  a.currentTime = (targetPage - 1) * pageDur + 0.05;
  player.page = targetPage;
  setChapterLabel();
  followPdfPage();
}

function setSpeed(v) {
  player.speed = Math.min(3, Math.max(0.5, Math.round(v * 100) / 100));
  audio().playbackRate = player.speed;
  $("#speed-value").textContent = `${player.speed.toFixed(2)}x`;
  $("#speed-down").disabled = player.speed <= 0.5;
  $("#speed-up").disabled = player.speed >= 3;
}

function armSleep(minutes) {
  clearSleep();
  document.querySelectorAll("#sleep-options .seg-btn").forEach((b) => {
    b.setAttribute("aria-pressed", String(Number(b.dataset.minutes) === minutes));
  });
  if (!minutes) return;
  player.sleepEndsAt = Date.now() + minutes * 60000;
  setHidden($("#sleep-countdown"), false);
  player.sleepTimer = setInterval(sleepTick, 1000);
  sleepTick();
}

function sleepTick() {
  const remainMs = player.sleepEndsAt - Date.now();
  if (remainMs <= 0) { clearSleep(); audio().pause(); return; }
  // fade over the final 10 seconds of the timer, then pause
  if (remainMs < 10000) {
    audio().volume = Math.max(0, remainMs / 10000);
  }
  const min = Math.floor(remainMs / 60000);
  const sec = Math.floor((remainMs % 60000) / 1000);
  $("#sleep-countdown").textContent = `Sleep in ${min}:${String(sec).padStart(2, "0")}`;
}

function clearSleep() {
  if (player.sleepTimer) clearInterval(player.sleepTimer);
  player.sleepTimer = null;
  player.sleepEndsAt = 0;
  audio().volume = 1;
  setHidden($("#sleep-countdown"), true);
}

function renderBookmarks() {
  const list = $("#bm-list");
  list.replaceChildren();
  const sorted = [...player.bookmarks].sort((a, b) => a.position_s - b.position_s);
  for (const bm of sorted) {
    const li = el("li", "bm-item");

    const jump = el("button", "bm-jump");
    jump.type = "button";
    jump.appendChild(el("span", "bm-label", bm.label || "Bookmark"));
    jump.appendChild(el("span", "bm-pos", `${fmtTime(bm.position_s)}, page ${bm.page || 1}`));
    jump.setAttribute("aria-label", `Seek to ${bm.label || "bookmark"} at ${fmtTime(bm.position_s)}`);
    jump.addEventListener("click", () => {
      audio().currentTime = bm.position_s;
      updateSeek();
    });

    const del = el("button", "icon-btn btn--danger");
    del.type = "button";
    del.setAttribute("aria-label", `Delete bookmark ${bm.label || "at " + fmtTime(bm.position_s)}`);
    del.appendChild(icon("trash"));
    del.addEventListener("click", async () => {
      try {
        await deleteBookmark(bm.id);
        player.bookmarks = player.bookmarks.filter((b) => b.id !== bm.id);
        renderBookmarks();
      } catch (err) { del.classList.add("is-invalid"); }
    });

    li.append(jump, del);
    list.appendChild(li);
  }
  if (sorted.length === 0) {
    list.appendChild(el("li", "empty-copy", "No bookmarks yet."));
  }
}

function initBookmarkForm() {
  const form = $("#bm-form");
  $("#bm-add").addEventListener("click", () => {
    setHidden(form, !form.hidden);
    if (!form.hidden) $("#bm-label").focus();
  });
  $("#bm-cancel").addEventListener("click", () => { setHidden(form, true); });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const label = $("#bm-label").value.trim();
    const input = $("#bm-label");
    if (!label) {
      input.classList.add("is-invalid");
      setHidden($("#bm-error"), false);
      return;
    }
    input.classList.remove("is-invalid");
    setHidden($("#bm-error"), true);
    const a = audio();
    const btn = $("#bm-save");
    btn.classList.add("is-loading");
    btn.setAttribute("aria-busy", "true");
    try {
      const bm = await addBookmark(player.book.id, {
        position_s: Math.round(globalPosition()),
        page: currentPageNo(),
        label,
      });
      player.bookmarks.push(bm);
      $("#bm-label").value = "";
      setHidden(form, true);
      renderBookmarks();
    } catch (err) {
      $("#bm-error").textContent = `Bookmark not saved: ${err.message}`;
      setHidden($("#bm-error"), false);
    } finally {
      btn.classList.remove("is-loading");
      btn.removeAttribute("aria-busy");
    }
  });
}

function initPlayerControls() {
  const a = audio();
  a.addEventListener("loadedmetadata", onLoadedMetadata);
  a.addEventListener("timeupdate", () => updateCaption());
  a.addEventListener("seeked", () => { updateCaption(); followPdfPage(); });
  a.addEventListener("playing", () => updateCaption());
  a.addEventListener("timeupdate", () => { updateSeek(); followPdfPage(); maybePutProgress(false); });
  a.addEventListener("pause", () => {
    $("#pl-play").setAttribute("aria-pressed", "false");
    maybePutProgress(true);
  });
  a.addEventListener("play", () => {
    $("#pl-play").setAttribute("aria-pressed", "true");
    /* a retry that works must clear the previous failure banner; it used to
       stay pinned over a perfectly playing book */
    setHidden($("#player-error"), true);
  });
  a.addEventListener("ended", () => { onTrackEnded(); });

  /* transport buffering: only a genuine stall shows the spinner, and only
     after 250 ms so a micro-stall never flickers. The play button keeps full
     contrast while it shows (data-buffering swaps the icons, dims nothing). */
  let bufferTimer = 0;
  const buffer = $("#pl-buffer");
  const playBtn = $("#pl-play");
  const showBuffer = () => {
    if (bufferTimer) return;
    bufferTimer = setTimeout(() => {
      bufferTimer = 0;
      if (!audio().paused) {
        setHidden(buffer, false);
        playBtn.setAttribute("data-buffering", "true");
      }
    }, 250);
  };
  const hideBuffer = () => {
    if (bufferTimer) { clearTimeout(bufferTimer); bufferTimer = 0; }
    setHidden(buffer, true);
    playBtn.removeAttribute("data-buffering");
  };
  a.addEventListener("waiting", () => { if (!a.paused) showBuffer(); });
  for (const ev of ["playing", "canplay", "seeked", "pause", "ended", "error"]) {
    a.addEventListener(ev, hideBuffer);
  }
  a.addEventListener("error", () => {
    if (!a.src) return;
    setHidden($("#player-error"), false);
  });

  $("#pl-play").addEventListener("click", togglePlay);
  $("#pl-skip-back").addEventListener("click", () => skip(-15));
  $("#pl-skip-fwd").addEventListener("click", () => skip(15));
  $("#pl-prev-chapter").addEventListener("click", () => stepChapter(-1));
  $("#pl-next-chapter").addEventListener("click", () => stepChapter(1));

  const seek = $("#pl-seek");
  seek.addEventListener("input", () => {
    const frac = Number(seek.value) / Math.max(1, seek.max);
    const live = player.live;
    if (live && live.pages.length) {
      $("#pl-time-cur").textContent = fmtTime(frac * (live.duration || 0));
      return;
    }
    const dur = a.duration || player.book?.duration_s || 0;
    $("#pl-time-cur").textContent = fmtTime(frac * dur);
  });
  seek.addEventListener("change", () => {
    const frac = Number(seek.value) / Math.max(1, seek.max);
    const live = player.live;
    if (live && live.pages.length) {
      seekToGlobal(frac * (live.duration || 0));
      return;
    }
    const dur = a.duration || player.book?.duration_s || 0;
    a.currentTime = frac * dur;
  });

  $("#speed-down").addEventListener("click", () => setSpeed(player.speed - 0.25));
  $("#speed-up").addEventListener("click", () => setSpeed(player.speed + 0.25));
  $("#speed-down").disabled = true;
  setSpeed(1);

  $("#sleep-options").addEventListener("click", (event) => {
    const btn = event.target.closest(".seg-btn");
    if (btn) armSleep(Number(btn.dataset.minutes));
  });

  initBookmarkForm();

  window.addEventListener("beforeunload", () => maybePutProgress(true));
}

/* a page (or the m4b) finished: in live mode walk to the next published page;
   at the end of what is published so far, just stop and say why */
function onTrackEnded() {
  const live = player.live;
  if (!live || !live.pages.length) return; /* finished m4b: nothing to chain */
  if (live.index < live.pages.length - 1) {
    live.index += 1;
    player.resumeAt = 0;
    player.autoAdvance = true;
    const a = audio();
    a.src = live.pages[live.index].url;
    setLivePageSentences();
    setChapterLabel();
    followPdfPage();
  } else if (live.status === "ready" || (live.totalPages && live.pages.length >= live.totalPages)) {
    void switchToFinalBook(); /* the swap has not landed yet: do it now */
  } else {
    updateLiveNote(); /* end of what is published so far */
  }
}

async function closePlayer() {
  viewSeq += 1; /* any in-flight openPlayer/loadPlayer now bails */
  stopLivePolling();
  const a = audio();
  maybePutProgress(true);
  a.pause();
  a.removeAttribute("src");
  a.load();
  clearSleep();
  clearReadingPane();
  destroyPdfPane();
  if (player.progressTimer) clearInterval(player.progressTimer);
  player.live = null;
  player.pendingSeek = null;
  player.autoAdvance = false;
  player.resumeAt = 0;
  player.book = null;
}

/* ---------- router ---------- */

function route() {
  const hash = location.hash || "#/";
  const match = hash.match(/^#\/book\/([^/]+)$/);
  if (match) {
    void openPlayer(decodeURIComponent(match[1]));
  } else {
    void closePlayer().then(renderLibrary);
  }
}

/* ---------- boot ---------- */

async function boot() {
  initTheme();
  initConvertForm();
  initPlayerControls();
  window.addEventListener("hashchange", route);
  if (new URLSearchParams(location.search).has("mock")) {
    const mod = await import("./mock/mock.js");
    mod.installMock();
  }
  route();
}

void boot();
