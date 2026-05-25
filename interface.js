'use strict';

/* =========================================================================
   Config
   ========================================================================= */
const CONFIG = {
  DB_NAME:       'gift-of-gab',
  DB_VERSION:    3,
  STORE:         'model',
  TOKENIZER_URL: 'tokenizer.json',
  MODELS: {
    q8: {
      name: 'Quantized',
      button: 'Download Quantized Q8 (100 MiB)',
      manifestUrl: 'model-q8.json',
      weightUrls: ['model-q8.bin'],
    },
    f32: {
      name: 'Full F32',
      button: 'Download Full F32 (380 MiB)',
      manifestUrl: 'model-f32.json',
      weightUrls: [
        'model-f32-000.bin',
        'model-f32-001.bin',
        'model-f32-002.bin',
        'model-f32-003.bin',
        'model-f32-004.bin',
      ],
    },
  },

  MAX_CONTEXT_TOKENS: 4096,
  MAX_NEW_TOKENS:     2048,
  TEMPERATURE:        0.8,
  TOP_K:              50,
};

const ROLE_TOKENS = {
  user:       '<|user|>',
  assistant:  '<|assistant|>',
  end:        '<|end|>',
  thinkOpen:  '<think>',
  thinkClose: '</think>',
};

/* =========================================================================
   ModelStorage — IndexedDB wrapper. Stores model blobs by key.
   ========================================================================= */
class ModelStorage {
  constructor(dbName, version, store) {
    this.dbName = dbName;
    this.version = version;
    this.store = store;
    this.db = null;
  }

  async open() {
    if (this.db) return this.db;
    this.db = await new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, this.version);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains(this.store)) {
          db.createObjectStore(this.store);
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror   = () => reject(req.error);
    });
    return this.db;
  }

  async _tx(mode) {
    const db = await this.open();
    return db.transaction(this.store, mode).objectStore(this.store);
  }

  async has(key) {
    const store = await this._tx('readonly');
    return new Promise((resolve, reject) => {
      const req = store.getKey(key);
      req.onsuccess = () => resolve(req.result !== undefined);
      req.onerror   = () => reject(req.error);
    });
  }

  async get(key) {
    const store = await this._tx('readonly');
    return new Promise((resolve, reject) => {
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror   = () => reject(req.error);
    });
  }

  async put(key, value) {
    const store = await this._tx('readwrite');
    return new Promise((resolve, reject) => {
      const req = store.put(value, key);
      req.onsuccess = () => resolve();
      req.onerror   = () => reject(req.error);
    });
  }

  close() {
    if (this.db) {
      this.db.close();
      this.db = null;
    }
  }

  async deleteDatabase() {
    this.close();
    return new Promise((resolve, reject) => {
      const req = indexedDB.deleteDatabase(this.dbName);
      req.onsuccess = () => resolve();
      req.onerror = () => reject(req.error || new Error('IndexedDB delete failed'));
      req.onblocked = () => reject(new Error('IndexedDB delete is blocked by another open tab.'));
    });
  }
}

/* =========================================================================
   ModelDownloader — streams tokenizer, manifest, and weights with progress.
   ========================================================================= */
class ModelDownloader {
  constructor(storage) {
    this.storage = storage;
  }

  async download({ mode, tokenizerUrl, manifestUrl, weightUrls, onProgress }) {
    if (!mode) throw new Error('No model mode selected.');
    const files = [
      { key: 'manifest',  label: 'manifest',  url: manifestUrl },
      { key: 'tokenizer', label: 'tokenizer', url: tokenizerUrl },
      ...weightUrls.map((url, i) => ({
        key: weightUrls.length === 1 ? 'weights' : 'weight:' + i,
        label: weightUrls.length === 1 ? 'weights' : 'weights ' + (i + 1) + '/' + weightUrls.length,
        url,
      })),
    ];
    if (files.some(f => !f.url)) throw new Error('Model URLs are not configured.');

    await this.storage.put('modelMode', mode);
    let completed = 0;
    for (const file of files) {
      const blob = await this._streamFetch(file.url, (received, total) => {
        onProgress({
          file: file.label,
          received,
          total,
          fraction: this._progressFraction(completed, received, total, files.length),
        });
      });
      await this.storage.put(file.key, blob);
      completed++;
      onProgress({
        file: file.label,
        received: blob.size,
        total: blob.size,
        fraction: completed / files.length,
      });
    }
  }

  async _streamFetch(url, onChunk) {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Download failed: ' + response.status + ' ' + url);
    const total = Number(response.headers.get('Content-Length')) || 0;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onChunk(received, total);
    }
    return new Blob(chunks);
  }

  _progressFraction(completedFiles, received, total, fileCount) {
    const fileFraction = total > 0 ? received / total : 0;
    return (completedFiles + fileFraction) / fileCount;
  }
}

/* =========================================================================
   Message + ChatHistory
   ========================================================================= */
class Message {
  constructor(role) {
    this.role = role;          // 'user' | 'assistant' | 'thinking'
    this.tokens = [];          // [{ id, text }]
    this.complete = false;
    this.id = 'm' + Math.random().toString(36).slice(2, 9);
  }

  get text() {
    let out = '';
    for (const tok of this.tokens) out += tok.text;
    return out;
  }

  addToken(tok) { this.tokens.push(tok); }

  setText(text, tokenizer) {
    this.tokens = tokenizer.tokenize(text);
  }
}

class ChatHistory {
  constructor() { this.messages = []; }
  add(msg) { this.messages.push(msg); return msg; }
  clear()  { this.messages = []; }
}

/* =========================================================================
   Renderers — both update on every token. The CSS toggle on .app[data-view]
   shows the one that's active. Switching views is instant.
   ========================================================================= */
class ChatRenderer {
  constructor(container) {
    this.container = container;
    this.bubbles = new Map(); // messageId -> bubble element
  }

  clear() {
    this.container.innerHTML = '';
    this.bubbles.clear();
  }

  appendMessage(msg) {
    const wrap = document.createElement('div');
    wrap.className = 'msg ' + msg.role;
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (!msg.complete) bubble.classList.add('streaming');
    bubble.textContent = msg.text;
    wrap.appendChild(bubble);
    this.container.appendChild(wrap);
    this.bubbles.set(msg.id, bubble);
  }

  appendToken(msg, tok) {
    const bubble = this.bubbles.get(msg.id);
    if (!bubble) return;
    bubble.textContent += tok.text;
  }

  finalize(msg) {
    const bubble = this.bubbles.get(msg.id);
    if (bubble) bubble.classList.remove('streaming');
  }
}

class RawRenderer {
  constructor(container) {
    this.container = container;
    this.lines = new Map();           // messageId -> wrapper element
    this._contentCount = 0;           // BPE content tokens (drives alternation)
    this._totalCount = 0;             // every token incl. role markers (status bar)
    this._lastFinalizedRole = null;   // for turn-boundary delimiter logic
    this._initStatusBar();
  }

  _initStatusBar() {
    this.statusBar = document.createElement('div');
    this.statusBar.className = 'raw-statusbar';
    this.statusBar.innerHTML = '<span id="rawTokens">0 tokens</span><span id="rawHover">&nbsp;</span>';
    this.container.appendChild(this.statusBar);
    this.tokenCounter = this.statusBar.querySelector('#rawTokens');
    this.hoverLabel   = this.statusBar.querySelector('#rawHover');

    this.container.addEventListener('mouseover', (e) => {
      const tok = e.target.closest('.tok');
      if (!tok) return;
      const id   = tok.dataset.id;
      const text = tok.dataset.text || '';
      if (id !== undefined && id !== '') {
        this.hoverLabel.textContent = 'id ' + id + '  •  ' + JSON.stringify(text);
      } else {
        this.hoverLabel.textContent = JSON.stringify(text);
      }
    });
    this.container.addEventListener('mouseleave', () => {
      this.hoverLabel.innerHTML = '&nbsp;';
    });
  }

  clear() {
    for (const node of [...this.container.querySelectorAll('.raw-line')]) node.remove();
    this.lines.clear();
    this._contentCount = 0;
    this._totalCount = 0;
    this._lastFinalizedRole = null;
    this._updateCount();
  }

  appendMessage(msg) {
    const line = document.createElement('span');
    line.className = 'raw-line';
    line.dataset.role = msg.role;
    this._emitOpener(line, msg.role, this._lastFinalizedRole);
    this.statusBar.before(line);
    this.lines.set(msg.id, line);
    // Flush any tokens already on the message. User messages arrive fully
    // tokenized via setText(); without this they'd render as just <|user|><|end|>.
    for (const tok of msg.tokens) this._appendContentToken(line, msg, tok);
  }

  appendToken(msg, tok) {
    const line = this.lines.get(msg.id);
    if (!line) return;
    this._appendContentToken(line, msg, tok);
  }

  _appendContentToken(line, msg, tok) {
    const span = document.createElement('span');
    const cls = 'tok-c' + (this._contentCount % 5);
    span.className = 'tok ' + cls;
    if (msg.role === 'thinking') span.classList.add('tok-think');
    span.dataset.id = tok.id;
    span.dataset.text = tok.text;
    span.textContent = tok.text;
    line.appendChild(span);
    this._contentCount++;
    this._totalCount++;
    this._updateCount();
  }

  finalize(msg) {
    const line = this.lines.get(msg.id);
    if (!line) return;
    this._emitCloser(line, msg.role);
    this._lastFinalizedRole = msg.role;
  }

  /* Wire format:
       <|user|>...<|end|>
       <|assistant|>[<think>...</think>]Answer<|end|>
     Thinking lives *inside* the assistant turn, so:
       - opening a 'thinking' msg emits <|assistant|><think>
       - opening an 'assistant' msg right after thinking emits only </think>
         (the assistant turn is already open) */
  _emitOpener(line, role, prevRole) {
    if (role === 'user') {
      this._appendMarker(line, ROLE_TOKENS.user, 'tok-role');
    } else if (role === 'thinking') {
      this._appendMarker(line, ROLE_TOKENS.assistant, 'tok-role');
      this._appendMarker(line, ROLE_TOKENS.thinkOpen, 'tok-mark');
    } else if (role === 'assistant') {
      if (prevRole === 'thinking') {
        this._appendMarker(line, ROLE_TOKENS.thinkClose, 'tok-mark');
      } else {
        this._appendMarker(line, ROLE_TOKENS.assistant, 'tok-role');
      }
    }
  }

  _emitCloser(line, role) {
    // 'thinking' has no closer of its own — </think> is emitted by the
    // following assistant content's opener.
    if (role === 'user' || role === 'assistant') {
      this._appendMarker(line, ROLE_TOKENS.end, 'tok-role');
    }
  }

  _appendMarker(line, text, cls) {
    const span = document.createElement('span');
    span.className = 'tok ' + cls;
    span.dataset.text = text;
    span.textContent = text;
    line.appendChild(span);
    this._totalCount++;
    this._updateCount();
  }

  _updateCount() {
    this.tokenCounter.textContent = this._totalCount + ' tokens';
  }
}

/* =========================================================================
   Modal
   ========================================================================= */
class HelpModal {
  constructor() {
    this.modal     = document.getElementById('modal');
    this.backdrop  = document.getElementById('modalBackdrop');
    this.tabs      = [...document.querySelectorAll('.tab')];
    this.panes     = [...document.querySelectorAll('.modal-pane')];
    this.closeBtn  = document.getElementById('closeModalBtn');
    this.resetBtn  = document.getElementById('resetDbBtn');
    this.resetStatus = document.getElementById('resetStatus');

    for (const tab of this.tabs) {
      tab.addEventListener('click', () => this.selectTab(tab.dataset.tab));
    }
    this.closeBtn.addEventListener('click', () => this.close());
    this.backdrop.addEventListener('click',  () => this.close());
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && this.isOpen()) this.close();
    });
  }

  isOpen() { return this.modal.classList.contains('open'); }

  open() {
    this.modal.classList.add('open');
    this.backdrop.classList.add('open');
  }

  close() {
    this.modal.classList.remove('open');
    this.backdrop.classList.remove('open');
  }

  selectTab(name) {
    for (const t of this.tabs)  t.classList.toggle('active', t.dataset.tab  === name);
    for (const p of this.panes) p.classList.toggle('active', p.dataset.pane === name);
  }
}

/* =========================================================================
   App — wires everything together.
   ========================================================================= */
class App {
  constructor() {
    this.appEl       = document.getElementById('app');
    this.scrollEl    = document.getElementById('chatScroll');
    this.chatEl      = document.getElementById('viewChat');
    this.rawEl       = document.getElementById('viewRaw');
    this.inputEl     = document.getElementById('input');
    this.sendBtn     = document.getElementById('sendBtn');
    this.thinkCb     = document.getElementById('thinkCb');
    this.viewCb      = document.getElementById('viewCb');
    this.helpBtn     = document.getElementById('helpBtn');

    this.downloadScreen = document.getElementById('downloadScreen');
    this.downloadBtns   = [...document.querySelectorAll('.download-btn[data-model]')];
    this.downloadBtn    = this.downloadBtns[0] || null;
    this.progressWrap   = document.getElementById('progressWrap');
    this.progressBar    = document.getElementById('progressBar');
    this.progressFile   = document.getElementById('progressFile');
    this.progressPct    = document.getElementById('progressPct');
    this.downloadErr    = document.getElementById('downloadError');
    this.resetBtn       = document.getElementById('resetDbBtn');
    this.resetStatus    = document.getElementById('resetStatus');

    this.storage    = new ModelStorage(CONFIG.DB_NAME, CONFIG.DB_VERSION, CONFIG.STORE);
    this.downloader = new ModelDownloader(this.storage);
    this.history    = new ChatHistory();
    this.modal      = new HelpModal();

    this.chatRenderer = new ChatRenderer(this.chatEl);
    this.rawRenderer  = new RawRenderer(this.rawEl);

    this.tokenizer = null;
    this.engine    = null;
    this.generating = false;
    this.stopRequested = false;
    this.generationAbort = null;
  }

  async init() {
    this._bindViewport();
    this._bindEvents();
    this._setState('boot');

    let present = false;
    try {
      present =
        (await this.storage.has('modelMode')) &&
        (await this.storage.has('manifest')) &&
        (await this.storage.has('tokenizer')) &&
        ((await this.storage.has('weights')) || (await this.storage.has('weight:0')));
    } catch (e) {
      console.error('IndexedDB unavailable:', e);
    }

    if (present) {
      this._setDownloadButtonsDisabled(true);
      this.progressWrap.classList.add('active');
      this.progressBar.style.width = '100%';
      this.progressPct.textContent = 'loading';
      this.progressFile.textContent = 'loading cached model';
      this._setState('loading');
      try {
        await this._loadModel();
        this._setState('ready');
      } catch (e) {
        console.warn('Cached model load failed:', e);
        this.downloadErr.textContent = 'Cached model could not be loaded. Download it again.';
        this.downloadErr.style.display = 'block';
        this._setDownloadButtonsDisabled(false);
        this._setState('download');
      }
    } else {
      this._setState('download');
    }
  }

  _bindEvents() {
    this.helpBtn.addEventListener('click', () => this.modal.open());
    this.viewCb.addEventListener('change', () => {
      this.appEl.dataset.view = this.viewCb.checked ? 'chat' : 'raw';
    });
    this.inputEl.addEventListener('input', () => {
      this._autosize();
      this._updateSendEnabled();
    });
    this.inputEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (!this.generating) this._handleSend();
      }
    });
    this.sendBtn.addEventListener('click', () => this._handlePrimaryAction());
    for (const btn of this.downloadBtns) {
      btn.addEventListener('click', () => this._handleDownload(btn.dataset.model));
    }
    if (this.resetBtn) {
      this.resetBtn.addEventListener('click', () => this._handleReset());
    }
    this._bindTouchScrollGuard();
  }

  _bindTouchScrollGuard() {
    const scrollables = [this.scrollEl, this.inputEl, document.querySelector('.modal-body')].filter(Boolean);

    const canScroll = (el) => el.scrollHeight > el.clientHeight + 1;
    const nearestScrollable = (target) => {
      for (const el of scrollables) {
        if (el.contains(target) && canScroll(el)) return el;
      }
      return null;
    };

    let startY = 0;
    let activeScroller = null;

    document.addEventListener('touchstart', (e) => {
      if (!e.touches.length) return;
      startY = e.touches[0].clientY;
      activeScroller = nearestScrollable(e.target);
    }, { passive: true });

    document.addEventListener('touchmove', (e) => {
      if (!e.touches.length) return;
      if (!activeScroller) {
        e.preventDefault();
        return;
      }

      const y = e.touches[0].clientY;
      const dy = y - startY;
      const atTop = activeScroller.scrollTop <= 0;
      const atBottom = activeScroller.scrollTop + activeScroller.clientHeight >= activeScroller.scrollHeight - 1;

      if ((dy > 0 && atTop) || (dy < 0 && atBottom)) {
        e.preventDefault();
      }
    }, { passive: false });

    document.addEventListener('touchend', () => {
      activeScroller = null;
    }, { passive: true });
  }

  _bindViewport() {
    const root = document.documentElement;
    let raf = 0;

    const apply = () => {
      raf = 0;
      const vv = window.visualViewport;
      const height = vv ? vv.height : window.innerHeight;
      const top = vv ? vv.offsetTop : 0;
      root.style.setProperty('--app-height', Math.max(1, Math.round(height)) + 'px');
      root.style.setProperty('--visual-top', Math.max(0, Math.round(top)) + 'px');
      this._autosize();
      this._scrollToBottom();
    };

    const schedule = () => {
      if (raf) return;
      raf = requestAnimationFrame(apply);
    };

    window.addEventListener('resize', schedule, { passive: true });
    window.addEventListener('orientationchange', () => {
      schedule();
      setTimeout(schedule, 250);
    }, { passive: true });

    if (window.visualViewport) {
      window.visualViewport.addEventListener('resize', schedule, { passive: true });
      window.visualViewport.addEventListener('scroll', schedule, { passive: true });
    }

    this.inputEl.addEventListener('focus', () => {
      schedule();
      setTimeout(schedule, 80);
      setTimeout(schedule, 320);
    });
    this.inputEl.addEventListener('blur', () => {
      schedule();
      setTimeout(schedule, 120);
      setTimeout(schedule, 360);
    });

    apply();
  }

  _setState(state) {
    this.appEl.dataset.state = state;
    this._updateSendEnabled();
  }

  _updateSendEnabled() {
    const ready = this.appEl.dataset.state === 'ready';
    const hasText = this.inputEl.value.trim().length > 0;
    if (this.generating) {
      this.sendBtn.textContent = 'Stop';
      this.sendBtn.setAttribute('aria-label', 'Stop generation');
      this.sendBtn.disabled = !ready || this.stopRequested;
    } else {
      this.sendBtn.textContent = 'Send';
      this.sendBtn.setAttribute('aria-label', 'Send message');
      this.sendBtn.disabled = !ready || !hasText;
    }
  }

  _setDownloadButtonsDisabled(disabled) {
    for (const btn of this.downloadBtns) btn.disabled = disabled;
  }

  _autosize() {
    const ta = this.inputEl;
    ta.style.height = 'auto';
    // 5 lines ≈ 5 × (15px × 1.5 line-height) + 22px vertical padding
    const max = 138;
    const wanted = ta.scrollHeight;
    if (wanted > max) {
      ta.style.height = max + 'px';
      ta.style.overflowY = 'auto';
    } else {
      ta.style.height = wanted + 'px';
      ta.style.overflowY = 'hidden';
    }
  }

  async _handleDownload(mode) {
    const model = CONFIG.MODELS[mode];
    if (!model) return;
    this._setDownloadButtonsDisabled(true);
    this.downloadErr.style.display = 'none';
    this.progressWrap.classList.add('active');
    this.progressBar.style.width = '0%';
    this.progressPct.textContent = '0%';
    this.progressFile.textContent = model.name;
    this._setState('loading');

    try {
      await this.downloader.download({
        mode,
        tokenizerUrl: CONFIG.TOKENIZER_URL,
        manifestUrl:  model.manifestUrl,
        weightUrls:   model.weightUrls,
        onProgress: (p) => this._onDownloadProgress(p),
      });
      await this._loadModel();
      this._setState('ready');
      this.inputEl.focus();
    } catch (err) {
      console.error(err);
      this.downloadErr.textContent = String(err.message || err);
      this.downloadErr.style.display = 'block';
      this._setDownloadButtonsDisabled(false);
      this._setState('download');
    }
  }

  async _handleReset() {
    if (!this.resetBtn) return;
    this.resetBtn.disabled = true;
    if (this.resetStatus) this.resetStatus.textContent = 'Deleting local model...';
    try {
      await this.storage.deleteDatabase();
      if (this.resetStatus) this.resetStatus.textContent = 'Deleted. Reloading...';
      window.location.reload();
    } catch (err) {
      console.error(err);
      if (this.resetStatus) {
        this.resetStatus.textContent = String(err.message || err);
      }
      this.resetBtn.disabled = false;
    }
  }

  _onDownloadProgress({ file, received, total, fraction }) {
    const pct = Math.round((fraction || 0) * 100);
    this.progressBar.style.width = pct + '%';
    this.progressPct.textContent = pct + '%';
    const mb = (n) => (n / (1024 * 1024)).toFixed(1) + ' MiB';
    if (total) {
      this.progressFile.textContent = file + ' — ' + mb(received) + ' / ' + mb(total);
    } else {
      this.progressFile.textContent = file;
    }
  }

  async _loadModel() {
    const [modeBlob, manifestBlob, tokenizerBlob] = await Promise.all([
      this.storage.get('modelMode'),
      this.storage.get('manifest'),
      this.storage.get('tokenizer'),
    ]);
    if (!modeBlob || !manifestBlob || !tokenizerBlob) {
      throw new Error('Cached model files are incomplete. Download the model again.');
    }

    this.progressFile.textContent = 'loading model';
    const [mode, manifest, tokenizerData] = await Promise.all([
      this._readTextValue(modeBlob),
      manifestBlob.text().then(JSON.parse),
      tokenizerBlob.text().then(JSON.parse),
    ]);
    const weightsBuffer = await this._loadWeightsBuffer(manifest);

    if (manifest.byte_length && weightsBuffer.byteLength !== manifest.byte_length) {
      throw new Error('Cached weights are incomplete. Download the model again.');
    }

    this.tokenizer = new GabTokenizer(tokenizerData);
    this.engine = await GabInferenceEngine.create({
      manifest,
      weightsBuffer,
      tokenizer: this.tokenizer,
      maxContext: CONFIG.MAX_CONTEXT_TOKENS,
    });
    console.info('Gift of Gab model:', mode, 'backend:', this.engine.backend || 'unknown');
  }

  async _readTextValue(value) {
    if (typeof value === 'string') return value;
    if (value && typeof value.text === 'function') return value.text();
    if (value instanceof ArrayBuffer) return new TextDecoder().decode(value);
    return String(value || '');
  }

  async _loadWeightsBuffer(manifest) {
    const shards = manifest.files && manifest.files.weights_shards;
    if (Array.isArray(shards) && shards.length) {
      const blobs = await Promise.all(shards.map((_, i) => this.storage.get('weight:' + i)));
      if (blobs.some(blob => !blob)) {
        throw new Error('Cached model shards are incomplete. Download the model again.');
      }
      const total = blobs.reduce((sum, blob) => sum + blob.size, 0);
      const out = new Uint8Array(total);
      let offset = 0;
      for (const blob of blobs) {
        const chunk = new Uint8Array(await blob.arrayBuffer());
        out.set(chunk, offset);
        offset += chunk.length;
      }
      return out.buffer;
    }

    const weightsBlob = await this.storage.get('weights');
    if (!weightsBlob) {
      throw new Error('Cached model weights are incomplete. Download the model again.');
    }
    return weightsBlob.arrayBuffer();
  }

  _handlePrimaryAction() {
    if (this.generating) {
      this._stopGeneration();
    } else {
      this._handleSend();
    }
  }

  _stopGeneration() {
    if (!this.generating || this.stopRequested) return;
    this.stopRequested = true;
    if (this.generationAbort) this.generationAbort.abort();
    this._updateSendEnabled();
  }

  _isAbortError(err) {
    return err && (err.name === 'AbortError' || err.code === 20);
  }

  async _handleSend() {
    const text = this.inputEl.value.trim();
    if (!text || this.generating || this.appEl.dataset.state !== 'ready') return;

    this.inputEl.value = '';
    this._autosize();
    this.generating = true;
    this.stopRequested = false;
    this.generationAbort = new AbortController();
    this._updateSendEnabled();

    // Add the user message
    const userMsg = new Message('user');
    userMsg.setText(text, this.tokenizer);
    userMsg.complete = true;
    this.history.add(userMsg);
    this.chatRenderer.appendMessage(userMsg);
    this.rawRenderer.appendMessage(userMsg);
    this.chatRenderer.finalize(userMsg);
    this.rawRenderer.finalize(userMsg);
    this._scrollToBottom();

    // Stream the assistant response (possibly preceded by a thinking message)
    const thinking = this.thinkCb.checked;
    let current = null;

    try {
      for await (const { kind, token } of this.engine.generate(this.history.messages, {
        thinking,
        maxNewTokens: CONFIG.MAX_NEW_TOKENS,
        temperature: CONFIG.TEMPERATURE,
        topK: CONFIG.TOP_K,
        signal: this.generationAbort.signal,
      })) {
        if (this.stopRequested) break;
        const wantRole = (kind === 'thinking') ? 'thinking' : 'assistant';
        if (!current || current.role !== wantRole) {
          if (current) {
            current.complete = true;
            this.chatRenderer.finalize(current);
            this.rawRenderer.finalize(current);
          }
          current = new Message(wantRole);
          this.history.add(current);
          this.chatRenderer.appendMessage(current);
          this.rawRenderer.appendMessage(current);
        }
        current.addToken(token);
        this.chatRenderer.appendToken(current, token);
        this.rawRenderer.appendToken(current, token);
        this._scrollToBottom();
      }
    } catch (e) {
      if (!this._isAbortError(e)) console.error('Generation error:', e);
    } finally {
      if (current && !current.complete) {
        current.complete = true;
        this.chatRenderer.finalize(current);
        this.rawRenderer.finalize(current);
      }
      this.generating = false;
      this.stopRequested = false;
      this.generationAbort = null;
      this._updateSendEnabled();
      this.inputEl.focus();
    }
  }

  _scrollToBottom() {
    // Pin to bottom if user hasn't scrolled away
    const el = this.scrollEl;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const app = new App();
  app.init();
  // Expose for console tinkering during development
  window.__app = app;
});
