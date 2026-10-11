// Страница «Распознавание речи»: модели по порядку, проба (микрофон или файл), история и расход за месяц.
// Всё — через мост Пульса: pulse.plugin(...) к серверу плагина, pulse.navigate(...) — в разделы Пульса.
// Мост передаёт только JSON — запись для пробы уходит в base64.

const app = document.getElementById("app");
const fileInput = document.getElementById("file");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const SOURCES = { telegram: "Telegram", chat: "Чат", agent: "Агент", page: "Проба" };
const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
const MAX_TRY_BYTES = 25 * 1024 * 1024;

let data = null;
let error = "";
let checks = null;          // результат «Проверить подключение»
let checking = false;
let tryEngine = "";
let tryState = "idle";      // idle | recording | recognizing
let tryResult = null;
let tryError = "";
let recStarted = 0;
let recorder = null;
let timer = null;

function clock(sec) {
  if (sec == null) return "";
  const s = Math.round(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}` : `${m}:${String(r).padStart(2, "0")}`;
}

function minutes(sec) {
  const m = sec / 60;
  return m < 1 ? `${Math.round(sec)} с` : `${m < 10 ? m.toFixed(1).replace(".", ",") : Math.round(m)} мин`;
}

function when(iso) {
  const d = new Date(iso), now = new Date();
  const hm = d.toLocaleTimeString("ru", { hour: "2-digit", minute: "2-digit" });
  if (d.toDateString() === now.toDateString()) return hm;
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return `вчера ${hm}`;
  return `${d.toLocaleDateString("ru", { day: "2-digit", month: "2-digit" })} ${hm}`;
}

function money(cost, currency) {
  if (!currency || !cost) return "";
  const v = cost < 1 ? cost.toFixed(2) : cost.toFixed(cost < 10 ? 1 : 0);
  return ` · ≈ ${currency === "$" ? "$" : ""}${v.replace(".", ",")}${currency === "₽" ? " ₽" : ""}`;
}

async function load() {
  try {
    data = await pulse.plugin("/state");
    error = "";
  } catch (e) {
    if (!data) { app.innerHTML = `<div class="pl-error">Плагин не отвечает: ${esc(e.message)}</div>`; return; }
    error = e.message;
  }
  render();
}

// --- отрисовка ----------------------------------------------------------------------------------------------

function render() {
  const engines = data.engines;
  const main = engines.find((e) => e.role === "основная");
  const total = data.usage.reduce((s, u) => s + u.seconds, 0);
  app.innerHTML = `
    <div class="status">
      <span>${main ? `основная модель — ${esc(main.title)}` : "<span class='warn'>не настроен ни один сервис</span>"}
        · ${minutes(total)} за ${MONTHS[new Date().getMonth()]}</span>
      <span class="grow"></span>
      <button class="btn" data-act="check" ${checking ? "disabled" : ""}>${checking ? '<span class="spinner"></span>Проверяю…' : "Проверить подключение"}</button>
      <button class="btn" data-go="/settings/plugins/stt">Настройки плагина</button>
    </div>
    ${error ? `<div class="error">${esc(error)}</div>` : ""}
    ${checks ? renderChecks() : ""}
    <div class="cols">
      <div class="col">${renderEngines(engines)}${renderHistory()}</div>
      <div class="col">${renderTry(engines)}${renderUsage()}${renderWhere()}</div>
    </div>`;
}

function renderChecks() {
  return `<div class="checks" aria-label="Проверка подключения">
    ${checks.map((c) => `<div><span class="${c.ok ? "ok" : "error"}">${c.ok ? "✓" : "✕"}</span>
      <span><b>${esc(c.title)}</b> — ${esc(c.detail || "")}</span></div>`).join("")}
  </div>`;
}

function renderEngines(engines) {
  const rows = engines.map((e) => `
    <tr>
      <td class="name"><b>${esc(e.title)}</b><div class="sub">${e.host ? esc(e.host) + " · " : ""}${e.configured ? "задан" : "не настроен"}</div></td>
      <td class="${e.configured ? "" : "muted"}">${esc(e.model)}</td>
      <td class="${e.configured ? "ok" : "muted"}">${e.configured ? "● готов" : `○ нужен ${esc(e.missing)}`}</td>
      <td>${e.role ? `<span class="badge ${e.role === "основная" ? "badge-main" : ""}">${e.role}</span>` : "<span class='muted'>—</span>"}</td>
    </tr>`).join("");
  return `<section class="box">
    <div class="box-head"><h2>Модели</h2><span class="hint">по порядку: не ответила основная — пробуется следующая</span></div>
    <table class="grid"><thead><tr><th>Сервис</th><th>Модель</th><th>Состояние</th><th>Роль</th></tr></thead><tbody>${rows}</tbody></table>
    <div class="box-foot hint">Ключи, порядок, язык (${data.language ? esc(data.language) : "определять сам"}) и словарь
      (${data.vocabulary ? "задан" : "не задан"} — имена и термины для точности) — в
      <button class="btn-link" data-go="/settings/plugins/stt">настройках плагина</button>; ключи — секреты, маской.
      ${data.proxy ? `Прокси: ${esc(data.proxy)}.` : ""} ${data.ffmpeg ? "" : "<span class='error'>Нет ffmpeg — переустановите плагин.</span>"}</div>
  </section>`;
}

function renderHistory() {
  const rows = data.history.map((h) => `
    <tr>
      <td>${esc(when(h.at))}</td>
      <td>${esc(SOURCES[h.source] || h.source)}</td>
      <td class="opt">${clock(h.duration)}</td>
      <td class="text" title="${esc(h.error || h.text)}">${h.error ? `<span class="error">${esc(h.error)}</span>` : `${h.name && h.source === "agent" ? `${esc(h.name)} — ` : ""}${esc(h.text)}`}</td>
      <td class="opt ${h.note ? "warn" : ""}">${esc(h.engine || "—")}${h.note ? " · основная не ответила" : ""}</td>
    </tr>`).join("");
  return `<section class="box">
    <div class="box-head"><h2>Последние распознавания</h2><span class="hint">текст хранится ${data.history_days} дней, сами записи — не хранятся</span></div>
    ${data.history.length ? `<table class="grid"><thead><tr><th>Когда</th><th>Откуда</th><th class="opt">Длит.</th><th>Текст</th><th class="opt">Модель</th></tr></thead><tbody>${rows}</tbody></table>`
      : `<div class="box-body hint">Пока пусто: отправьте голосовое боту в Telegram или нажмите микрофон в чате.</div>`}
  </section>`;
}

function renderTry(engines) {
  const ready = engines.filter((e) => e.configured);
  let body = "";
  if (tryState === "recording") {
    body = `<div class="rec"><span class="dot"></span><b>${clock((Date.now() - recStarted) / 1000)}</b> идёт запись
      <span style="flex-grow:1"></span><button class="btn" data-act="rec-cancel">Отмена</button>
      <button class="btn btn-danger" data-act="rec-stop">Готово</button></div>`;
  } else if (tryState === "recognizing") {
    body = `<div class="rec"><span class="spinner"></span>Распознаю…</div>`;
  } else if (tryResult) {
    body = `<div class="result">«${esc(tryResult.text || "—")}»
      <div class="hint">${esc(tryResult.model)} · ${clock(tryResult.duration_sec)} · ${String(tryResult.elapsed_sec).replace(".", ",")} с${tryResult.language ? ` · ${esc(tryResult.language)}` : ""}${tryResult.tried?.length ? ` · до неё: ${esc(tryResult.tried.join("; "))}` : ""}</div></div>`;
  }
  return `<section class="box">
    <div class="box-head"><h2>Попробовать</h2></div>
    <div class="box-body">
      <div class="try-row">
        <button class="btn btn-primary" data-act="rec" ${tryState !== "idle" || !ready.length ? "disabled" : ""}>● Записать</button>
        <button class="btn" data-act="file" ${tryState !== "idle" || !ready.length ? "disabled" : ""}>Файл…</button>
        <label>Модель <select data-act="engine" aria-label="Модель для пробы">
          <option value="">все по порядку</option>
          ${ready.map((e) => `<option value="${e.id}" ${tryEngine === e.id ? "selected" : ""}>${esc(e.title)}</option>`).join("")}
        </select></label>
      </div>
      ${body}
      ${tryError ? `<div class="error">${esc(tryError)}</div>` : ""}
      <div class="hint">${ready.length ? "Сравнить модели — выбрать одну и распознать ту же запись ещё раз." : "Сначала задайте ключ хотя бы одного сервиса в настройках плагина."}</div>
    </div>
  </section>`;
}

function renderUsage() {
  const max = Math.max(1, ...data.usage.map((u) => u.seconds));
  const titles = Object.fromEntries(data.engines.map((e) => [e.id, e.title]));
  const rows = data.usage.map((u) => `<div class="usage"><span class="name">${esc(titles[u.engine] || u.engine)}</span>
    <div class="bar"><div style="width:${Math.round((u.seconds / max) * 100)}%"></div></div>
    <span class="num">${minutes(u.seconds)}${money(u.cost, u.currency)}</span></div>`).join("");
  return `<section class="box">
    <div class="box-head"><h2>За ${MONTHS[new Date().getMonth()]}</h2></div>
    <div class="box-body">${rows || "<span class='hint'>Распознаваний ещё не было.</span>"}
      <div class="hint">Оценка по прайсу сервиса; точный счёт — в кабинете сервиса.</div></div>
  </section>`;
}

function renderWhere() {
  return `<section class="box"><div class="box-head"><h2>Где работает</h2></div><div class="box-body">
    <div class="where"><span class="ok">✓</span><span>Голосовые и «кружочки» боту в Telegram — сразу ответ агента</span></div>
    <div class="where"><span class="ok">✓</span><span>Микрофон в поле ввода чатов Пульса — сказанное сразу отправляется</span></div>
    <div class="where"><span class="ok">✓</span><span>Агенты: <code>pulse stt transcribe</code> — аудио и видео любой длины, файлы с @yadisk и @gdrive, субтитры</span></div>
  </div></section>`;
}

// --- действия -----------------------------------------------------------------------------------------------

async function recognize(blob, name) {
  if (blob.size > MAX_TRY_BYTES) { tryError = "Для пробы — файл до 25 МБ; длинные записи расшифровывает агент."; render(); return; }
  tryState = "recognizing";
  tryError = "";
  tryResult = null;
  render();
  try {
    const b64 = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1] || "");
      reader.onerror = () => reject(new Error("файл не прочитался"));
      reader.readAsDataURL(blob);
    });
    tryResult = await pulse.plugin("/try", { method: "POST", body: { data: b64, filename: name, mime: blob.type, engine: tryEngine } });
  } catch (e) {
    tryError = `Не распознано: ${e.message}`;
  }
  tryState = "idle";
  await load();
}

async function startRecording() {
  tryError = "";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (e) {
    tryError = "Браузер не дал микрофон странице плагина. Попробуйте «Файл…» или микрофон в поле ввода чата.";
    render();
    return;
  }
  const chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    clearInterval(timer);
    const cancelled = recorder.cancelled;
    recorder = null;
    if (cancelled || !chunks.length) { tryState = "idle"; render(); return; }
    const type = chunks[0].type || "audio/webm";
    void recognize(new Blob(chunks, { type }), type.includes("mp4") ? "probe.m4a" : type.includes("ogg") ? "probe.ogg" : "probe.webm");
  };
  recorder.start();
  recStarted = Date.now();
  tryState = "recording";
  timer = setInterval(render, 500);
  render();
}

async function check() {
  checking = true;
  render();
  try {
    checks = (await pulse.plugin("/check")).checks;
  } catch (e) {
    checks = [{ title: "Проверка", ok: false, detail: e.message }];
  }
  checking = false;
  render();
}

app.addEventListener("click", (event) => {
  const el = event.target.closest("[data-act], [data-go]");
  if (!el) return;
  if (el.dataset.go) { pulse.navigate(el.dataset.go); return; }
  const act = el.dataset.act;
  if (act === "check") void check();
  if (act === "rec") void startRecording();
  if (act === "rec-stop" && recorder) recorder.stop();
  if (act === "rec-cancel" && recorder) { recorder.cancelled = true; recorder.stop(); }
  if (act === "file") fileInput.click();
});

app.addEventListener("change", (event) => {
  if (event.target.dataset.act === "engine") tryEngine = event.target.value;
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) void recognize(file, file.name);
});

void load();
