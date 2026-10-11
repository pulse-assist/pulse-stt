"""Плагин «SpeechToText»: распознавание речи для Пульса — голосовые в Telegram и микрофон в чатах (POST /speech/transcribe,
контракт 1.6), расшифровки записей и видео по просьбе агентов (pulse stt transcribe), страница с моделями, пробой,
историей и расходом.

Страница (pulse.plugin): GET /state, POST /try, GET /check.
Агенты: pulse stt transcribe | result | models | history | check.
"""

import base64
import tempfile
import threading
import time
from pathlib import Path

import httpx
from fastapi import Body, HTTPException
from pulse_plugin import CommandError, Plugin
from pulse_plugin.speech import BadAudio, NotConfigured, TooLong, Upstream

from . import net
from .audio import AudioError, Ffmpeg
from .service import NotConfiguredError, RecognizeError, Recognizer, TooLongError, render
from .store import Store

MAX_VOICE_SEC = 30 * 60          # голосовое в Telegram или запись в чате; длиннее — командой агента
MAX_TRY_BYTES = 25 * 1024 * 1024
AGENT_TEXT_CHARS = 30_000        # без --out агенту — текст до стольких символов
WAIT_DEFAULT, WAIT_MAX = 100, 280
FORMATS = ("text", "srt", "vtt", "md", "json")
KEEPALIVE_SEC = 240              # билет файла живёт 10 минут с последнего обращения

plugin = Plugin()
app = plugin.app
store = Store(plugin.data_dir, int(plugin.settings.get("history_days") or 30))
store.interrupt_running()
recognizer = Recognizer(store, plugin.settings, plugin.secrets, Ffmpeg(), lambda: net.build_http(plugin.secrets))
_done: dict[str, threading.Event] = {}


def run_speech(fn):
    """Ошибки распознавания → ошибки протокола /speech/transcribe."""
    try:
        return fn()
    except AudioError as exc:
        raise BadAudio(str(exc)) from exc
    except TooLongError as exc:
        raise TooLong(str(exc)) from exc
    except NotConfiguredError as exc:
        raise NotConfigured(str(exc)) from exc
    except RecognizeError as exc:
        raise Upstream(str(exc)) from exc


@plugin.provide_speech
def speech(path: Path, *, language, source, filename, mime) -> dict:
    result = run_speech(lambda: recognizer.transcribe(path, language=language, source=source, name=filename,
                                                      max_sec=MAX_VOICE_SEC))
    return {k: result[k] for k in ("text", "language", "duration_sec", "model")}


# --- страница -------------------------------------------------------------------------------------------

def engines_state() -> list[dict]:
    out, first = [], True
    for engine in recognizer.engines():
        item = engine.out()
        item["role"] = ("основная" if first else "резервная") if engine.configured else None
        first = first and not engine.configured
        out.append(item)
    return out


@app.get("/state")
def state() -> dict:
    try:
        proxy = net.proxy_label(net.proxy_of(plugin.secrets))
    except net.ProxyError as exc:
        proxy = f"ошибка: {exc}"
    return {"engines": engines_state(), "language": recognizer.language(None) or "", "proxy": proxy,
            "vocabulary": bool((plugin.settings.get("vocabulary") or "").strip()),
            "ffmpeg": bool(recognizer.ffmpeg.exe), "usage": store.usage(), "history": store.history(20),
            "history_days": store.history_days}


@app.post("/try")
def try_it(body: dict = Body(...)) -> dict:
    """Проба со страницы: запись с микрофона или файл (base64 — мост страниц передаёт только JSON)."""
    try:
        data = base64.b64decode(body.get("data") or "", validate=True)
    except ValueError:
        raise HTTPException(400, "Запись не прочиталась") from None
    if not data:
        raise HTTPException(400, "Пустая запись")
    if len(data) > MAX_TRY_BYTES:
        raise HTTPException(413, "Для пробы — файл до 25 МБ; длинные записи — командой агента")
    name = Path(str(body.get("filename") or "probe.webm")).name
    with tempfile.TemporaryDirectory(prefix="pulse-stt-") as tmp:
        src = Path(tmp) / ("probe" + (Path(name).suffix[:10] or ".webm"))
        src.write_bytes(data)
        began = time.monotonic()
        try:
            result = recognizer.transcribe(src, source="page", name=name, engine=body.get("engine") or None,
                                           language=body.get("language") or None, max_sec=MAX_VOICE_SEC)
        except (AudioError, TooLongError, NotConfiguredError, RecognizeError) as exc:
            raise HTTPException(400, str(exc)) from exc
    return {k: result[k] for k in ("text", "language", "duration_sec", "model", "engine", "tried")} | {
        "elapsed_sec": round(time.monotonic() - began, 1)}


@app.get("/check")
def check_page() -> dict:
    return connection_check()


def connection_check() -> dict:
    checks = []
    try:
        proxy = net.proxy_of(plugin.secrets)
        if proxy:
            checks.append({"title": "Прокси", "ok": True, "detail": net.proxy_label(proxy)})
    except net.ProxyError as exc:
        checks.append({"title": "Прокси", "ok": False, "detail": str(exc)})
    checks += recognizer.check()
    return {"ok": all(c["ok"] for c in checks), "checks": checks}


# --- команды агентов ------------------------------------------------------------------------------------

def fetch(arg, dest: Path) -> str:
    """Файл из аргумента-билета → во временную папку; имя файла."""
    ticket = plugin.file(arg)
    name = Path(str((arg.get("name") if isinstance(arg, dict) else None) or ticket.meta().get("name") or "audio")).name
    target = dest / name
    with plugin.api._http.stream("GET", f"/api/plugin-files/{ticket.ticket}/content",  # noqa: SLF001
                                 timeout=httpx.Timeout(600, connect=10)) as response:
        if response.status_code >= 400:
            response.read()
            raise CommandError(f"Файл не скачался: {response.status_code} {response.text[:200]}")
        with target.open("wb") as out:
            for chunk in response.iter_bytes(1024 * 1024):
                out.write(chunk)
    return name


def save(arg, text: str) -> str:
    ticket = plugin.file(arg)
    ticket.write(text.encode("utf-8"))
    return (arg.get("path") if isinstance(arg, dict) else None) or ticket.meta().get("path") or ""


def _keepalive(arg, stop: threading.Event) -> None:
    """Долгая расшифровка: билет файла результата не должен истечь, пока идёт работа."""
    while not stop.wait(KEEPALIVE_SEC):
        try:
            plugin.api.get(f"/api/plugin-files/{plugin.file(arg).ticket}")     # обращение продлевает билет
        except Exception:  # noqa: BLE001 — истечёт так истечёт: ошибка будет при сохранении
            pass


def _job(job: str, args: dict, fmt: str) -> None:
    stop = threading.Event()
    if args.get("out"):
        threading.Thread(target=_keepalive, args=(args["out"], stop), daemon=True).start()
    try:
        with tempfile.TemporaryDirectory(prefix="pulse-stt-") as tmp:
            store.update_job(job, progress="скачиваю файл")
            name = fetch(args["file"], Path(tmp))
            result = recognizer.transcribe(
                Path(tmp) / name, language=args.get("language") or None, source="agent", name=name,
                engine=args.get("engine") or None, segments=fmt in ("srt", "vtt", "md", "json"),
                progress=lambda p: store.update_job(job, progress=p))
        body = render(result, "text" if fmt == "json" else fmt)
        out = {"status": "done", "file": name, "duration_sec": result["duration_sec"], "model": result["model"],
               "parts": result["parts"], "language": result.get("language"), "chars": len(result["text"]),
               "format": fmt}
        if result["tried"]:
            out["fallback"] = result["tried"]
        if args.get("out"):
            content = body if fmt != "json" else _json(result)
            out["saved"] = save(args["out"], content)
        elif fmt == "json":
            out["segments"] = result["segments"][:2000]
            out["text"] = result["text"][:AGENT_TEXT_CHARS]
        else:
            out["text"] = body[:AGENT_TEXT_CHARS]
        if not args.get("out") and len(body) > AGENT_TEXT_CHARS:
            out["truncated"] = f"показаны первые {AGENT_TEXT_CHARS} символов — полностью: --out ПУТЬ"
        store.update_job(job, status="done", result=out)
    except (AudioError, TooLongError, NotConfiguredError, RecognizeError, CommandError) as exc:
        store.update_job(job, status="error", error=str(exc))
    except Exception as exc:  # noqa: BLE001 — задание не должно висеть «в работе»
        store.update_job(job, status="error", error=f"{exc.__class__.__name__}: {exc}")
    finally:
        stop.set()
        _done.pop(job, threading.Event()).set()


def _json(result: dict) -> str:
    import json

    keep = {k: result[k] for k in ("text", "language", "duration_sec", "model", "segments")}
    return json.dumps(keep, ensure_ascii=False, indent=1)


def _wait(job: str, seconds) -> dict:
    try:
        seconds = min(max(int(seconds), 0), WAIT_MAX)
    except (TypeError, ValueError):
        seconds = WAIT_DEFAULT
    event = _done.get(job)
    if event:
        event.wait(seconds)
    info = store.job(job)
    if not info:
        raise CommandError(f"Нет задания {job}")
    if info["status"] == "done":
        return info["result"]
    if info["status"] == "error":
        raise CommandError(f"Расшифровка не удалась: {info['error']}")
    return {"status": "running", "job": job, "progress": info["progress"],
            "next": f"pulse stt result --job {job} — дождаться результата (ждёт до {WAIT_MAX} с за вызов)"}


@plugin.command("transcribe", "Расшифровать аудио или видео: текст, субтитры или текст с метками времени")
def transcribe_command(args: dict) -> dict:
    if not args.get("file"):
        raise CommandError("Нужен --file: запись или видео (путь в «Файлах», @yadisk/…, вложение чата)")
    fmt = str(args.get("format") or "text").lower()
    if fmt not in FORMATS:
        raise CommandError(f"--format — один из: {', '.join(FORMATS)}")
    if not recognizer.active():
        raise CommandError("Не задан ни один сервис распознавания — ключ OpenAI, Groq, SpeechKit или адрес своего "
                           "сервера в настройках плагина «SpeechToText»")
    job = store.new_job()
    _done[job] = threading.Event()
    threading.Thread(target=_job, args=(job, args, fmt), daemon=True).start()
    return _wait(job, args.get("wait", WAIT_DEFAULT))


@plugin.command("result", "Результат расшифровки, которая ещё шла: ждать до --wait секунд")
def result_command(args: dict) -> dict:
    if not args.get("job"):
        raise CommandError("Нужен --job из ответа pulse stt transcribe")
    return _wait(str(args["job"]), args.get("wait", WAIT_DEFAULT))


@plugin.command("models", "Сервисы распознавания: порядок, модели, что настроено")
def models_command(args: dict) -> dict:
    return {"engines": engines_state(), "language": recognizer.language(None) or "auto"}


@plugin.command("history", "Последние распознавания: откуда, длительность, модель, начало текста")
def history_command(args: dict) -> dict:
    try:
        limit = min(max(int(args.get("limit") or 10), 1), 100)
    except (TypeError, ValueError):
        limit = 10
    return {"history": store.history(limit), "usage": store.usage()}


@plugin.command("check", "Проверить подключение: ffmpeg, прокси, каждый настроенный сервис")
def check_command(args: dict) -> dict:
    return connection_check()
