"""Распознавание: модели по порядку (не ответила основная — следующая), запись любой длины — частями по паузам,
параллельно; текст, время фраз, субтитры. История и расход — в Store."""

import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from . import engines as eng
from .audio import AudioError, Ffmpeg, plan
from .store import Store

SUBTITLE_PART_SEC = 30          # модель без времени фраз: для субтитров режем мельче — время по частям


class NotConfiguredError(Exception):
    pass


class TooLongError(Exception):
    pass


class RecognizeError(Exception):
    """Ни одна модель не распознала — причины по каждой."""


class Recognizer:
    def __init__(self, store: Store, settings: dict, secrets: dict, ffmpeg: Ffmpeg | None = None,
                 http_factory: Callable[[], httpx.Client] | None = None, workers: int = 3):
        self.store = store
        self.settings = settings
        self.secrets = secrets
        self.ffmpeg = ffmpeg or Ffmpeg()
        self.http_factory = http_factory or (lambda: httpx.Client())
        self.workers = workers
        self._http: httpx.Client | None = None

    @property
    def http(self) -> httpx.Client:
        if self._http is None:
            self._http = self.http_factory()
        return self._http

    def engines(self) -> list[eng.Engine]:
        """Все сервисы по порядку из настроек."""
        found = eng.build(self.settings, self.secrets)
        return [found[i] for i in eng.order(self.settings)]

    def active(self) -> list[eng.Engine]:
        return [e for e in self.engines() if e.configured]

    def language(self, language: str | None) -> str | None:
        value = (language or self.settings.get("language") or "").strip().lower()
        return None if value in ("", "auto") else value[:5]

    def transcribe(self, src: Path, *, language: str | None = None, source: str = "chat", name: str | None = None,
                   engine: str | None = None, segments: bool = False, max_sec: float | None = None,
                   progress: Callable[[str], None] | None = None) -> dict:
        """Запись → {text, language, duration_sec, model, engine, segments, parts, tried}."""
        candidates = self.active()
        if engine:
            candidates = [e for e in candidates if e.id == engine]
            if not candidates:
                raise NotConfiguredError(f"сервис {engine} не настроен или неизвестен — pulse stt models")
        if not candidates:
            raise NotConfiguredError("не задан ни один сервис распознавания — ключ OpenAI, Groq, SpeechKit или адрес "
                                     "своего сервера в настройках плагина")
        language = self.language(language)
        prompt = (self.settings.get("vocabulary") or "").strip() or None
        duration = None
        try:
            with tempfile.TemporaryDirectory(prefix="pulse-stt-") as tmp:
                work = Path(tmp)
                (progress or (lambda _: None))("готовлю звук")
                prepared = self.ffmpeg.prepare(src, work)
                duration = prepared.duration
                if max_sec and duration > max_sec:
                    raise TooLongError(f"запись {clock(duration)} — длиннее {clock(max_sec)}; длинные записи — командой "
                                       "pulse stt transcribe")
                pauses: list[float] | None = None
                tried: list[str] = []
                for candidate in candidates:
                    part_sec = candidate.max_sec
                    if segments and not candidate.segments:
                        part_sec = min(part_sec, SUBTITLE_PART_SEC)
                    if duration > part_sec and pauses is None:
                        pauses = self.ffmpeg.silences(prepared.path)
                    parts = plan(duration, part_sec, pauses or [])
                    try:
                        result = self._run(candidate, prepared.path, parts, work, language, prompt, progress)
                    except eng.EngineError as exc:
                        tried.append(str(exc))
                        continue
                    self.store.add_usage(candidate.id, duration, candidate.cost(duration))
                    self.store.add_history(source=source, name=name, duration=duration, engine=candidate.id,
                                           model=candidate.model, text=result["text"],
                                           note="; ".join(tried))
                    return {**result, "language": result.get("language") or language, "duration_sec": round(duration, 2),
                            "model": f"{candidate.title} {candidate.model}", "engine": candidate.id,
                            "parts": len(parts), "tried": tried}
                raise RecognizeError("; ".join(tried))
        except (AudioError, TooLongError, RecognizeError) as exc:
            self.store.add_history(source=source, name=name, duration=duration, error=str(exc))
            raise

    def _run(self, engine: eng.Engine, path: Path, parts: list[tuple[float, float]], work: Path,
             language: str | None, prompt: str | None, progress) -> dict:
        if engine.pcm:
            files = [self.ffmpeg.cut_pcm(path, a, b, work / f"{engine.id}-{i:04d}.raw") for i, (a, b) in enumerate(parts)]
        elif len(parts) == 1:
            files = [path]
        else:
            files = [self.ffmpeg.cut(path, a, b, work / f"{engine.id}-{i:04d}{path.suffix}")
                     for i, (a, b) in enumerate(parts)]
        done = 0

        def one(item):
            nonlocal done
            index, file = item
            result = engine.transcribe(self.http, file, language, prompt)
            done += 1
            if progress and len(files) > 1:
                progress(f"{engine.title}: {done} из {len(files)} частей")
            return index, result

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            results = dict(pool.map(one, enumerate(files)))
        texts, segs, lang = [], [], None
        for index, (start, end) in enumerate(parts):
            result = results[index]
            text = result.get("text") or ""
            lang = lang or result.get("language")
            if text:
                texts.append(text)
            if result.get("segments"):
                segs += [{"start": round(start + s["start"], 2), "end": round(start + s["end"], 2), "text": s["text"]}
                         for s in result["segments"] if s["text"]]
            elif text:
                segs.append({"start": round(start, 2), "end": round(end, 2), "text": text})
        return {"text": " ".join(texts).strip(), "language": lang, "segments": segs}

    def check(self) -> list[dict]:
        """Проверка подключения: ffmpeg и каждый сервис — настроен ли, отвечает ли, принимает ли ключ."""
        checks = []
        version = self.ffmpeg.version() if self.ffmpeg.exe else ""
        checks.append({"title": "ffmpeg", "ok": bool(version),
                       "detail": version or "нет ffmpeg — переустановите плагин (пакет imageio-ffmpeg)"})
        configured = self.active()
        if not configured:
            checks.append({"title": "Сервисы распознавания", "ok": False,
                           "detail": "не задан ни один: ключ OpenAI, Groq, SpeechKit или адрес своего сервера"})
        for engine in configured:
            ok, detail = engine.check(self.http)
            checks.append({"title": f"{engine.title}: {engine.host}", "ok": ok, "detail": detail})
        return checks


def clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _stamp(seconds: float, sep: str) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def render(result: dict, fmt: str) -> str:
    """Текст расшифровки в формате text | srt | vtt | md (с метками времени)."""
    segments = result.get("segments") or []
    if fmt == "srt":
        return "\n".join(f"{i}\n{_stamp(s['start'], ',')} --> {_stamp(s['end'], ',')}\n{s['text']}\n"
                         for i, s in enumerate(segments, 1))
    if fmt == "vtt":
        return "WEBVTT\n\n" + "\n".join(f"{_stamp(s['start'], '.')} --> {_stamp(s['end'], '.')}\n{s['text']}\n"
                                         for s in segments)
    if fmt == "md":
        return "\n\n".join(f"**[{clock(s['start'])}]** {s['text']}" for s in segments) + "\n"
    return (result.get("text") or "") + "\n"
