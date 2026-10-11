"""Сервисы распознавания: OpenAI, Groq и свой сервер — один протокол (/audio/transcriptions), Yandex SpeechKit — свой.

Каждый сервис знает, как длинную часть записи он принимает (max_sec), отдаёт ли время фраз (segments) и сколько
примерно стоит минута. Ошибка сервиса — EngineError: распознавание переходит к следующему по порядку.
"""

from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

TIMEOUT = httpx.Timeout(300, connect=15)
CHECK_TIMEOUT = httpx.Timeout(15, connect=10)
OPENAI_URL = "https://api.openai.com/v1"
GROQ_URL = "https://api.groq.com/openai/v1"
YANDEX_URL = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
OPENAI_MODELS = ("gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1")
GROQ_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
# $ за минуту по прайсу сервисов (10.2026) — оценка для страницы; точный счёт — в кабинете сервиса
PRICE_USD = {"gpt-4o-transcribe": 0.006, "gpt-4o-mini-transcribe": 0.003, "whisper-1": 0.006,
             "whisper-large-v3-turbo": 0.04 / 60, "whisper-large-v3": 0.111 / 60}
YANDEX_RUB_PER_15S = 0.16
YANDEX_LANGS = {"ru": "ru-RU", "en": "en-US", "de": "de-DE", "fr": "fr-FR", "es": "es-ES", "it": "it-IT",
                "pl": "pl-PL", "tr": "tr-TR", "kk": "kk-KZ", "uz": "uz-UZ", "he": "he-IL", "pt": "pt-PT"}


class EngineError(Exception):
    """Сервис не распознал: ключ, лимит, сеть, ответ — пробуем следующий."""


@dataclass
class Engine:
    id: str                          # openai | groq | yandex | custom
    title: str
    model: str
    configured: bool
    missing: str = ""                # чего не хватает в настройках
    max_sec: float = 600
    segments: bool = False           # отдаёт время фраз (для субтитров)
    pcm: bool = False                # принимает несжатый звук (s16le 16 кГц), а не файл
    host: str = ""
    extra: dict = field(default_factory=dict)

    def out(self) -> dict:
        return {"id": self.id, "title": self.title, "model": self.model, "configured": self.configured,
                "missing": self.missing, "host": self.host}

    def cost(self, seconds: float) -> tuple[float, str] | None:
        if self.id == "yandex":
            return -(-seconds // 15) * YANDEX_RUB_PER_15S, "₽"
        price = PRICE_USD.get(self.model)
        return (price * seconds / 60, "$") if price and self.id != "custom" else None

    def transcribe(self, http: httpx.Client, path: Path, language: str | None, prompt: str | None) -> dict:
        raise NotImplementedError

    def check(self, http: httpx.Client) -> tuple[bool, str]:
        raise NotImplementedError


def _fail(response: httpx.Response, title: str) -> EngineError:
    try:
        payload = response.json()
        err = payload.get("error", payload) if isinstance(payload, dict) else payload
        detail = (err.get("message") or err.get("error_message") or err) if isinstance(err, dict) else err
    except ValueError:
        detail = response.text
    detail = str(detail)[:300]
    if response.status_code in (401, 403):
        return EngineError(f"{title}: ключ отклонён ({response.status_code}) — проверьте его в настройках плагина")
    if response.status_code == 429:
        return EngineError(f"{title}: превышен лимит запросов или кончились деньги на счёте — {detail}")
    return EngineError(f"{title}: HTTP {response.status_code} — {detail}")


def _post(http: httpx.Client, title: str, url: str, **kwargs) -> httpx.Response:
    try:
        response = http.post(url, timeout=TIMEOUT, **kwargs)
    except httpx.TimeoutException:
        raise EngineError(f"{title}: не ответил вовремя") from None
    except httpx.HTTPError as exc:
        raise EngineError(f"{title}: сеть — {exc.__class__.__name__}") from None
    if response.status_code >= 400:
        raise _fail(response, title)
    return response


class OpenAICompatible(Engine):
    """OpenAI, Groq и любой сервер с /audio/transcriptions (faster-whisper-server, speaches, vLLM, LocalAI)."""

    def transcribe(self, http, path, language, prompt):
        verbose = not self.model.startswith("gpt-4o")        # у gpt-4o-*-transcribe нет времени фраз
        form = {"model": self.model, "response_format": "verbose_json" if verbose else "json"}
        if language:
            form["language"] = language
        if prompt:
            form["prompt"] = prompt[:800]
        headers = {"Authorization": f"Bearer {self.extra['key']}"} if self.extra.get("key") else {}
        with path.open("rb") as handle:
            mime = "audio/mp4" if path.suffix == ".m4a" else "audio/ogg"
            response = _post(http, self.title, f"{self.extra['url']}/audio/transcriptions", headers=headers,
                             data=form, files={"file": (path.name, handle, mime)})
        try:
            data = response.json()
        except ValueError:
            return {"text": response.text.strip()}
        segments = [{"start": float(s.get("start", 0)), "end": float(s.get("end", 0)), "text": str(s.get("text", "")).strip()}
                    for s in data.get("segments") or [] if isinstance(s, dict)]
        return {"text": str(data.get("text") or "").strip(), "language": data.get("language"), "segments": segments}

    def check(self, http):
        headers = {"Authorization": f"Bearer {self.extra['key']}"} if self.extra.get("key") else {}
        try:
            response = http.get(f"{self.extra['url']}/models", headers=headers, timeout=CHECK_TIMEOUT)
        except httpx.TimeoutException:
            return False, "нет ответа за 15 с — сервис недоступен с сервера Пульса (нужен прокси?)"
        except httpx.HTTPError as exc:
            return False, f"сеть: {exc.__class__.__name__}"
        if response.status_code in (401, 403):
            return False, f"ключ отклонён ({response.status_code})"
        if response.status_code >= 400 and self.id != "custom":
            return False, f"HTTP {response.status_code}"
        if self.id != "custom":
            ids = {m.get("id") for m in (response.json().get("data") or []) if isinstance(m, dict)}
            if ids and self.model not in ids:
                return False, f"ключ работает, но модели {self.model} у него нет"
        return True, f"отвечает, модель {self.model}"


class Yandex(Engine):
    """Yandex SpeechKit, синхронное распознавание: часть до 30 с и 1 МБ — несжатый звук 16 кГц."""

    def _params(self, language: str | None) -> dict:
        params = {"topic": self.model or "general", "format": "lpcm", "sampleRateHertz": "16000",
                  "lang": YANDEX_LANGS.get((language or "ru").lower()[:2], "ru-RU")}
        if self.extra.get("folder"):
            params["folderId"] = self.extra["folder"]
        return params

    def transcribe(self, http, path, language, prompt):
        response = _post(http, self.title, YANDEX_URL, params=self._params(language), content=path.read_bytes(),
                         headers={"Authorization": f"Api-Key {self.extra['key']}"})
        return {"text": str(response.json().get("result") or "").strip(), "language": language}

    def check(self, http):
        silence = bytes(16000 * 2)                       # секунда тишины
        try:
            response = http.post(YANDEX_URL, params=self._params("ru"), content=silence, timeout=CHECK_TIMEOUT,
                                 headers={"Authorization": f"Api-Key {self.extra['key']}"})
        except httpx.TimeoutException:
            return False, "нет ответа за 15 с — SpeechKit недоступен с сервера Пульса (нужен прокси?)"
        except httpx.HTTPError as exc:
            return False, f"сеть: {exc.__class__.__name__}"
        if response.status_code >= 400:
            return False, str(_fail(response, "SpeechKit")).removeprefix("SpeechKit: ")
        return True, "отвечает (пробное распознавание секунды тишины)"


def _url(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    if value.endswith("/audio/transcriptions"):
        value = value[: -len("/audio/transcriptions")]
    return value


def build(settings: dict, secrets: dict) -> dict[str, Engine]:
    """Все сервисы с их настройками — и настроенные, и нет (страница показывает, чего не хватает)."""
    openai_key, groq_key = (secrets.get("openai_api_key") or "").strip(), (secrets.get("groq_api_key") or "").strip()
    yandex_key = (secrets.get("yandex_api_key") or "").strip()
    custom_url = _url(settings.get("custom_url") or "")
    custom_model = (settings.get("custom_model") or "").strip() or "whisper-1"
    engines = [
        OpenAICompatible("openai", "OpenAI", settings.get("openai_model") or OPENAI_MODELS[0], bool(openai_key),
                         "" if openai_key else "API-ключ OpenAI", 600, host="api.openai.com",
                         extra={"key": openai_key, "url": OPENAI_URL}),
        OpenAICompatible("groq", "Groq", settings.get("groq_model") or GROQ_MODELS[0], bool(groq_key),
                         "" if groq_key else "API-ключ Groq", 600, segments=True, host="api.groq.com",
                         extra={"key": groq_key, "url": GROQ_URL}),
        Yandex("yandex", "Yandex SpeechKit", "general", bool(yandex_key), "" if yandex_key else "API-ключ SpeechKit", 25,
               pcm=True, host="stt.api.cloud.yandex.net",
               extra={"key": yandex_key, "folder": (settings.get("yandex_folder_id") or "").strip()}),
        OpenAICompatible("custom", "Свой сервер", custom_model, bool(custom_url), "" if custom_url else "адрес сервера",
                         600, segments=True, host=urlsplit(custom_url).netloc if custom_url else "",
                         extra={"key": (secrets.get("custom_api_key") or "").strip(), "url": custom_url}),
    ]
    for engine in engines:
        if engine.id in ("openai",):
            engine.segments = not engine.model.startswith("gpt-4o")
    return {e.id: e for e in engines}


def order(settings: dict) -> list[str]:
    """Порядок из настроек «openai, groq»; чего там нет — следом в обычном порядке."""
    wanted = [p.strip().lower() for p in (settings.get("order") or "").replace(";", ",").split(",") if p.strip()]
    known = ["openai", "groq", "yandex", "custom"]
    return [p for p in dict.fromkeys(wanted) if p in known] + [p for p in known if p not in wanted]
