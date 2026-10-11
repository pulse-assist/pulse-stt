"""Плагин без сети: сервисы распознавания подменены (httpx.MockTransport), звук — настоящий ffmpeg (imageio-ffmpeg).
Запуск: cd backend && pytest"""

import base64
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("PULSE_PLUGIN_DATA", tempfile.mkdtemp(prefix="stt-test-"))
os.environ.setdefault("PULSE_PLUGIN_SETTINGS", json.dumps({"order": "openai, groq", "language": "ru",
                                                           "vocabulary": "Пульс, ИТМО"}))
os.environ.setdefault("PULSE_PLUGIN_SECRETS", json.dumps({"openai_api_key": "sk-test", "groq_api_key": "gsk-test"}))

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from stt_plugin import app as module  # noqa: E402
from stt_plugin import engines as eng  # noqa: E402
from stt_plugin.audio import AudioError, Ffmpeg, plan  # noqa: E402
from stt_plugin.service import NotConfiguredError, RecognizeError, Recognizer, render  # noqa: E402
from stt_plugin.store import Store  # noqa: E402

FFMPEG = Ffmpeg()
needs_ffmpeg = pytest.mark.skipif(not FFMPEG.exe, reason="нет ffmpeg")


class FakeServices:
    """OpenAI, Groq и SpeechKit: запоминают запросы; failing — сервисы, которые отвечают ошибкой."""

    def __init__(self):
        self.calls: list[dict] = []
        self.failing: dict[str, int] = {}
        self.lock = threading.Lock()
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        service = {"api.openai.com": "openai", "api.groq.com": "groq", "stt.api.cloud.yandex.net": "yandex"}.get(host, "custom")
        with self.lock:
            index = len([c for c in self.calls if c["service"] == service])
            self.calls.append({"service": service, "path": request.url.path, "params": dict(request.url.params),
                               "auth": request.headers.get("authorization"), "size": len(request.content),
                               "body": request.content})
        if request.url.path.endswith("/models"):
            if service in self.failing:
                return httpx.Response(self.failing[service], json={"error": {"message": "bad key"}})
            return httpx.Response(200, json={"data": [{"id": "gpt-4o-transcribe"}, {"id": "whisper-large-v3-turbo"}]})
        if service in self.failing:
            return httpx.Response(self.failing[service], json={"error": {"message": "Incorrect API key"}})
        if service == "yandex":
            return httpx.Response(200, json={"result": f"часть {index + 1}"})
        verbose = b'name="response_format"\r\n\r\nverbose_json' in request.content
        text = f"{service} часть {index + 1}"
        body = {"text": text, "language": "russian"}
        if verbose:
            body["segments"] = [{"start": 0.0, "end": 1.5, "text": text}]
        return httpx.Response(200, json=body)


def tone(path: Path, seconds: int = 20, video: bool = False) -> Path:
    """Тон с паузами: 3 с звук, 1 с тишина — есть где резать."""
    args = ["-y", "-f", "lavfi", "-i", f"aevalsrc='if(lt(mod(t,4),3),0.5*sin(2*PI*440*t),0)':s=16000:d={seconds}"]
    if video:
        args += ["-f", "lavfi", "-i", f"color=c=black:s=64x64:d={seconds}", "-shortest", "-c:v", "mpeg4"]
    result = FFMPEG._run([*args, str(path)])  # noqa: SLF001
    assert result.returncode == 0, result.stderr[-500:]
    return path


def recognizer(tmp_path: Path, services: FakeServices, settings: dict | None = None, secrets: dict | None = None):
    store = Store(tmp_path / "data")
    settings = {"order": "openai, groq"} if settings is None else settings
    secrets = {"openai_api_key": "k1", "groq_api_key": "k2"} if secrets is None else secrets
    rec = Recognizer(store, settings, secrets,
                     FFMPEG, lambda: httpx.Client(transport=services.transport))
    return rec, store


# --- нарезка ------------------------------------------------------------------------------------------------

def test_plan_cuts_in_pauses():
    assert plan(30, 600, []) == [(0.0, 30)]
    parts = plan(1300, 600, [100, 590, 1150])
    assert parts == [(0.0, 590), (590, 1150), (1150, 1300)]
    hard = plan(1300, 600, [])
    assert hard == [(0.0, 600), (600, 1200), (1200, 1300)]


@needs_ffmpeg
def test_prepare_video_and_find_pauses(tmp_path):
    src = tone(tmp_path / "встреча.mp4", 12, video=True)
    prepared = FFMPEG.prepare(src, tmp_path)
    assert prepared.path.suffix in (".ogg", ".m4a") and 11.5 < prepared.duration < 12.5
    pauses = FFMPEG.silences(prepared.path)
    assert len(pauses) >= 2 and all(abs(p % 4 - 3.5) < 0.3 for p in pauses)
    (tmp_path / "bad.ogg").write_bytes(b"not audio at all")
    with pytest.raises(AudioError):
        FFMPEG.prepare(tmp_path / "bad.ogg", tmp_path)


# --- распознавание ------------------------------------------------------------------------------------------

@needs_ffmpeg
def test_short_voice_one_request_with_vocabulary(tmp_path):
    services = FakeServices()
    rec, store = recognizer(tmp_path, services, {"order": "openai", "vocabulary": "Пульс, ИТМО", "language": "ru"})
    result = rec.transcribe(tone(tmp_path / "v.webm", 5), source="telegram", name="voice.ogg")
    assert result["text"] == "openai часть 1" and result["engine"] == "openai" and result["parts"] == 1
    assert result["model"] == "OpenAI gpt-4o-transcribe" and result["tried"] == []
    [call] = services.calls
    assert call["auth"] == "Bearer k1" and b"\xd0\x9f\xd1\x83\xd0\xbb\xd1\x8c\xd1\x81" in call["body"]   # словарь — prompt
    assert b'name="language"\r\n\r\nru' in call["body"]
    [row] = store.history()
    assert row["source"] == "telegram" and row["engine"] == "openai" and row["text"] == "openai часть 1"
    [usage] = store.usage()
    assert usage["engine"] == "openai" and 4.5 < usage["seconds"] < 5.5 and usage["currency"] == "$"


@needs_ffmpeg
def test_fallback_to_next_service(tmp_path):
    services = FakeServices()
    services.failing["openai"] = 401
    rec, store = recognizer(tmp_path, services)
    result = rec.transcribe(tone(tmp_path / "v.ogg", 4))
    assert result["engine"] == "groq" and "ключ отклонён" in result["tried"][0]
    assert store.history()[0]["note"].startswith("OpenAI: ключ отклонён")
    services.failing["groq"] = 429
    with pytest.raises(RecognizeError, match="Groq: превышен лимит"):
        rec.transcribe(tone(tmp_path / "v2.ogg", 4))
    assert store.history()[0]["error"]


@needs_ffmpeg
def test_long_record_in_parts_and_subtitles(tmp_path, monkeypatch):
    services = FakeServices()
    rec, _ = recognizer(tmp_path, services, {"order": "groq"}, {"groq_api_key": "k2"})
    engines_build = eng.build

    def small(settings, secrets):
        found = engines_build(settings, secrets)
        found["groq"].max_sec = 9
        return found

    monkeypatch.setattr(eng, "build", small)
    result = rec.transcribe(tone(tmp_path / "long.ogg", 20), segments=True)
    assert result["parts"] == 3 and len(services.calls) == 3
    assert result["text"].count("groq часть") == 3
    starts = [s["start"] for s in result["segments"]]
    assert starts[0] == 0 and 7.2 < starts[1] < 7.8 and 15.2 < starts[2] < 15.8     # резали в паузах (7,5 и 15,5 с)
    srt = render(result, "srt")
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,500\n") and "-->" in srt
    assert render(result, "vtt").startswith("WEBVTT")
    assert render(result, "md").startswith("**[0:00]**")


@needs_ffmpeg
def test_yandex_gets_pcm_parts_of_25s(tmp_path):
    services = FakeServices()
    rec, store = recognizer(tmp_path, services, {"order": "yandex"}, {"yandex_api_key": "y-key"})
    result = rec.transcribe(tone(tmp_path / "v.ogg", 60), language="en")
    assert result["parts"] == 3 and result["text"].count("часть") == 3
    for call in services.calls:
        assert call["auth"] == "Api-Key y-key" and call["params"]["format"] == "lpcm"
        assert call["params"]["lang"] == "en-US" and call["size"] <= 25.5 * 32000
    assert store.usage()[0]["currency"] == "₽"


def test_nothing_configured(tmp_path):
    rec, _ = recognizer(tmp_path, FakeServices(), {}, {})
    with pytest.raises(NotConfiguredError):
        rec.transcribe(tmp_path / "x.ogg")
    assert [e.id for e in rec.engines()] == ["openai", "groq", "yandex", "custom"]
    assert eng.order({"order": "custom; yandex"}) == ["custom", "yandex", "openai", "groq"]


def test_check_reports_each_service(tmp_path):
    services = FakeServices()
    services.failing["groq"] = 401
    rec, _ = recognizer(tmp_path, services, {"order": "openai, groq, custom", "custom_url": "http://gpu:8000/v1/"})
    checks = {c["title"]: c for c in rec.check()}
    assert checks["OpenAI: api.openai.com"]["ok"] and "gpt-4o-transcribe" in checks["OpenAI: api.openai.com"]["detail"]
    assert not checks["Groq: api.groq.com"]["ok"] and "ключ отклонён" in checks["Groq: api.groq.com"]["detail"]
    assert checks["Свой сервер: gpu:8000"]["ok"]


# --- процесс плагина: протокол Пульса, страница, команды ----------------------------------------------------

@pytest.fixture
def client(monkeypatch, tmp_path):
    services = FakeServices()
    rec, store = recognizer(tmp_path, services, json.loads(os.environ["PULSE_PLUGIN_SETTINGS"]),
                            json.loads(os.environ["PULSE_PLUGIN_SECRETS"]))
    monkeypatch.setattr(module, "recognizer", rec)
    monkeypatch.setattr(module, "store", store)
    http = TestClient(module.app)
    http.services = services
    return http


@needs_ffmpeg
def test_speech_protocol_for_pulse(client, tmp_path):
    data = tone(tmp_path / "voice.ogg", 3).read_bytes()
    response = client.post("/speech/transcribe", files={"file": ("voice.ogg", data, "audio/ogg")},
                           data={"source": "telegram"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["text"] == "openai часть 1" and body["model"] == "OpenAI gpt-4o-transcribe" and body["duration_sec"] > 2
    bad = client.post("/speech/transcribe", files={"file": ("x.ogg", b"garbage", "audio/ogg")})
    assert bad.status_code == 400
    client.services.failing.update({"openai": 500, "groq": 500})
    failed = client.post("/speech/transcribe", files={"file": ("voice.ogg", data, "audio/ogg")})
    assert failed.status_code == 502 and "OpenAI" in failed.json()["detail"]


def test_speech_not_configured_is_503(monkeypatch, tmp_path):
    rec, _ = recognizer(tmp_path, FakeServices(), {}, {})
    monkeypatch.setattr(module, "recognizer", rec)
    response = TestClient(module.app).post("/speech/transcribe", files={"file": ("v.ogg", b"x", "audio/ogg")})
    assert response.status_code == 503 and "ключ" in response.json()["detail"]


@needs_ffmpeg
def test_page_state_and_try(client, tmp_path):
    state = client.get("/state").json()
    roles = {e["id"]: e["role"] for e in state["engines"]}
    assert roles == {"openai": "основная", "groq": "резервная", "yandex": None, "custom": None}
    assert state["language"] == "ru" and state["vocabulary"] and state["ffmpeg"]
    data = base64.b64encode(tone(tmp_path / "probe.webm", 2).read_bytes()).decode()
    tried = client.post("/try", json={"data": data, "filename": "probe.webm", "engine": "groq"}).json()
    assert tried["engine"] == "groq" and tried["text"] == "groq часть 1"
    state = client.get("/state").json()
    assert state["history"][0]["source"] == "page" and state["usage"][0]["engine"] == "groq"
    assert client.post("/try", json={"data": "%%%"}).status_code == 400


@needs_ffmpeg
def test_agent_command_transcribes_to_file(client, monkeypatch, tmp_path):
    saved = {}

    def fetch(arg, dest):
        shutil.copy(tone(tmp_path / "meeting.mp4", 6, video=True), dest / "meeting.mp4")
        return "meeting.mp4"

    def save(arg, text):
        saved[arg["path"]] = text
        return arg["path"]

    monkeypatch.setattr(module, "fetch", fetch)
    monkeypatch.setattr(module, "save", save)
    args = {"file": {"ticket": "t1", "name": "meeting.mp4", "path": "Встречи/meeting.mp4"}, "format": "md",
            "out": {"ticket": "t2", "name": "meeting.md", "path": "Встречи/meeting.md"}}
    result = client.post("/commands/transcribe", json=args).json()
    assert result["status"] == "done" and result["saved"] == "Встречи/meeting.md" and "text" not in result
    assert saved["Встречи/meeting.md"].startswith("**[0:00]**")
    plain = client.post("/commands/transcribe", json={"file": args["file"]}).json()
    assert plain["text"].strip() == "openai часть 2"

    # долгая: ответ «в работе» и job, результат — командой result
    gate = threading.Event()
    slow = module.recognizer.transcribe

    def wait_then(*a, **kw):
        gate.wait(5)
        return slow(*a, **kw)

    monkeypatch.setattr(module.recognizer, "transcribe", wait_then)
    running = client.post("/commands/transcribe", json={"file": args["file"], "wait": 0}).json()
    assert running["status"] == "running" and "pulse stt result" in running["next"]
    gate.set()
    done = client.post("/commands/result", json={"job": running["job"], "wait": 10}).json()
    assert done["status"] == "done"
    assert client.post("/commands/result", json={"job": "nope"}).status_code == 400


def test_commands_json_matches_code():
    names = {c["name"] for c in json.loads(Path(__file__).parents[2].joinpath("commands.json").read_text(encoding="utf-8"))}
    assert names == set(module.plugin._commands)  # noqa: SLF001
