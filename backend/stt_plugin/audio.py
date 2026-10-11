"""Звук через ffmpeg: достать дорожку из любой записи (голосовое, webm из браузера, видео), сжать, разрезать по паузам.

ffmpeg — системный, если есть, иначе из пакета imageio-ffmpeg (статическая сборка; ставить на сервер ничего не нужно).
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000
BITRATE = "32k"                 # речь в Opus 32 кбит/с: 10 минут ≈ 2,4 МБ — под лимит 25 МБ у OpenAI и Groq
_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_TIME = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(\d+(?:\.\d+)?)")


class AudioError(Exception):
    """Запись не разобрать: нет звука, повреждена, неизвестный формат."""


@dataclass
class Prepared:
    path: Path              # моно 16 кГц: Opus в OGG или AAC в M4A
    duration: float         # секунд


def ffmpeg_path() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 — нет пакета или бинарника под эту платформу
        return None


def _seconds(match: re.Match) -> float:
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


class Ffmpeg:
    def __init__(self, exe: str | None = None, timeout: int = 1800):
        self.exe = exe or ffmpeg_path()
        self.timeout = timeout

    def _run(self, args: list[str]) -> subprocess.CompletedProcess:
        if not self.exe:
            raise AudioError("нет ffmpeg: переустановите плагин (пакет imageio-ffmpeg) или поставьте ffmpeg на сервер")
        try:
            return subprocess.run([self.exe, "-hide_banner", "-nostdin", *args], capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self.timeout)
        except subprocess.TimeoutExpired:
            raise AudioError("ffmpeg не справился за отведённое время") from None

    def version(self) -> str:
        result = self._run(["-version"])
        return (result.stdout or "").splitlines()[0] if result.returncode == 0 else ""

    def prepare(self, src: Path, workdir: Path) -> Prepared:
        """Любая запись → моно 16 кГц: Opus/OGG (нет libopus в сборке ffmpeg — AAC/M4A) и её длительность."""
        result, out = None, None
        for codec, suffix in (("libopus", ".ogg"), ("aac", ".m4a")):
            out = workdir / f"audio{suffix}"
            extra = ["-application", "voip"] if codec == "libopus" else []
            result = self._run(["-y", "-i", str(src), "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(SAMPLE_RATE),
                                "-c:a", codec, "-b:a", BITRATE, *extra, str(out)])
            err = result.stderr or ""
            if result.returncode == 0 and out.is_file() and out.stat().st_size:
                break
            if "Unknown encoder" not in err and "Encoder not found" not in err:
                break
        err = result.stderr or ""
        if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
            if "matches no streams" in err or "does not contain any stream" in err:
                raise AudioError("в файле нет звуковой дорожки")
            if "Invalid data found" in err or "could not find codec" in err.lower():
                raise AudioError("формат записи не распознан")
            tail = " ".join(err.strip().splitlines()[-2:])[:300]
            raise AudioError(f"ffmpeg не смог обработать запись: {tail or result.returncode}")
        times = list(_TIME.finditer(err))
        duration = _seconds(times[-1]) if times else self.duration(out)
        return Prepared(out, duration)

    def duration(self, path: Path) -> float:
        result = self._run(["-i", str(path)])
        match = _DURATION.search(result.stderr or "")
        return _seconds(match) if match else 0.0

    def silences(self, path: Path, noise_db: int = -35, min_sec: float = 0.35) -> list[float]:
        """Середины пауз, секунд от начала — места, где резать запись, не разрывая слов."""
        result = self._run(["-i", str(path), "-af", f"silencedetect=noise={noise_db}dB:d={min_sec}", "-f", "null", "-"])
        starts = [float(m.group(1)) for m in _SILENCE_START.finditer(result.stderr or "")]
        ends = [float(m.group(1)) for m in _SILENCE_END.finditer(result.stderr or "")]
        return [(max(a, 0.0) + b) / 2 for a, b in zip(starts, ends)]

    def cut(self, path: Path, start: float, end: float, out: Path) -> Path:
        result = self._run(["-y", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}", "-c", "copy",
                            str(out)])
        if result.returncode != 0 or not out.is_file():
            raise AudioError("не удалось разрезать запись на части")
        return out


    def cut_pcm(self, path: Path, start: float, end: float, out: Path) -> Path:
        """Часть записи несжатой: 16 кГц, 16 бит, моно (SpeechKit принимает её без кодеков)."""
        result = self._run(["-y", "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{end - start:.3f}", "-f", "s16le",
                            "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", str(out)])
        if result.returncode != 0 or not out.is_file():
            raise AudioError("не удалось подготовить часть записи")
        return out


def plan(duration: float, max_sec: float, pauses: list[float]) -> list[tuple[float, float]]:
    """Части не длиннее max_sec; режем в паузе ближе к концу части, пауз нет — по max_sec."""
    if duration <= max_sec:
        return [(0.0, duration)]
    slack = min(max(max_sec * 0.25, 3.0), 90.0)
    parts, pos = [], 0.0
    while duration - pos > max_sec:
        target = pos + max_sec
        near = [p for p in pauses if target - slack <= p <= target]
        cut = max(near) if near else target
        parts.append((pos, cut))
        pos = cut
    parts.append((pos, duration))
    return parts
