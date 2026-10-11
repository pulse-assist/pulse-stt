"""Сеть до сервисов распознавания. Если сервис с сервера Пульса недоступен (OpenAI из России, SpeechKit из-за
рубежа), в настройках задаётся прокси — через него идут запросы к OpenAI, Groq и SpeechKit (свой сервер — напрямую).
Прокси — http://… или socks5://… (с логином и паролем в адресе); хранится секретом."""

from urllib.parse import urlsplit

import httpx

HOSTS = ("api.openai.com", "api.groq.com", "stt.api.cloud.yandex.net")
TIMEOUT = httpx.Timeout(300, connect=15)


class ProxyError(ValueError):
    pass


def proxy_of(secrets: dict) -> str | None:
    url = (secrets.get("proxy") or "").strip()
    if not url:
        return None
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https", "socks5", "socks5h") or not parts.hostname or not parts.port:
        raise ProxyError("Прокси — адрес вида http://логин:пароль@хост:порт или socks5://хост:порт")
    return url


def proxy_label(url: str | None) -> str:
    """Прокси без логина и пароля — для страницы и проверки."""
    if not url:
        return ""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname}:{parts.port}"


def build_http(secrets: dict) -> httpx.Client:
    headers = {"User-Agent": "Pulse-stt"}
    try:
        proxy = proxy_of(secrets)
    except ProxyError:
        proxy = None                                     # ошибку покажет проверка подключения
    if not proxy:
        return httpx.Client(timeout=TIMEOUT, headers=headers)
    try:
        mounts = {f"all://{host}": httpx.HTTPTransport(proxy=proxy) for host in HOSTS}
        return httpx.Client(timeout=TIMEOUT, headers=headers, mounts=mounts)
    except ImportError:                                  # socks5 без socksio — напрямую, проверка подскажет
        return httpx.Client(timeout=TIMEOUT, headers=headers)
