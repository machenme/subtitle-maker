"""
Bing Translator web backend.

The public Edge token endpoint used by the original implementation was
removed.  Bing Translator still exposes the same internal web translation
service, which provides a short-lived anti-abuse token from its translator
page and requires no subscription key.

``EdgeTranslator`` remains the public class name for compatibility with the
CLI and GUI.  The project uses ``python-dotenv`` to persist the short-lived
session credentials without putting them in source control.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from dotenv import dotenv_values, set_key

from src.translator.types import TranslationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRANSLATOR_URL = "https://www.bing.com/Translator"
_DEFAULT_TOKEN_TTL = 480  # 8 minutes
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)

_LANGUAGE_MAP = {
    "auto": "auto-detect",
    "zh": "zh-Hans",
    "zh-cn": "zh-Hans",
    "zh-hans": "zh-Hans",
    "zh-tw": "zh-Hant",
    "zh-hant": "zh-Hant",
}


class EdgeTranslator:
    """
    Bing Translator web backend exposed through the legacy class name.

    The translator page credentials are cached in the repository's ``.env``.
    Rate-limit cooldown is shared by all instances so concurrent GUI/CLI
    workers do not immediately repeat a throttled request.
    """

    _rate_lock = threading.Lock()
    _rate_limit_until: float = 0.0

    def __init__(
        self,
        token_ttl: int = _DEFAULT_TOKEN_TTL,
        env_path: str | Path | None = None,
    ) -> None:
        # ``token_ttl`` is retained for API compatibility.  Bing publishes
        # its own TTL, but this cap keeps long-running jobs refreshable.
        self._token_ttl = token_ttl
        self._env_path = (
            Path(env_path)
            if env_path
            else Path(__file__).resolve().parents[2] / ".env"
        )
        self._token_value: str | None = None
        self._token_key: str | None = None
        self._token_expires: float = 0.0
        self._lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _USER_AGENT})
        self._host_url = _TRANSLATOR_URL
        self._api_url: str | None = None
        self._ig: str | None = None
        self._iid: str | None = None

    # ------------------------------------------------------------------
    # TranslationProvider protocol
    # ------------------------------------------------------------------

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "zh"
    ) -> str:
        """Translate *text* through Bing Translator's web endpoint."""
        if not text:
            return ""

        last_exc: Exception | None = None
        auth_refreshes = 0

        for attempt in range(_MAX_RETRIES + 1):
            with EdgeTranslator._rate_lock:
                wait = EdgeTranslator._rate_limit_until - time.time()
            if wait > 0:
                logger.debug("Rate-limit cooldown: waiting %.1fs", wait)
                time.sleep(wait)

            try:
                token = self._get_token()
                response = self._post_translation(
                    text,
                    source_lang,
                    target_lang,
                    token,
                )
            except requests.RequestException as exc:
                last_exc = TranslationError(f"Bing translate request failed: {exc}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                continue

            if response.status_code in (401, 403) and auth_refreshes == 0:
                logger.info("Bing token rejected (%d), refreshing", response.status_code)
                auth_refreshes += 1
                self._fetch_fresh_token()
                continue

            if response.status_code == 429:
                backoff = _RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "Bing translate rate-limited, backoff %.1fs (attempt %d/%d)",
                    backoff,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                with EdgeTranslator._rate_lock:
                    EdgeTranslator._rate_limit_until = time.time() + backoff
                last_exc = TranslationError("Bing translate rate-limited (429)")
                time.sleep(backoff)
                continue

            if not response.ok:
                snippet = response.text[:200]
                raise TranslationError(
                    f"Bing translate HTTP {response.status_code}: {snippet}"
                )

            try:
                data = response.json()
                return data[0]["translations"][0]["text"]
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise TranslationError(
                    f"Invalid response format from Bing Translator: {exc}"
                ) from exc

        raise last_exc or TranslationError("Bing translate failed after retries")

    def translate_batch(
        self, texts: list[str], source_lang: str = "auto", target_lang: str = "zh"
    ) -> list[str]:
        """Batch translate (default: one request per text)."""
        return [self.translate(t, source_lang, target_lang) for t in texts]

    # ------------------------------------------------------------------
    # Request construction
    # ------------------------------------------------------------------

    def _post_translation(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        token: str,
    ) -> requests.Response:
        if not self._api_url or not self._token_key or not self._ig or not self._iid:
            raise TranslationError("Bing Translator session is not initialized")

        host = urlparse(self._host_url)
        headers = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "Origin": f"{host.scheme}://{host.netloc}",
            "Referer": self._host_url,
            "X-Requested-With": "XMLHttpRequest",
        }
        payload = {
            "text": text,
            "fromLang": _map_language(source_lang),
            "to": _map_language(target_lang),
            "tryFetchingGenderDebiasedTranslations": "true",
            "key": self._token_key,
            "token": token,
        }
        return self._session.post(
            self._api_url,
            params={"isVertical": "1", "IG": self._ig, "IID": self._iid},
            data=payload,
            headers=headers,
            timeout=30,
        )

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Return a valid in-memory/.env token or fetch a fresh page."""
        with self._lock:
            if self._token_value is not None and time.time() < self._token_expires:
                return self._token_value

        # Avoid multiple concurrent workers fetching separate page tokens.
        with self._refresh_lock:
            with self._lock:
                if self._token_value is not None and time.time() < self._token_expires:
                    return self._token_value
                cached = self._load_cached_credentials()
                if cached is not None:
                    return cached
            return self._fetch_fresh_token()

    def _load_cached_credentials(self) -> str | None:
        """Load fresh Bing credentials from ``.env`` if all fields exist."""
        try:
            values = dotenv_values(self._env_path)
        except (OSError, ValueError) as exc:
            logger.warning("Unable to read translation cache %s: %s", self._env_path, exc)
            return None

        raw_timestamp = values.get("TRANSLATE_KEY_TIMESTAMP")
        try:
            timestamp = float(raw_timestamp) if raw_timestamp is not None else 0.0
        except (TypeError, ValueError):
            return None

        age = time.time() - timestamp
        if timestamp <= 0 or age < 0 or age >= self._token_ttl:
            return None

        token = _env_value(values.get("TRANSLATE_TOKEN"))
        key = _env_value(values.get("TRANSLATE_KEY"))
        ig = _env_value(values.get("TRANSLATE_IG"))
        iid = _env_value(values.get("TRANSLATE_IID"))
        host_url = _env_value(values.get("TRANSLATE_HOST_URL"))
        if not all((token, key, ig, iid, host_url)):
            return None

        self._token_value = token
        self._token_key = key
        self._ig = ig
        self._iid = iid
        self._host_url = host_url
        self._api_url = re.sub(r"/translator$", "/ttranslatev3", host_url, flags=re.I)
        self._token_expires = timestamp + self._token_ttl
        logger.debug("Loaded Bing translation credentials from .env (age=%.0fs)", age)
        return token

    def _fetch_fresh_token(self) -> str:
        """Fetch and cache the token embedded in the translator page."""
        try:
            response = self._session.get(_TRANSLATOR_URL, timeout=15)
        except requests.RequestException as exc:
            raise TranslationError(f"Bing auth request failed: {exc}") from exc

        if not response.ok:
            raise TranslationError(
                f"Bing auth page HTTP {response.status_code}: {response.text[:200]}"
            )

        html = response.text
        try:
            token_data = re.search(
                r"var params_AbusePreventionHelper = (.*?);", html
            )
            if not token_data:
                raise ValueError("anti-abuse token not found")
            params = json.loads(token_data.group(1))
            if len(params) < 2:
                raise ValueError("anti-abuse token is incomplete")

            ig_match = re.search(r'IG:"([^"]+)"', html)
            iid_match = re.search(
                r'id="tta_outGDCont"[^>]*data-iid="([^"]+)"', html
            )
            if not iid_match:
                iid_match = re.search(r'data-iid="([^"]+)"', html)
            if not ig_match or not iid_match:
                raise ValueError("translation request identifiers not found")
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise TranslationError(f"Invalid Bing auth page: {exc}") from exc

        self._host_url = response.url.rstrip("/")
        self._api_url = re.sub(r"/translator$", "/ttranslatev3", self._host_url, flags=re.I)
        self._ig = ig_match.group(1)
        self._iid = iid_match.group(1)
        ttl = self._token_ttl
        if len(params) >= 3:
            try:
                ttl = min(ttl, max(60, int(params[2]) // 1000 - 60))
            except (TypeError, ValueError):
                pass

        token = str(params[1])
        timestamp = time.time()
        with self._lock:
            self._token_key = str(params[0])
            self._token_value = token
            self._token_expires = timestamp + ttl
            self._persist_credentials(timestamp)
        logger.debug("Bing token refreshed (TTL=%ds)", ttl)
        return token

    def _persist_credentials(self, timestamp: float) -> None:
        """Persist the current Bing session in the ignored ``.env`` file."""
        values = {
            "TRANSLATE_KEY": self._token_key,
            "TRANSLATE_TOKEN": self._token_value,
            "TRANSLATE_KEY_TIMESTAMP": str(int(timestamp)),
            "TRANSLATE_IG": self._ig,
            "TRANSLATE_IID": self._iid,
            "TRANSLATE_HOST_URL": self._host_url,
        }
        try:
            for key, value in values.items():
                if value is not None:
                    set_key(str(self._env_path), key, value, quote_mode="auto")
        except OSError as exc:
            # A read-only checkout should still be able to translate once.
            logger.warning("Unable to persist translation credentials: %s", exc)


def _map_language(language: str) -> str:
    """Map the project's ISO code to Bing Translator's language code."""
    normalized = (language or "auto").strip().lower()
    return _LANGUAGE_MAP.get(normalized, language)


def _env_value(value: object) -> str | None:
    """Normalize a value read by python-dotenv."""
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
