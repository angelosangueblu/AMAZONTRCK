import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import keepa
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from telegram.ext import ApplicationBuilder, ContextTypes
from urllib3.util.retry import Retry


# ---------------- CONFIG ----------------
@dataclass(frozen=True)
class Config:
    bot_token: str
    keepa_key: str
    channel_id: int
    affiliate_tag: str
    interval_seconds: int = 300
    first_delay_seconds: int = 20
    state_file: str = "sent_asins.json"


def load_config() -> Config:
    bot_token = os.getenv("BOT_TOKEN")
    keepa_key = os.getenv("KEEPA_KEY")
    channel_id = os.getenv("CHANNEL_ID")
    affiliate_tag = os.getenv("AFFILIATE_TAG", "dragonofferte-21")

    missing = [
        name
        for name, value in {
            "BOT_TOKEN": bot_token,
            "KEEPA_KEY": keepa_key,
            "CHANNEL_ID": channel_id,
        }.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            "Variabili d'ambiente mancanti: {}. Configura prima il bot.".format(
                ", ".join(missing)
            )
        )

    return Config(
        bot_token=bot_token,
        keepa_key=keepa_key,
        channel_id=int(channel_id),
        affiliate_tag=affiliate_tag,
    )


def build_http_session() -> requests.Session:
    retry_kwargs: Dict[str, Any] = {
        "total": 4,
        "read": 4,
        "connect": 4,
        "backoff_factor": 0.7,
        "status_forcelist": (429, 500, 502, 503, 504),
    }

    # Compatibilità urllib3 vecchio/nuovo
    try:
        retry = Retry(allowed_methods=frozenset(["GET"]), **retry_kwargs)
    except TypeError:
        retry = Retry(method_whitelist=frozenset(["GET"]), **retry_kwargs)

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


setup_logging()
logger = logging.getLogger("amazon_deals_bot")


# -------- STATO ASIN INVIATI --------
def load_sent_asins(path: str) -> Set[str]:
    state_path = Path(path)
    if not state_path.exists():
        return set()

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return set(str(x) for x in data)
    except Exception as exc:
        logger.warning("Impossibile leggere lo stato ASIN: %s", exc)

    return set()


def save_sent_asins(path: str, asins: Set[str]) -> None:
    Path(path).write_text(
        json.dumps(sorted(asins), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_price_to_float(price_text: Optional[str]) -> Optional[float]:
    if not price_text:
        return None

    normalized = (
        price_text.replace("€", "")
        .replace("EUR", "")
        .replace("\xa0", " ")
        .strip()
    )

    matches = re.findall(r"\d+[\.,]?\d*", normalized)
    if not matches:
        return None

    number = matches[-1].replace(".", "").replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


def format_eur(value: float) -> str:
    text = "{:.2f}".format(value)
    return text.replace(".", ",") + "€"


def compute_discount_percent(current_price: Optional[str], old_price: Optional[str], fallback: Optional[float]) -> Optional[float]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is not None and old is not None and old > 0 and old > current:
        return round(((old - current) / old) * 100.0, 1)

    if fallback is not None:
        return round(float(fallback), 1)

    return None


def discount_badge(discount_percent: Optional[float]) -> str:
    if discount_percent is None:
        return "💥 PREZZO INTERESSANTE"

    if discount_percent > 40:
        return "❌ POSSIBILE ERRORE ❌"

    if 15 <= discount_percent <= 20:
        return "💣 PREZZO BOMBA 💣"

    if discount_percent < 15:
        return "🔥 PREZZO AFFARE 🔥"

    return "‼️ SUPER PREZZO ‼️"


# -------- PRENDI DATI AMAZON --------
def get_amazon_data(
    session: requests.Session,
    asin: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Linux; Android 10; SM-G973F) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120 Mobile Safari/537.36"
        ),
        "Accept-Language": "it-IT,it;q=0.9",
    }

    url = "https://www.amazon.it/gp/aw/d/{}".format(asin)
    response = session.get(url, headers=headers, timeout=(8, 20))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_tag = soup.select_one("#productTitle") or soup.select_one("h1.a-size-large")
    title = title_tag.get_text(strip=True) if title_tag else "Offerta Amazon"

    price = None
    current_price_selectors = [
        "span.a-price span.a-offscreen",
        "span.a-offscreen",
        "#corePriceDisplay_desktop_feature_div span.a-offscreen",
        "#priceblock_ourprice",
        "#priceblock_dealprice",
    ]

    for selector in current_price_selectors:
        tag = soup.select_one(selector)
        if tag:
            text = tag.get_text(strip=True)
            if text and "€" in text:
                price = text
                break

    old_price = None
    old_price_selectors = [
        "span.a-price.a-text-price span.a-offscreen",
        "#corePriceDisplay_desktop_feature_div .a-text-price .a-offscreen",
        ".basisPrice .a-offscreen",
        "span.priceBlockStrikePriceString",
    ]

    for selector in old_price_selectors:
        tag = soup.select_one(selector)
        if tag:
            text = tag.get_text(strip=True)
            if text and "€" in text:
                old_price = text
                break

    image_selectors = [
        "img#landingImage",
        "#imgTagWrapperId img",
        "img.a-dynamic-image",
        "meta[property='og:image']",
    ]
    image_url = None

    for selector in image_selectors:
        tag = soup.select_one(selector)
        if not tag:
            continue

        candidate = tag.get("content") if tag.name == "meta" else tag.get("src")
        if candidate:
            image_url = candidate
            break

    if image_url and image_url.startswith("/"):
        image_url = "https://m.media-amazon.com" + image_url

    return title, price, old_price, image_url


def download_image_bytes(session: requests.Session, url: str) -> Optional[BytesIO]:
    try:
        response = session.get(url, timeout=(8, 20))
        response.raise_for_status()
        image_bytes = BytesIO(response.content)
        image_bytes.name = "prodotto.jpg"
        image_bytes.seek(0)
        return image_bytes
    except Exception as exc:
        logger.warning("Download immagine fallito: %s", exc)
        return None


def pick_best_deal(deals: Dict[str, Any], sent_asins: Set[str]) -> Optional[Dict[str, Any]]:
    items = deals.get("dr") or []
    for item in items:
        asin = item.get("asin")
        if asin and asin not in sent_asins:
            return item

    if items and items[0].get("asin"):
        return items[0]
    return None


def extract_keepa_discount_percent(deal_item: Dict[str, Any]) -> Optional[float]:
    for key in ("deltaPercent", "delta", "savingsPercent"):
        value = deal_item.get(key)
        if isinstance(value, (int, float)):
            if value > 100:
                value = value / 100.0
            return float(value)
    return None


def build_caption(
    title: str,
    price: str,
    old_price: Optional[str],
    link: str,
    discount_percent: Optional[float],
) -> str:
    badge = discount_badge(discount_percent)

    if old_price:
        price_line = "⭕{} anziché {}".format(price, old_price)
    else:
        price_line = "⭕{}".format(price)

    discount_line = ""
    if discount_percent is not None:
        discount_line = "\n📉 Sconto: -{}%".format(str(discount_percent).replace(".", ","))

    return "{}\n{}\n{}{}\n{}".format(
        badge,
        title,
        price_line,
        discount_line,
        link,
    )

def pick_best_asin(deals: Dict[str, Any], sent_asins: Set[str]) -> Optional[str]:
    items = deals.get("dr") or []
    for item in items:
        asin = item.get("asin")
        if asin and asin not in sent_asins:
            return asin

    if items and items[0].get("asin"):
        return items[0]["asin"]
    return None

# ---------------- AUTO OFFERTE ----------------
async def auto_offers(context: ContextTypes.DEFAULT_TYPE) -> None:
    loop = asyncio.get_running_loop()
    cfg = context.application.bot_data["config"]
    api = context.application.bot_data["keepa_api"]
    session = context.application.bot_data["http_session"]

    try:
        deals = await loop.run_in_executor(
            None,
            lambda: api.deals(
                {
                    "domainId": 8,
                    "priceTypes": [0],
                    "deltaPercentRange": [15, 90],
                }
            ),
        )

        if not deals or "dr" not in deals:
            logger.info("Nessuna offerta trovata")
            return

        sent_asins = load_sent_asins(cfg.state_file)
        deal_item = pick_best_deal(deals, sent_asins)

        if not deal_item:
            logger.info("Nessun ASIN valido trovato")
            return

        asin = deal_item.get("asin")
        if not asin:
            logger.info("Deal senza ASIN")
            return

        logger.info("ASIN selezionato: %s", asin)

        title, price, old_price, image_url = await loop.run_in_executor(
            None, lambda: get_amazon_data(session, asin)
        )

        if not price:
            logger.warning("Prezzo non trovato per %s", asin)
            return

        keepa_discount = extract_keepa_discount_percent(deal_item)
        discount_percent = compute_discount_percent(price, old_price, keepa_discount)

        link = "https://www.amazon.it/dp/{}?tag={}".format(asin, cfg.affiliate_tag)
        caption = build_caption(title, price, old_price, link, discount_percent)

        sent = False
        if image_url:
            image_bytes = await loop.run_in_executor(
                None, lambda: download_image_bytes(session, image_url)
            )
            if image_bytes:
                await context.bot.send_photo(
                    chat_id=cfg.channel_id,
                    photo=image_bytes,
                    caption=caption,
                )
                sent = True

        if not sent:
            await context.bot.send_message(chat_id=cfg.channel_id, text=caption)

        sent_asins.add(asin)
        save_sent_asins(cfg.state_file, sent_asins)
        logger.info("Offerta inviata correttamente")

    except Exception:
        logger.exception("Errore durante l'invio offerta")


# ---------------- AVVIO ----------------
def main() -> None:
    cfg = load_config()
    logger.info("Avvio bot offerte Amazon")

    app = ApplicationBuilder().token(cfg.bot_token).build()
    app.bot_data["config"] = cfg
    app.bot_data["keepa_api"] = keepa.Keepa(cfg.keepa_key)
    app.bot_data["http_session"] = build_http_session()

    if app.job_queue is None:
        raise RuntimeError(
            "JobQueue non disponibile. Installa python-telegram-bot con extra job-queue: pip install 'python-telegram-bot[job-queue]'"
        )

    app.job_queue.run_repeating(
        auto_offers,
        interval=cfg.interval_seconds,
        first=cfg.first_delay_seconds,
    )
    app.run_polling()


if __name__ == "__main__":
    main()
