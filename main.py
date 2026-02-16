import asyncio
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    max_deals_per_cycle: int = 40
    max_keepa_queries_per_cycle: int = 8
    min_discount_percent: float = 10.0
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


def resolve_state_path(path: str) -> str:
    state_path = Path(path)
    if state_path.is_absolute():
        return str(state_path)

    # Mantiene lo stato stabile anche dopo restart/cwd diversi
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / state_path)


# -------- STATO ASIN INVIATI --------
def normalize_asin(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    asin = value.strip().upper()
    if re.fullmatch(r"[A-Z0-9]{10}", asin):
        return asin
    return None


def load_sent_asins(path: str) -> Set[str]:
    state_path = Path(resolve_state_path(path))
    if not state_path.exists():
        return set()

    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        raw_asins: List[Any] = []
        if isinstance(data, list):
            raw_asins = data
        elif isinstance(data, dict) and isinstance(data.get("sent_asins"), list):
            raw_asins = data["sent_asins"]

        deduped: List[str] = []
        seen: Set[str] = set()
        for item in raw_asins:
            asin = normalize_asin(item)
            if asin and asin not in seen:
                deduped.append(asin)
                seen.add(asin)

        return set(deduped)
    except Exception as exc:
        logger.warning("Impossibile leggere lo stato ASIN: %s", exc)

    return set()


def save_sent_asins(path: str, asins: Set[str]) -> None:
    cleaned = [asin for asin in sorted(asins) if normalize_asin(asin)]
    Path(path).write_text(
        json.dumps({"sent_asins": cleaned}, ensure_ascii=False, indent=2),
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




def keepa_cents_to_eur_text(value: Any) -> Optional[str]:
    if not isinstance(value, (int, float)):
        return None

    if value <= 0:
        return None

    return format_eur(float(value) / 100.0)


def extract_price_from_keepa_product(product: Dict[str, Any]) -> Optional[str]:
    stats = product.get("stats") if isinstance(product, dict) else None
    if not isinstance(stats, dict):
        return None

    # Priorità: buybox/current amazon/new
    candidates: List[Any] = []
    for key in ("buyBoxPrice", "current"):
        value = stats.get(key)
        if isinstance(value, (list, tuple)):
            candidates.extend(list(value))
        else:
            candidates.append(value)

    for candidate in candidates:
        text = keepa_cents_to_eur_text(candidate)
        if text:
            return text

    return None


def has_valid_current_price(price_text: Optional[str]) -> bool:
    value = parse_price_to_float(price_text)
    return value is not None and value > 0

def compute_discount_percent(current_price: Optional[str], old_price: Optional[str], fallback: Optional[float]) -> Optional[float]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is not None and old is not None and old > 0 and old > current:
        return round(((old - current) / old) * 100.0, 1)

    if fallback is not None:
        return round(float(fallback), 1)

    return None


def discount_badge(discount_percent: Optional[float]) -> str:
    if discount_percent >= 15:
        return "🛍️ OFFERTA FLASH🛍️"

    if discount_percent >= 70:
        return "🚨 ERRORE DI PREZZO 🚨"

    if discount_percent >= 50:
        return "💣 SCONTO PAZZESCO 💣"

    if discount_percent >= 40:
        return "🔥 PREZZO BOMBA 🔥"

    if discount_percent >= 30:
        return "✅ SUPER OFFERTA ✅"

    return "✨ PREZZO SCONTATO ✨"


def is_placeholder_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in {"", "offerta amazon", "galleria prodotti"}


def is_bad_image_url(image_url: Optional[str]) -> bool:
    if not image_url:
        return True

    lowered = image_url.lower()
    # Evita immagini placeholder generiche (es. logo Prime) nei post.
    bad_markers = (
        "prime",
        "nav-sprite",
        "amazon-logo",
        "icon",
        "sprite",
    )
    return any(marker in lowered for marker in bad_markers)


def sanitize_old_price(current_price: Optional[str], old_price: Optional[str]) -> Optional[str]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is None or old is None:
        return None

    if old <= current:
        return None

    return format_eur(old)


def is_placeholder_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in {"", "offerta amazon", "galleria prodotti"}


def is_bad_image_url(image_url: Optional[str]) -> bool:
    if not image_url:
        return True

    lowered = image_url.lower()
    # Evita immagini placeholder generiche (es. logo Prime) nei post.
    bad_markers = (
        "prime",
        "nav-sprite",
        "amazon-logo",
        "icon",
        "sprite",
    )
    return any(marker in lowered for marker in bad_markers)


def sanitize_old_price(current_price: Optional[str], old_price: Optional[str]) -> Optional[str]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is None or old is None:
        return None

    if old <= current:
        return None

    return format_eur(old)


def is_placeholder_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in {"", "offerta amazon", "galleria prodotti"}


def is_bad_image_url(image_url: Optional[str]) -> bool:
    if not image_url:
        return True

    lowered = image_url.lower()
    # Evita immagini placeholder generiche (es. logo Prime) nei post.
    bad_markers = (
        "prime",
        "nav-sprite",
        "amazon-logo",
        "icon",
        "sprite",
    )
    return any(marker in lowered for marker in bad_markers)


def sanitize_old_price(current_price: Optional[str], old_price: Optional[str]) -> Optional[str]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is None or old is None:
        return None

    if old <= current:
        return None

    return format_eur(old)


def is_placeholder_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in {"", "offerta amazon", "galleria prodotti"}


def is_bad_image_url(image_url: Optional[str]) -> bool:
    if not image_url:
        return True

    lowered = image_url.lower()
    # Evita immagini placeholder generiche (es. logo Prime) nei post.
    bad_markers = (
        "prime",
        "nav-sprite",
        "amazon-logo",
        "icon",
        "sprite",
    )
    return any(marker in lowered for marker in bad_markers)


def sanitize_old_price(current_price: Optional[str], old_price: Optional[str]) -> Optional[str]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is None or old is None:
        return None

    if old <= current:
        return None

    return format_eur(old)


def is_placeholder_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized in {"", "offerta amazon", "galleria prodotti"}


def is_bad_image_url(image_url: Optional[str]) -> bool:
    if not image_url:
        return True

    lowered = image_url.lower()
    # Evita immagini placeholder generiche (es. logo Prime) nei post.
    bad_markers = (
        "prime",
        "nav-sprite",
        "amazon-logo",
        "icon",
        "sprite",
    )
    return any(marker in lowered for marker in bad_markers)


def normalize_image_url(image_url: Optional[str]) -> Optional[str]:
    if not image_url:
        return None

    cleaned = image_url.replace("&amp;", "&")
    # Rimuove i modificatori di resize aggressivi Amazon per evitare miniature/sfocatura.
    cleaned = re.sub(r"\._[^.]+_\.", ".", cleaned)
    cleaned = re.sub(r"\._[A-Z0-9,]+_", "", cleaned)
    return cleaned


def asin_in_url(url: Optional[str], asin: str) -> bool:
    if not url:
        return False
    return asin.lower() in url.lower()


def sanitize_old_price(current_price: Optional[str], old_price: Optional[str]) -> Optional[str]:
    current = parse_price_to_float(current_price)
    old = parse_price_to_float(old_price)

    if current is None or old is None:
        return None

    if old <= current:
        return None

    return format_eur(old)


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

    canonical_href = None
    canonical_tag = soup.select_one("link[rel='canonical']")
    if canonical_tag:
        canonical_href = canonical_tag.get("href")

    og_url = None
    og_url_tag = soup.select_one("meta[property='og:url']")
    if og_url_tag:
        og_url = og_url_tag.get("content")

    if is_placeholder_title(title):
        og_title = soup.select_one("meta[property='og:title']")
        if og_title and og_title.get("content"):
            title = og_title.get("content").strip()

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

    image_url = normalize_image_url(image_url)

    # Se Amazon reindirizza/mostra un ASIN differente, evita titolo/immagine incoerenti.
    if not asin_in_url(canonical_href, asin) and not asin_in_url(og_url, asin):
        logger.warning("Pagina Amazon non coerente per %s (canonical/og:url mismatch)", asin)
        title = "Offerta Amazon"
        image_url = None

    return title, price, old_price, image_url


def download_image_bytes(session: requests.Session, url: str) -> Optional[BytesIO]:
    try:
        normalized = normalize_image_url(url)
        response = session.get(normalized or url, timeout=(8, 20))
        response.raise_for_status()
        if len(response.content) < 25_000:
            logger.warning("Immagine troppo piccola, possibile miniatura (%s bytes)", len(response.content))
            return None
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
            if value <= 0:
                return None
            return float(value)
    return None


def extract_price_from_deal_item(deal_item: Dict[str, Any]) -> Optional[str]:
    for key in ("current", "currentPrice", "price"):
        value = deal_item.get(key)

        if isinstance(value, (list, tuple)) and value:
            value = value[0]

        text = keepa_cents_to_eur_text(value)
        if text:
            return text

    return None


def enrich_from_keepa_product(product: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    title = product.get("title") if isinstance(product, dict) else None

    image_url = None
    images_csv = product.get("imagesCSV") if isinstance(product, dict) else None
    if isinstance(images_csv, str) and images_csv.strip():
        first_image = images_csv.split(",")[0].strip()
        if first_image:
            image_url = normalize_image_url(
                "https://images-na.ssl-images-amazon.com/images/I/{}".format(first_image)
            )

    return title, image_url


def build_caption(
    title: str,
    price: str,
    old_price: Optional[str],
    link: str,
    discount_percent: Optional[float],
) -> str:
    badge = discount_badge(discount_percent)
    angle_line = build_marketing_angle(price, discount_percent)
    urgency_line = build_urgency_line(discount_percent)

    if old_price:
        price_line = "🔴 Ora {} invece di {}".format(price, old_price)
    else:
        price_line = "🔴 Prezzo lampo: {}".format(price)

    if discount_percent is not None:
        discount_line = "\n📉 Sconto: -{}%".format(str(discount_percent).replace(".", ","))

    return "{}\n{}\n{}\n{}{}\n⏳ {}\n{}".format(
        badge,
        title,
        angle_line,
        price_line,
        discount_line,
        urgency_line,
        link,
    )


def build_marketing_angle(price: str, discount_percent: Optional[float]) -> str:
    if discount_percent is None:
        templates = [
            "💥 In offerta a {}",
            "👉 Prezzo interessante: {}",
            "💸 Sotto i radar a {}",
        ]
    elif discount_percent >= 35:
        templates = [
            "💥 Ora a {}",
            "🔥 Scende a {}",
            "📉 Calato a {}",
        ]
    elif discount_percent >= 20:
        templates = [
            "💥 In offerta a {}",
            "💸 Trovato a {}",
            "👉 Si prende a {}",
        ]
    else:
        templates = [
            "🔥 Buon prezzo: {}",
            "👉 Prezzo interessante: {}",
            "💸 Sotto i radar a {}",
        ]

    return random.choice(templates).format(price)


def build_urgency_line(discount_percent: Optional[float]) -> str:
    high_urgency = [
        "Sta andando via veloce",
        "Finché dura",
        "Difficile rivederlo a questo prezzo",
    ]
    medium_urgency = [
        "Di solito costa di più",
        "Finché dura",
        "Sta andando via veloce",
    ]

    if discount_percent is not None and discount_percent >= 25:
        return random.choice(high_urgency)
    return random.choice(medium_urgency)

def pick_best_asin(deals: Dict[str, Any], sent_asins: Set[str]) -> Optional[str]:
    items = deals.get("dr") or []
    for item in items:
        asin = item.get("asin")
        if asin and asin not in sent_asins:
            return asin

    if items and items[0].get("asin"):
        return items[0]["asin"]
    return None



def product_looks_eligible(product_payload: Dict[str, Any], price: Optional[str]) -> bool:
    if not has_valid_current_price(price):
        return False

    root_category = product_payload.get("rootCategory")
    if isinstance(root_category, int) and root_category <= 0:
        return False

    return True


def discount_is_eligible(discount_percent: Optional[float], min_discount_percent: float) -> bool:
    if discount_percent is None:
        return False
    return discount_percent >= min_discount_percent


def trim_sent_asins(sent_asins: Set[str], limit: int = 5000) -> Set[str]:
    if len(sent_asins) <= limit:
        return sent_asins

    return set(sorted(sent_asins)[-limit:])


def select_matching_keepa_product(products: List[Dict[str, Any]], asin: str) -> Dict[str, Any]:
    for product in products:
        if normalize_asin(product.get("asin")) == asin:
            return product
    return products[0] if products else {}


def fetch_keepa_deals(api: keepa.Keepa) -> Optional[Dict[str, Any]]:
    payload = {
        "domainId": 8,
        "priceTypes": [0],
        "deltaPercentRange": [15, 90],
    }

    try:
        return api.deals(payload)
    except Exception as exc:
        logger.warning("Keepa deals non disponibile (tentativo 1): %s", exc)

    # Fallback più permissivo quando Keepa risponde con errori temporanei/503.
    fallback_payload = {
        "domainId": 8,
        "priceTypes": [0],
        "deltaPercentRange": [10, 90],
    }

    try:
        return api.deals(fallback_payload)
    except Exception as exc:
        logger.warning("Keepa deals non disponibile (tentativo 2): %s", exc)
        return None

# ---------------- AUTO OFFERTE ----------------
async def auto_offers(context: ContextTypes.DEFAULT_TYPE) -> None:
    loop = asyncio.get_running_loop()
    cfg = context.application.bot_data["config"]
    api = context.application.bot_data["keepa_api"]
    session = context.application.bot_data["http_session"]

    try:
        deals = await loop.run_in_executor(None, lambda: fetch_keepa_deals(api))

        if not deals or "dr" not in deals:
            logger.info("Nessuna offerta trovata (Keepa assente o vuoto)")
            return

        sent_asins = load_sent_asins(cfg.state_file)
        deal_items = deals.get("dr") or []
        if not deal_items:
            logger.info("Keepa ha restituito lista offerte vuota")
            return

        scanned = 0
        keepa_queries_done = 0
        for deal_item in deal_items[: cfg.max_deals_per_cycle]:
            asin = normalize_asin(deal_item.get("asin"))
            if not asin:
                logger.info("Deal scartato: asin mancante")
                continue

            if asin in sent_asins:
                logger.debug("Deal scartato (%s): già inviato", asin)
                continue

            scanned += 1
            logger.info("Analizzo ASIN %s (%s/%s)", asin, scanned, cfg.max_deals_per_cycle)

            try:
                title, price, old_price, image_url = await loop.run_in_executor(
                    None, lambda asin=asin: get_amazon_data(session, asin)
                )
            except Exception as exc:
                logger.warning("Scraping Amazon fallito per %s: %s", asin, exc)
                title, price, old_price, image_url = "Offerta Amazon", None, None, None

            product_payload: Dict[str, Any] = {}
            keepa_title: Optional[str] = None
            keepa_image_url: Optional[str] = None

            need_keepa_query = is_placeholder_title(title) or (not price) or is_bad_image_url(image_url)
            if need_keepa_query and keepa_queries_done < cfg.max_keepa_queries_per_cycle:
                try:
                    keepa_products = await loop.run_in_executor(None, lambda asin=asin: api.query(asin))
                    keepa_queries_done += 1
                except Exception as exc:
                    logger.warning("Query Keepa fallita per %s: %s", asin, exc)
                    keepa_products = []

                product_payload = select_matching_keepa_product(keepa_products, asin)
                keepa_title, keepa_image_url = enrich_from_keepa_product(product_payload)
            elif need_keepa_query:
                logger.info("Budget query Keepa esaurito nel ciclo, skip enrich per %s", asin)

            if is_placeholder_title(title) and keepa_title:
                title = keepa_title

            if not price:
                price = extract_price_from_keepa_product(product_payload) or extract_price_from_deal_item(deal_item)
                if price:
                    logger.warning("Prezzo Amazon non trovato per %s, uso fallback Keepa: %s", asin, price)

            if not product_looks_eligible(product_payload, price):
                logger.info("Deal scartato (%s): prezzo/metadata non validi", asin)
                continue

            keepa_discount = extract_keepa_discount_percent(deal_item)
            old_price = sanitize_old_price(price, old_price)
            discount_percent = compute_discount_percent(price, old_price, keepa_discount)

            if not discount_is_eligible(discount_percent, cfg.min_discount_percent):
                logger.info(
                    "Deal scartato (%s): sconto insufficiente (%s%% < %s%%)",
                    asin,
                    "n/d" if discount_percent is None else str(discount_percent).replace(".", ","),
                    str(cfg.min_discount_percent).replace(".", ","),
                )
                continue

            link = "https://www.amazon.it/dp/{}?tag={}".format(asin, cfg.affiliate_tag)
            caption = build_caption(title, price or "Prezzo non disponibile", old_price, link, discount_percent)

            # Salva subito lo stato per evitare duplicati in caso di riavvio/crash
            # subito dopo l'invio Telegram.
            sent_asins.add(asin)
            save_sent_asins(cfg.state_file, trim_sent_asins(sent_asins))

            sent = False
            if is_bad_image_url(image_url) and keepa_image_url:
                image_url = keepa_image_url

            try:
                if image_url and not is_bad_image_url(image_url):
                    image_bytes = await loop.run_in_executor(
                        None, lambda image_url=image_url: download_image_bytes(session, image_url)
                    )
                    if image_bytes:
                        await context.bot.send_photo(chat_id=cfg.channel_id, photo=image_bytes, caption=caption)
                        sent = True

                if not sent:
                    await context.bot.send_message(chat_id=cfg.channel_id, text=caption)
            except Exception as exc:
                sent_asins.discard(asin)
                save_sent_asins(cfg.state_file, trim_sent_asins(sent_asins))
                logger.warning("Invio Telegram fallito per %s: %s", asin, exc)
                continue

            logger.info("Offerta inviata correttamente per %s", asin)
            return

        logger.info("Nessuna offerta idonea inviata in questo ciclo (analizzati: %s, query_keepa: %s)", scanned, keepa_queries_done)

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
