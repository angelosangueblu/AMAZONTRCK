from telegram.ext import ApplicationBuilder, ContextTypes
import keepa
import asyncio
import requests
from bs4 import BeautifulSoup

# ---------------- CONFIG ----------------
BOT_TOKEN = "8509703982:AAEor0InXeHPbmQ0IYQSI1A4RKLe6tdSYVs"
KEEPA_KEY = "avc79soldui39hrj39uj3q7vj1op76dagis5mq9ldj9u3364oh0eclu5nir2670q"
CHANNEL_ID = -1003574273138
AFFILIATE_TAG = "dragonofferte-21"

api = keepa.Keepa(KEEPA_KEY)


# -------- PRENDI DATI AMAZON --------
def get_amazon_data(asin):

    session = requests.Session()

    headers = {
        "User-Agent": "Mozilla/5.0 (Linux; Android 10; SM-G973F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Mobile Safari/537.36",
        "Accept-Language": "it-IT,it;q=0.9"
    }

    url = f"https://www.amazon.it/gp/aw/d/{asin}"

    r = session.get(url, headers=headers, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")

    # titolo
    title_tag = soup.select_one("#productTitle")
    title = title_tag.get_text(strip=True) if title_tag else "Offerta Amazon"

    # prezzo
    price = None
    selectors = [
        "span.a-price-whole",
        "span.a-offscreen",
        "#priceblock_ourprice",
        "#priceblock_dealprice"
    ]

    for s in selectors:
        tag = soup.select_one(s)
        if tag:
            price = tag.get_text(strip=True)
            break

    # immagine corretta
    img = None
    img_tag = soup.select_one("img#landingImage")

    if img_tag:
        img = img_tag.get("src")

        # FIX: url relativo amazon
        if img and img.startswith("/"):
            img = "https://m.media-amazon.com" + img

    return title, price, img


# ---------------- AUTO OFFERTE ----------------
async def auto_offers(context: ContextTypes.DEFAULT_TYPE):

    loop = asyncio.get_running_loop()

    try:
        deals = await loop.run_in_executor(None, lambda: api.deals({
            "domainId": 8,
            "priceTypes": [0],
            "deltaPercentRange": [40, 90]
        }))

        if not deals or "dr" not in deals:
            print("Nessuna offerta trovata")
            return

        asin = deals["dr"][0]["asin"]
        print("Trovato:", asin)

        title, price, img = await loop.run_in_executor(None, lambda: get_amazon_data(asin))

        if not price:
            print("Prezzo non trovato (Amazon blocco temporaneo)")
            return

        link = f"https://www.amazon.it/dp/{asin}?tag={AFFILIATE_TAG}"

        caption = f"""🔥 {title}

💰 Prezzo: {price}

👉 {link}"""

        # manda foto o testo
        if img:
            await context.bot.send_photo(chat_id=CHANNEL_ID, photo=img, caption=caption)
        else:
            await context.bot.send_message(chat_id=CHANNEL_ID, text=caption)

        print("POSTATO!")

    except Exception as e:
        print("Errore:", e)


# ---------------- AVVIO ----------------
app = ApplicationBuilder().token(BOT_TOKEN).build()

app.job_queue.run_repeating(auto_offers, interval=300, first=20)

print("BOT OFFERTE ATTIVO")
app.run_polling()






