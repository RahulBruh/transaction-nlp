"""Merchant directory and a seeded generator of messy card-statement descriptors.

All data is synthetic: real brand names are used only so descriptors look like
real statements. No real cardholder or transaction data is involved.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass

CATEGORIES = [
    "Dining",
    "Groceries",
    "Travel",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Utilities",
    "Health",
    "Gas",
    "Subscriptions",
]


@dataclass(frozen=True)
class Merchant:
    name: str
    category: str
    aliases: tuple[str, ...]
    # Small merchants are often paid through a processor (Square, Toast, ...)
    small: bool = False
    online: bool = False


def _m(name, category, *aliases, small=False, online=False):
    return Merchant(name, category, tuple(aliases) or (name.upper(),), small, online)


# 8 merchants per category. The last two in each category are held out of
# classifier training to measure generalization to merchants it has never seen.
MERCHANTS: list[Merchant] = [
    # Dining
    _m("Starbucks", "Dining", "STARBUCKS", "STARBUCKS STORE"),
    _m("McDonald's", "Dining", "MCDONALD'S", "MCDONALDS"),
    _m("Chipotle", "Dining", "CHIPOTLE", "CHIPOTLE MEXICAN GRILL"),
    _m("Blue Bottle Coffee", "Dining", "BLUE BOTTLE COFFEE", small=True),
    _m("Joe's Pizza", "Dining", "JOES PIZZA", small=True),
    _m("Chick-fil-A", "Dining", "CHICK-FIL-A", "CHICK FIL A"),
    _m("Panera Bread", "Dining", "PANERA BREAD", "PANERA"),
    _m("Shake Shack", "Dining", "SHAKE SHACK"),
    # Groceries
    _m("Kroger", "Groceries", "KROGER", "KROGER FUEL"),
    _m("Trader Joe's", "Groceries", "TRADER JOE'S", "TRADER JOES"),
    _m("Safeway", "Groceries", "SAFEWAY", "SAFEWAY STORE"),
    _m("Publix", "Groceries", "PUBLIX SUPER MARKETS", "PUBLIX"),
    _m("Aldi", "Groceries", "ALDI"),
    _m("Sprouts Farmers Market", "Groceries", "SPROUTS FARMERS MAR", "SPROUTS FARMERS MARKET"),
    _m("Whole Foods Market", "Groceries", "WHOLE FOODS MARKET", "WHOLEFDS"),
    _m("H-E-B", "Groceries", "H-E-B", "HEB GROCERY"),
    # Travel
    _m("Delta Air Lines", "Travel", "DELTA AIR LINES", "DELTA AIR"),
    _m("Marriott", "Travel", "MARRIOTT", "MARRIOTT HOTELS"),
    _m("Hilton", "Travel", "HILTON HOTELS", "HILTON"),
    _m("Expedia", "Travel", "EXPEDIA", online=True),
    _m("Airbnb", "Travel", "AIRBNB", online=True),
    _m("United Airlines", "Travel", "UNITED AIRLINES", "UNITED AIR"),
    _m("Southwest Airlines", "Travel", "SOUTHWEST AIRLINES", "SOUTHWES"),
    _m("Hyatt Hotels", "Travel", "HYATT HOTELS", "HYATT"),
    # Transportation
    _m("Uber", "Transportation", "UBER TRIP", "UBER", online=True),
    _m("Lyft", "Transportation", "LYFT RIDE", "LYFT", online=True),
    _m("Amtrak", "Transportation", "AMTRAK"),
    _m("Valley Metro", "Transportation", "VALLEY METRO TRANSIT", "VALLEY METRO"),
    _m("ParkMobile", "Transportation", "PARKMOBILE", online=True),
    _m("Greyhound", "Transportation", "GREYHOUND LINES", "GREYHOUND"),
    _m("Hertz Car Rental", "Transportation", "HERTZ CAR RENTAL", "HERTZ"),
    _m("Enterprise Rent-A-Car", "Transportation", "ENTERPRISE RENT-A-CAR", "ENTERPRISE RENT"),
    # Shopping
    _m("Amazon", "Shopping", "AMZN MKTP US", "AMAZON.COM", online=True),
    _m("Target", "Shopping", "TARGET", "TARGET.COM"),
    _m("Walmart", "Shopping", "WALMART", "WAL-MART"),
    _m("Best Buy", "Shopping", "BEST BUY", "BESTBUY.COM"),
    _m("Etsy", "Shopping", "ETSY.COM", "ETSY", online=True),
    _m("The Home Depot", "Shopping", "THE HOME DEPOT", "HOME DEPOT"),
    _m("Nordstrom", "Shopping", "NORDSTROM"),
    _m("IKEA", "Shopping", "IKEA"),
    # Entertainment
    _m("AMC Theatres", "Entertainment", "AMC THEATRES", "AMC"),
    _m("Ticketmaster", "Entertainment", "TICKETMASTER", online=True),
    _m("Steam", "Entertainment", "STEAMGAMES.COM", "STEAM PURCHASE", online=True),
    _m("Dave & Buster's", "Entertainment", "DAVE & BUSTERS", "DAVE&BUSTERS"),
    _m("Topgolf", "Entertainment", "TOPGOLF"),
    _m("Bowlero", "Entertainment", "BOWLERO"),
    _m("Regal Cinemas", "Entertainment", "REGAL CINEMAS", "REGAL"),
    _m("StubHub", "Entertainment", "STUBHUB", online=True),
    # Utilities
    _m("APS Electric", "Utilities", "APS ELECTRIC", "ARIZONA PUBLIC SERVICE"),
    _m("Comcast Xfinity", "Utilities", "COMCAST XFINITY", "XFINITY"),
    _m("AT&T", "Utilities", "AT&T BILL PAYMENT", "ATT"),
    _m("Verizon Wireless", "Utilities", "VERIZON WIRELESS", "VZWRLSS"),
    _m("City of Phoenix Water", "Utilities", "CITY OF PHOENIX WATER", "CITY OF PHX WATER"),
    _m("Southwest Gas", "Utilities", "SOUTHWEST GAS", "SW GAS"),
    _m("T-Mobile", "Utilities", "T-MOBILE", "TMOBILE"),
    _m("Cox Communications", "Utilities", "COX COMMUNICATIONS", "COX COMM"),
    # Health
    _m("CVS Pharmacy", "Health", "CVS/PHARMACY", "CVS PHARMACY"),
    _m("Walgreens", "Health", "WALGREENS"),
    _m("Banner Health", "Health", "BANNER HEALTH"),
    _m("Planet Fitness", "Health", "PLANET FITNESS", "PLANET FIT"),
    _m("LabCorp", "Health", "LABCORP"),
    _m("Quest Diagnostics", "Health", "QUEST DIAGNOSTICS"),
    _m("Rite Aid Pharmacy", "Health", "RITE AID PHARMACY", "RITE AID"),
    _m("Dental Care Clinic", "Health", "DENTAL CARE CLINIC", small=True),
    # Gas
    _m("Shell", "Gas", "SHELL OIL", "SHELL SERVICE STATION"),
    _m("Chevron", "Gas", "CHEVRON"),
    _m("Exxon", "Gas", "EXXONMOBIL", "EXXON"),
    _m("Circle K", "Gas", "CIRCLE K"),
    _m("QuikTrip", "Gas", "QUIKTRIP", "QT"),
    _m("Arco", "Gas", "ARCO"),
    _m("BP Gas Station", "Gas", "BP GAS STATION", "BP"),
    _m("Valero", "Gas", "VALERO"),
    # Subscriptions
    _m("Netflix", "Subscriptions", "NETFLIX.COM", "NETFLIX", online=True),
    _m("Spotify", "Subscriptions", "SPOTIFY USA", "SPOTIFY", online=True),
    _m("Apple iCloud", "Subscriptions", "APPLE.COM/BILL", online=True),
    _m("Adobe", "Subscriptions", "ADOBE CREATIVE CLOUD", "ADOBE", online=True),
    _m("Hulu", "Subscriptions", "HULU", online=True),
    _m("Microsoft 365", "Subscriptions", "MICROSOFT 365", "MSFT", online=True),
    _m("Disney Plus", "Subscriptions", "DISNEY PLUS", online=True),
    _m("YouTube Premium", "Subscriptions", "YOUTUBE PREMIUM", "GOOGLE YOUTUBE", online=True),
]

HELD_OUT_PER_CATEGORY = 2

CITIES = [
    ("PHOENIX", "AZ"), ("TEMPE", "AZ"), ("SCOTTSDALE", "AZ"), ("MEMPHIS", "TN"),
    ("NASHVILLE", "TN"), ("NEW YORK", "NY"), ("SAN FRANCISCO", "CA"),
    ("AUSTIN", "TX"), ("CHICAGO", "IL"), ("SEATTLE", "WA"), ("DENVER", "CO"),
]

PROCESSORS = ["SQ *", "TST* ", "SP * ", "PY *", "PAYPAL *"]


def split_merchants() -> tuple[list[Merchant], list[Merchant]]:
    """Return (seen, held_out) merchants for classifier training and testing."""
    seen, held = [], []
    for cat in CATEGORIES:
        group = [m for m in MERCHANTS if m.category == cat]
        seen += group[:-HELD_OUT_PER_CATEGORY]
        held += group[-HELD_OUT_PER_CATEGORY:]
    return seen, held


def _typo(s: str, rng: random.Random) -> str:
    letters = [i for i, c in enumerate(s) if c.isalpha()]
    if len(letters) < 5:
        return s
    i = rng.choice(letters[1:-1])
    op = rng.random()
    if op < 0.4:  # drop a character
        return s[:i] + s[i + 1:]
    if op < 0.8 and i + 1 < len(s):  # swap adjacent characters
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    return s[:i] + s[i] + s[i:]  # double a character


def make_descriptor(m: Merchant, rng: random.Random, hard: bool = False) -> str:
    """Render one noisy statement line for merchant m.

    hard=True adds a second typo, drops a word and truncates aggressively,
    which is closer to the worst lines seen on real statements.
    """
    s = rng.choice(m.aliases)
    if hard:
        words = s.split()
        if len(words) > 1 and rng.random() < 0.5:
            words.pop(rng.randrange(1, len(words)))
            s = " ".join(words)
        s = _typo(_typo(s, rng), rng)
        if rng.random() < 0.5:
            s = s[: rng.randint(8, 14)].rstrip()
    elif rng.random() < 0.2:
        s = _typo(s, rng)
    if m.small and rng.random() < 0.7:
        s = rng.choice(PROCESSORS) + s
    elif m.online and rng.random() < 0.5:
        s += "*" + "".join(rng.choices(string.ascii_uppercase + string.digits, k=rng.randint(6, 9)))
    if not m.online and rng.random() < 0.6:
        s += rng.choice([" #", " ", " 0"]) + str(rng.randint(100, 99999))
    # Card networks truncate the merchant name field to roughly 22-25 chars
    if rng.random() < 0.3:
        s = s[: rng.randint(18, 25)].rstrip()
    if rng.random() < 0.6:
        city, state = rng.choice(CITIES)
        s += " " + (city if rng.random() < 0.7 else city[:8]) + " " + state
    elif m.online and rng.random() < 0.5:
        s += rng.choice([" 866-216-1072 WA", " WWW.", " ONLINE"])
    if rng.random() < 0.1:
        s = s.title()
    return " ".join(s.split())


def generate(merchants: list[Merchant], per_merchant: int, seed: int, hard: bool = False) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for m in merchants:
        for _ in range(per_merchant):
            rows.append({"descriptor": make_descriptor(m, rng, hard), "merchant": m.name, "category": m.category})
    rng.shuffle(rows)
    return rows
