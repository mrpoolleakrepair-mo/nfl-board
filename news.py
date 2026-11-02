"""Check 9 (fade the media's favorite total): scan NFL.com (NFL Network) and ESPN coverage of each upcoming game.

Usage: python news.py <raw_dir> <data_dir>
For every game in the next 8 days it searches news for the matchup on nfl.com and espn.com (Bing News RSS),
also pulls ESPN's own game-page news (ESPN public API), reads each article, and tags over / under language
near the two team names. Results are appended to data/news_snapshots.csv so the call can be graded later.
"""
import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

SEASON = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
TEAMS = json.load(open(os.path.join(HERE, "data", "teams.json")))  # abbr -> {name: nickname}
UA = "Mozilla/5.0 (compatible; nfl-board/1.0; +https://github.com/mrpoolleakrepair-mo/nfl-board)"
SITES = {"nfl.com": "NFL.com / NFL Network", "espn.com": "ESPN"}



def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def bing_news(query):
    url = "https://www.bing.com/news/search?" + urllib.parse.urlencode({"q": query, "format": "rss", "count": 15})
    try:
        root = ET.fromstring(fetch(url))
    except Exception as e:
        print(f"  bing error '{query}': {e}")
        return []
    out = []
    for it in root.iter("item"):
        link = it.findtext("link") or ""
        m = re.search(r"[?&]url=([^&]+)", link)  # bing wraps links
        link = urllib.parse.unquote(m.group(1)) if m else link
        out.append(dict(title=html.unescape(it.findtext("title") or ""), url=link,
                        desc=html.unescape(re.sub("<[^>]+>", " ", it.findtext("description") or "")), published=it.findtext("pubDate") or ""))
    return out


def espn_game_news(away, home, gameday):
    """ESPN's scoreboard -> event -> summary 'news' and 'article' blocks."""
    out = []
    try:
        d = gameday.replace("-", "")
        sb = json.loads(fetch(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={d}"))
        for ev in sb.get("events", []):
            names = ev.get("shortName", "")  # e.g. "CHI @ SEA"
            if not (away in names and home in names) and not (TEAMS[away]["name"] in ev.get("name", "") and TEAMS[home]["name"] in ev.get("name", "")):
                continue
            sm = json.loads(fetch(f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event={ev['id']}"))
            arts = (sm.get("news") or {}).get("articles", []) + ([sm["article"]] if isinstance(sm.get("article"), dict) else [])
            for a in arts:
                link = ((a.get("links") or {}).get("web") or {}).get("href", "")
                out.append(dict(title=a.get("headline", ""), url=link, desc=a.get("description", "") + " " + re.sub("<[^>]+>", " ", a.get("story", "") or ""),
                                published=a.get("published", ""), body=re.sub("<[^>]+>", " ", a.get("story", "") or "")))
    except Exception as e:
        print(f"  espn api error {away}@{home}: {e}")
    return out


def article_text(url):
    try:
        h = fetch(url)
    except Exception:
        return ""
    h = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", h)
    paras = re.findall(r"(?is)<p[^>]*>(.*?)</p>", h)
    txt = " ".join(re.sub("<[^>]+>", " ", p) for p in paras) or re.sub("<[^>]+>", " ", h)
    return html.unescape(re.sub(r"\s+", " ", txt))


NEG = re.compile(r"\b(not|no|never|isn't|won't|doesn't|unlikely|avoid|last thing|rather than|instead of)\b", re.I)
PICK_OVER = [r"\btake the over\b", r"\b(?:hit|go|goes|going|sail|sails) (?:the )?over\b", r"\blean(?:ing)? (?:to(?:ward)? )?(?:the )?over\b"]
PICK_UNDER = [r"\btake the under\b", r"\b(?:stay|stays|go|goes|going) under\b", r"\blean(?:ing)? (?:to(?:ward)? )?(?:the )?under\b"]
VAGUE_OVER = [r"\bshoot-?out\b", r"\bhigh[- ]scoring\b", r"\bpoints? galore\b", r"\bfireworks\b", r"\bscore(?:s|fest)? in bunches\b", r"\bback-and-forth\b"]
VAGUE_UNDER = [r"\bdefensive (?:battle|struggle|slugfest|showdown|game)\b", r"\blow[- ]scoring\b", r"\bslugfest\b", r"\bgrind(?:-it-out)?\b", r"\bugly game\b"]


def lean_of(text, away, home, total=None):
    """Score over/under talk about this game. Explicit picks on this game's total count 3, other picks 2 (when the
    sentence or its neighbour names a team), vague words 1 (skipped when negated, e.g. 'the last thing they want is a shootout')."""
    names = [TEAMS[away]["name"], TEAMS[home]["name"]]
    sents = re.split(r"(?<=[.!?])\s+", text)
    about = [any(n.lower() in x.lower() for n in names) or bool(re.search(r"\b(this|the) (game|matchup|contest)\b", x, re.I)) for x in sents]
    o = u = 0
    hits = []
    nums = set()
    if total is not None and pd.notna(total):
        nums = {f"{float(total) + d:g}" for d in (-1, -0.5, 0, 0.5, 1)}
    for i, x in enumerate(sents):
        near = about[i] or (i > 0 and about[i - 1]) or (i + 1 < len(about) and about[i + 1])
        so = su = 0
        for m in re.finditer(r"\b(over|under)\s+(\d{2}(?:\.5)?)\b", x, re.I):
            v = float(m.group(2))
            ok = abs(v - float(total)) <= 4 if (total is not None and pd.notna(total)) else 35 <= v <= 60  # a plausible game total, not "over 30 players"
            w = 3 if m.group(2) in nums else (2 if (near and ok) else 0)
            if m.group(1).lower() == "over": so += w
            else: su += w
        if near:
            so += 2 * sum(len(re.findall(p, x, re.I)) for p in PICK_OVER)
            su += 2 * sum(len(re.findall(p, x, re.I)) for p in PICK_UNDER)
            if not NEG.search(x):
                so += sum(len(re.findall(p, x, re.I)) for p in VAGUE_OVER)
                su += sum(len(re.findall(p, x, re.I)) for p in VAGUE_UNDER)
        if so or su:
            hits.append(x.strip()[:220])
        o += so
        u += su
    lean = "Over" if o > u else "Under" if u > o else ("Mixed" if o else "")
    return lean, o, u, hits[:2]


def main(raw, out):
    g = pd.read_csv(f"{raw}/games.csv")
    g = g[(g.season == SEASON) & (g.game_type == "REG") & g.result.isna()]
    today = pd.Timestamp.now(tz="America/Chicago").date()
    g = g[g.gameday.apply(lambda d: 0 <= (pd.Timestamp(d).date() - today).days <= 8)]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, seen = [], set()
    for _, r in g.iterrows():
        a, h = r.away_team, r.home_team
        na, nh = TEAMS[a]["name"], TEAMS[h]["name"]
        items = []
        for site in SITES:
            for q in (f'{na} {nh} site:{site}', f'{na} {nh} over under site:{site}', f'{na} {nh} prediction picks site:{site}'):
                for it in bing_news(q):
                    if site in it["url"]:
                        items.append(dict(it, site=site))
                time.sleep(0.8)
        items += [dict(it, site="espn.com") for it in espn_game_news(a, h, r.gameday)]
        n_art = 0
        for it in items:
            key = (r.game_id, it["url"] or it["title"])
            if key in seen or not (na.lower() in (it["title"] + it["desc"]).lower() or nh.lower() in (it["title"] + it["desc"]).lower()):
                continue
            seen.add(key)
            body = it.get("body") or (article_text(it["url"]) if it["url"] else "")
            lean, o, u, hits = lean_of(" ".join([it["title"], it["desc"], body]), a, h, r.total_line)
            n_art += 1
            rows.append(dict(ts=ts, game_id=r.game_id, source=SITES[it["site"]], title=it["title"][:200], url=it["url"], published=it["published"],
                             lean=lean, over_hits=o, under_hits=u, quote=" … ".join(hits)))
        print(f"  {a}@{h}: {n_art} articles, " + ", ".join(f"{k} {v}" for k, v in pd.Series([x['lean'] for x in rows if x['game_id'] == r.game_id and x['lean']]).value_counts().items()))
    if not rows:
        print("  no articles found")
        return
    df = pd.DataFrame(rows)
    path = f"{out}/news_snapshots.csv"
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  wrote {len(rows)} article rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
