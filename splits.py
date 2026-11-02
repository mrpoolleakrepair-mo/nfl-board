"""Pull DraftKings and Circa betting splits (bet % and handle %) from SharpAPI and keep a running log.

Usage: python splits.py <raw_dir> <data_dir>
Needs SHARPAPI_KEY (Pro plan or higher). Without it this script does nothing, so the rest of the refresh still runs.
Docs: https://docs.sharpapi.io/en/api-reference/splits/
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

KEY = os.environ.get("SHARPAPI_KEY", "").strip()
BASE = "https://api.sharpapi.io/api/v1"
SEASON = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
TEAMS = json.load(open(os.path.join(HERE, "data", "teams.json")))       # abbr -> {name: nickname}
FULL = json.load(open(os.path.join(HERE, "team_names.json")))           # "Chicago Bears" -> CHI
NICK = {v["name"].lower(): k for k, v in TEAMS.items()}                 # "bears" -> CHI
COLS = ["ts", "game_id", "book", "market", "side", "bet_pct", "handle_pct"]


def get(path, **params):
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-API-Key": KEY, "Accept": "application/json", "User-Agent": "nfl-board"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} on {path}: {e.read().decode()[:300]}")
            if e.code in (401, 403, 404, 422):
                return None
        except Exception as e:
            print(f"  error on {path}: {e}")
        time.sleep(3 * (attempt + 1))
    return None


def team_of(text):
    """Find an NFL team in a name or id string: full name, nickname or abbreviation."""
    if not text:
        return None
    t = str(text)
    if t in FULL:
        return FULL[t]
    low = t.lower()
    for nick, ab in NICK.items():
        if re.search(rf"(^|[^a-z0-9]){re.escape(nick)}([^a-z0-9]|$)", low):
            return ab
    if t.upper() in TEAMS:
        return t.upper()
    return None


def first(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return None


def pct(v):
    if v is None or v == "":
        return None
    v = float(v)
    return round(v * 100 if v <= 1.0 else v, 1)


def match_game(item, games):
    eid = str(first(item, "event_id", "eventId", "id") or "")
    away = team_of(first(item, "away_team", "awayTeam", "away"))
    home = team_of(first(item, "home_team", "homeTeam", "home"))
    if not (away and home):  # fall back to the canonical id, e.g. nfl_bears_seahawks_2026-11-02
        found = [ab for nick, ab in NICK.items() if re.search(rf"_{re.escape(nick)}(_|$)", eid.lower())]
        if len(set(found)) == 2:
            a, b = list(dict.fromkeys(found))
            x = games[((games.away_team == a) & (games.home_team == b)) | ((games.away_team == b) & (games.home_team == a))]
            if len(x):
                away, home = x.away_team.iloc[0], x.home_team.iloc[0]
    if not (away and home):
        return None, eid
    m = re.search(r"(\d{4}-\d{2}-\d{2})", eid + " " + str(first(item, "start_time", "commence_time", "event_start") or ""))
    x = games[(games.away_team == away) & (games.home_team == home)]
    if m and len(x) > 1:
        d = pd.Timestamp(m.group(1)).date()
        x = x[x.gameday.apply(lambda g: abs((pd.Timestamp(g).date() - d).days) <= 1)]
    x = x[x.result.isna()] if len(x) > 1 else x
    return (x.game_id.iloc[0] if len(x) else None), eid


def norm_market(m):
    m = str(m or "").lower()
    if "half" in m or "quarter" in m or m.startswith(("1st", "2nd", "q1", "h1")):
        return None
    if "money" in m or m in ("ml", "h2h"):
        return "moneyline"
    if "spread" in m or "handicap" in m:
        return "spread"
    if "total" in m or "over" in m:
        return "total"
    return None


def rows_from(item, gid, ts, default_book=None):
    """Handles both shapes: one item per market, or one item per event with markets nested inside."""
    out = []
    book = first(item, "sportsbook", "book", "source") or default_book
    nested = first(item, "markets", "splits")
    if isinstance(nested, dict):
        nested = [dict(v, market=k) if isinstance(v, dict) else v for k, v in nested.items()]
    if isinstance(nested, list):
        for sub in nested:
            out += rows_from(dict(sub, event_id=item.get("event_id")), gid, ts, book)
        return out
    mk = norm_market(first(item, "market", "market_type", "type", "bet_type"))
    bets = first(item, "bet_pct", "bets_pct", "ticket_pct", "tickets_pct", "bet_percent", "bets")
    hand = first(item, "handle_pct", "money_pct", "handle_percent", "money", "handle")
    if not (mk and isinstance(bets, dict) and isinstance(hand, dict)):
        return out
    for side in ("away", "home", "over", "under"):
        if side in bets or side in hand:
            out.append(dict(ts=ts, game_id=gid, book=book, market=mk, side=side, bet_pct=pct(bets.get(side)), handle_pct=pct(hand.get(side))))
    return out


def main(raw, out):
    g = pd.read_csv(f"{raw}/games.csv")
    games = g[(g.season == SEASON) & (g.game_type == "REG")]
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items, offset = [], 0
    for _ in range(10):
        res = get("/splits", league="nfl", sportsbook="draftkings,circa", limit=200, offset=offset)
        if res is None:
            break
        data = res.get("data", res) if isinstance(res, dict) else res
        items += data if isinstance(data, list) else []
        pg = (res.get("pagination") or res.get("meta", {}).get("pagination") or {}) if isinstance(res, dict) else {}
        if not pg.get("has_more"):
            break
        offset = pg.get("next_offset", offset + 200)
    print(f"  splits: {len(items)} items")
    json.dump(items[:4], open(f"{out}/splits_raw_sample.json", "w"), indent=1, default=str)  # format check
    rows, miss = [], set()
    for it in items:
        gid, eid = match_game(it, games)
        if not gid:
            miss.add(eid); continue
        rows += rows_from(it, gid, ts)
    if miss:
        print(f"  unmatched events: {sorted(miss)[:6]}")
    if not rows:
        print("  no split rows parsed; see splits_raw_sample.json for the response format")
        return
    df = pd.DataFrame(rows, columns=COLS)
    path = f"{out}/splits_snapshots.csv"
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True).drop_duplicates()
    df.to_csv(path, index=False)
    print(f"  wrote {len(rows)} split rows for {len(set(r['game_id'] for r in rows))} games")


if __name__ == "__main__":
    if not KEY:
        print("SHARPAPI_KEY not set: skipping splits pull.")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2])
