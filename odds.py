"""Pull NFL lines from The Odds API (https://the-odds-api.com) and keep a running history.

Usage:
  python odds.py daily    <raw_dir> <data_dir>   # current lines + Caesars 1Q/1H for games in the next 30h
  python odds.py backfill <raw_dir> <data_dir>   # paid plan: opening/closing lines + Caesars 1Q/1H for played games

Needs ODDS_API_KEY in the environment; without it this script does nothing, so the rest of the refresh still runs.
Caesars is the bookmaker key williamhill_us. Featured markets: h2h, spreads, totals.
"""
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

KEY = os.environ.get("ODDS_API_KEY", "").strip()
BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
BOOKS = "williamhill_us,draftkings,fanduel,betmgm"
FEATURED = "h2h,spreads,totals"
PERIODS = "h2h_q1,spreads_q1,totals_q1,h2h_h1,spreads_h1,totals_h1"
SEASON = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = json.load(open(os.path.join(HERE, "team_names.json")))
COLS = ["ts", "game_id", "book", "market", "outcome", "point", "price"]


def get(path, **params):
    params["apiKey"] = KEY
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                left = r.headers.get("x-requests-remaining")
                used = r.headers.get("x-requests-last")
                print(f"  {path.split('/')[-1]} ok · cost {used} · {left} credits left")
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:300]
            print(f"  HTTP {e.code} on {path}: {body}")
            if e.code in (401, 403, 422):  # bad key, plan doesn't allow it, bad params: don't retry
                return None
        except Exception as e:  # network hiccup
            print(f"  error on {path}: {e}")
        time.sleep(3 * (attempt + 1))
    return None


def load_games(raw):
    g = pd.read_csv(f"{raw}/games.csv")
    return g[(g.season == SEASON) & (g.game_type == "REG")].copy()


def match(ev, games):
    """Map an API event to an nflverse game_id by teams and kickoff date (ET)."""
    a, h = NAMES.get(ev.get("away_team")), NAMES.get(ev.get("home_team"))
    if not a or not h:
        return None
    ko = pd.Timestamp(ev["commence_time"]).tz_convert("America/New_York").date()
    x = games[(games.away_team == a) & (games.home_team == h)]
    x = x[x.gameday.apply(lambda d: abs((pd.Timestamp(d).date() - ko).days) <= 1)]
    return x.game_id.iloc[0] if len(x) else None


def flatten(ev, gid, ts, games):
    a = games.loc[games.game_id == gid, "away_team"].iloc[0]
    rows = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                nm = o.get("name", "")
                side = "over" if nm == "Over" else "under" if nm == "Under" else ("away" if NAMES.get(nm) == a else "home")
                rows.append(dict(ts=ts, game_id=gid, book=bk["key"], market=mk["key"], outcome=side,
                                 point=o.get("point"), price=o.get("price")))
    return rows


def append(path, rows):
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COLS)
    if os.path.exists(path):
        df = pd.concat([pd.read_csv(path), df], ignore_index=True).drop_duplicates()
    df.to_csv(path, index=False)
    print(f"  wrote {len(rows)} rows -> {os.path.basename(path)}")


def daily(raw, out):
    games = load_games(raw)
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    evs = get(f"/sports/{SPORT}/odds", bookmakers=BOOKS, markets=FEATURED, oddsFormat="american") or []
    rows, soon = [], []
    now = pd.Timestamp.now(tz="UTC")
    for ev in evs:
        gid = match(ev, games)
        if not gid:
            continue
        rows += flatten(ev, gid, ts, games)
        ko = pd.Timestamp(ev["commence_time"])
        if now < ko <= now + pd.Timedelta(hours=30):
            soon.append((ev["id"], gid))
    append(f"{out}/odds_snapshots.csv", rows)
    if os.environ.get("ODDS_PERIOD_LINES", "1") == "1":
        prow = []
        for eid, gid in soon:
            ev = get(f"/sports/{SPORT}/events/{eid}/odds", bookmakers=BOOKS, markets=PERIODS, oddsFormat="american")  # same credit cost for up to 10 books
            if ev:
                prow += flatten(ev, gid, ts, games)
        append(f"{out}/caesars_periods.csv", prow)


def backfill(raw, out):
    """Paid plans only. Opening (6 days out) and closing (10 min before kickoff) lines for every played game."""
    games = load_games(raw)
    done = games[games.result.notna()]
    have = set(pd.read_csv(f"{out}/closing_lines.csv").game_id) if os.path.exists(f"{out}/closing_lines.csv") else set()
    todo = done[~done.game_id.isin(have)]
    print(f"backfill: {len(todo)} played games without stored lines")
    rows = []
    # nflverse gametime is ET; group by kickoff so one historical call covers every game at that time
    todo = todo.assign(ko=pd.to_datetime(todo.gameday + " " + todo.gametime).dt.tz_localize("America/New_York").dt.tz_convert("UTC"))
    for ko, grp in todo.groupby("ko"):
        for kind, when in (("open", ko - pd.Timedelta(days=6)), ("close", ko - pd.Timedelta(minutes=10))):
            snap = get(f"/historical/sports/{SPORT}/odds", bookmakers=BOOKS, markets=FEATURED, oddsFormat="american",
                       date=when.strftime("%Y-%m-%dT%H:%M:%SZ"))
            for ev in (snap or {}).get("data", []):
                gid = match(ev, games)
                if gid in set(grp.game_id):
                    rows += [dict(r, ts=kind) for r in flatten(ev, gid, kind, games)]
                    if kind == "close":
                        pe = get(f"/historical/sports/{SPORT}/events/{ev['id']}/odds", bookmakers=BOOKS,
                                 markets=PERIODS, oddsFormat="american", date=when.strftime("%Y-%m-%dT%H:%M:%SZ"))
                        if pe and pe.get("data"):
                            rows += [dict(r, ts="close") for r in flatten(pe["data"], gid, "close", games)]
    append(f"{out}/closing_lines.csv", rows)


if __name__ == "__main__":
    if not KEY:
        print("ODDS_API_KEY not set: skipping odds pull.")
        sys.exit(0)
    mode, raw, out = sys.argv[1], sys.argv[2], sys.argv[3]
    (daily if mode == "daily" else backfill)(raw, out)
