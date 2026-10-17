"""LLM-commanded, model-driven lane-defence agent for GDG NSUT Autopilot.

Every call (opening, then every 3 s of game time):
  1. assess(): a per-row combat model from the live board. For each row: enemy HP, regen and
     the damage they put on our front; our DPS there (soldiers in the row + all archers, which
     sweep rows lowest-first); time to clear the row vs. how long the front and the whole line
     last. Status: breach / critical / weak / ok / open (no defence, no enemy).
  2. Fixes per row, by counter: Penguin vs stacks & Wizards (500 to every enemy in the row),
     Soldiers where a row needs DPS (40 dps per 200), Shields where it needs time (700 hp per
     150), a far Shield to keep every row obstructed (a Pirate crosses faster than one pause).
  3. Options: hold (critical fixes only) / needed (+ weak rows) / speed (+ archers from coins
     above the reserve). Coins below the reserve go only to breach/critical rows.
  4. The LLM gets the rules (system prompt), the assessment and options, can call tools
     (assess_board, row_detail, check_plan) and picks an option or its own cards. Its plan is
     held to the reserve, and critical fixes are added for rows it left untouched. Any LLM
     failure -> the rules' default option.

Layout: Soldier @1, Archers @0 (then @2) behind, line Shield @6. At x=6 a Clown stops ~3.3 tiles
ahead (x~9.3, still on the board and in our range); a Wizard stops at the spawn (x=10) and its
5.3-tile beam can't reach anything at x<=4; a Pirate blast at x=6 (2 u = 1.3 tiles) can't reach
x<=4. Soldiers and archers have unlimited range.

SDK facts: each call runs in a FRESH subprocess (no memory between calls) and a PUBLISH run may
open no other file (requests' trust_env would read ~/.netrc, so it is switched off).
"""
import json
import math
import sys
import time

try:
    from autopilot_sdk.llm import LLMClient
    from autopilot_sdk.tools import run_tool_loop, tool
except Exception:  # offline tests without the SDK
    LLMClient = run_tool_loop = None

    def tool(f):
        return f

# --- game facts (player guide) ----------------------------------------------
PENGUIN, SOLDIER, ARCHER, SHIELD = 0, 1, 2, 3
CLOWN, MONK, WIZARD, PIRATE = 0, 1, 2, 3
ENEMY_IDS = {"EnemClown": CLOWN, "EnemyMonk": MONK, "EnemyWizard": WIZARD, "EnemyPirate": PIRATE}
DEF_IDS = {"defencePenguin": PENGUIN, "defenceSolider": SOLDIER, "defenceArcher": ARCHER, "defenceShield": SHIELD}
DEF_NAME = {PENGUIN: "penguin", SOLDIER: "soldier", ARCHER: "archer", SHIELD: "shield"}
FOE_NAME = {CLOWN: "clown", MONK: "monk", WIZARD: "wizard", PIRATE: "pirate"}
DEFAULT_COSTS = {PENGUIN: 250, SOLDIER: 200, ARCHER: 300, SHIELD: 150}
FOE_HP = {CLOWN: 200, MONK: 280, WIZARD: 500, PIRATE: 30}
REGEN = {CLOWN: 15, WIZARD: 15}                    # hp/s while hurt
FOE_DPS = {CLOWN: 17, MONK: 36, WIZARD: 50, PIRATE: 0}  # on the first defence in their row
FOE_SPEED = {CLOWN: 2.0, MONK: 3.5, WIZARD: 3.0, PIRATE: 8.0}  # units/s; 1 tile = 1.5 units
PIRATE_BLAST = 200
DPS = {SOLDIER: 40, ARCHER: 26}
PENGUIN_HIT = 500
ROWS = range(6)
MAX_X = 10

# --- tuning -----------------------------------------------------------------
USE_LLM = True
T = {
    "soldier_x": 1,
    "shield_x": 6,            # furthest useful blocker (see docstring); 5, 4 if 6 is taken
    "penguin_x": 7,           # opening penguin in wizard rows: in front of the shield, beamed first
    "archer_cols": (0, 2),    # archers behind the line: x=0 first, then x=2 (out of beam/blast reach)
    "max_archers": 9,
    "reserve_frac": 0.25,     # coin reserve: 25% of the starting coins, clamped to [min, max]...
    "reserve_min": 600,
    "reserve_max": 1200,
    "small_start": 2000,      # ...or 20% of them when the level starts with less than this
    "small_frac": 0.2,
    "wave_hp_per_archer": 1300,  # opening: one archer per this much wave HP (2..6 of them)
    "weak_margin": 1.0,       # row is weak if time-to-clear > front time * this
    "late_frac": 0.15,        # <= this share of the wave still queued = end of wave: critical fixes only
    "min_archers_open": 3,    # always pre-place at least this many archers at the opening...
    "archer_floor_frac": 0.5, # ...even below the reserve, but never below this share of it
    "heavy_walkers": 3,       # a row this busy (or with heavy_hp enemy hp) gets a second shield layer
    "heavy_hp": 700,
    "top_row": 3,             # rows >= this are "top": soldiers there (archers reach them last), archers below
    "llm_deadline": 11.0,     # s, hard cap (guide: 15 s per call; keep ~3 s spare)
    "llm_rounds_start": 3,
    "llm_rounds_pause": 3,    # 60 calls/min per team; pauses come every ~4 s real time
    "llm_max_tokens": 500,
}
INF = float("inf")


def log(msg):
    try:
        print(f"[agent] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _xy(u):
    return int(round(u["position"]["x"])), int(round(u["position"]["y"]))


def card(kind, x, y):
    return {"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}}


def _kxy(c):
    p = c["defenceGridPosition"]
    return c["defenceId"], p["x"], p["y"]


def check(cards, taken, coins, costs):
    """Keep cards that are on the board, on a free tile, not duplicated and affordable in order."""
    out = []
    for c in cards:
        try:
            kind, x, y = (int(v) for v in _kxy(c))
        except (KeyError, TypeError, ValueError):
            continue
        if kind in costs and 0 <= x <= MAX_X and 0 <= y <= 5 and (x, y) not in taken and costs[kind] <= coins:
            taken.add((x, y))
            coins -= costs[kind]
            out.append(card(kind, x, y))
    return out


def cost_of(cards, costs):
    return sum(costs.get(c["defenceId"], 999) for c in cards)


def _r(v, n=1):
    return None if v == INF else round(v, n)


# --- board ------------------------------------------------------------------
def analyse(state):
    costs = {int(k): int(v) for k, v in (state.get("unitCosts") or {}).items()} or dict(DEFAULT_COSTS)
    defs, foes = [], []
    for d in state.get("defences") or []:
        if d.get("health", 0) > 0 and "position" in d:
            x, y = _xy(d)
            defs.append({"k": DEF_IDS.get(d.get("type"), SHIELD), "x": x, "y": y, "hp": float(d["health"]),
                         "max": float(d.get("maxHealth") or d["health"])})
    for e in state.get("enemies") or []:
        if not e.get("isDead") and e.get("health", 0) > 0 and "position" in e:
            x, y = _xy(e)
            foes.append({"k": ENEMY_IDS.get(e.get("type"), CLOWN), "x": x, "y": y, "hp": float(e["health"]),
                         "atk": bool(e.get("isAttacking"))})
    counts, budget = state.get("counts") or {}, state.get("budget") or {}
    gs = state.get("gameSettings") or {}
    queued = int(counts.get("enemiesQueued", 0) or 0)
    total = int(counts.get("totalEnemiesInWave", 0) or 0)
    kills = max(0, total - queued - int(counts.get("activeEnemyCount", 0) or 0))
    coins = float(budget.get("remaining", 0) or 0)
    start = float(budget.get("unlocked", coins) or coins) - 50 * kills
    B = {
        "phase": state.get("phase", "pause"), "costs": costs, "coins": coins, "start": start,
        "defs": defs, "foes": foes, "occ": {(d["x"], d["y"]) for d in defs},
        "queued": queued, "total": total, "reinf": gs.get("reinforcementsEnabled", True),
        "threats": row_threats(state) if state.get("phase") == "start" else {},
        "budget_s": float(state.get("timeBudgetSeconds") or 15), "t": state.get("elapsedSeconds"),
    }
    B["reserve"] = reserve(B)
    return B


def reserve(B):
    """Coins banked for last-moment fixes; only breach/critical rows may spend below it."""
    if B["phase"] == "start" and not B["reinf"]:
        return 0.0  # no later decisions to save for
    s = B["start"]
    if s < T["small_start"]:
        return T["small_frac"] * max(0.0, s)
    return min(T["reserve_max"], max(T["reserve_min"], T["reserve_frac"] * s))


def row_threats(state):
    """{row: {enemyId: expected count}}. Missing allowedRows means any row."""
    threats = {y: {} for y in ROWS}
    for g in state.get("upcomingEnemies") or []:
        try:
            eid, n = int(g.get("enemyId", 0)), int(g.get("count", 0))
        except (TypeError, ValueError):
            continue
        rows = [r for r in (g.get("allowedRows") or ROWS) if r in threats] or list(ROWS)
        for r in rows:
            threats[r][eid] = threats[r].get(eid, 0) + n / len(rows)
    for e in state.get("enemies") or []:
        eid = ENEMY_IDS.get(e.get("type"))
        if eid is not None and "position" in e:
            r = _xy(e)[1]
            if r in threats:
                threats[r][eid] = max(threats[r].get(eid, 0), 1)
    return {y: t for y, t in threats.items() if t}


# --- combat model -----------------------------------------------------------
def assess(B, extra=()):
    """Per-row model of the live board (plus hypothetical `extra` cards [(kind, x, y)])."""
    defs = B["defs"] + [{"k": k, "x": x, "y": y, "hp": 700.0 if k == SHIELD else 150.0, "max": 1.0}
                        for k, x, y in extra]
    archer_dps = DPS[ARCHER] * sum(1 for d in defs if d["k"] == ARCHER)
    rows, t_prev = {}, 0.0
    for y in ROWS:
        foes = sorted((f for f in B["foes"] if f["y"] == y), key=lambda f: f["x"])
        mine = sorted((d for d in defs if d["y"] == y), key=lambda d: d["x"])
        blockers = [d for d in mine if d["k"] != ARCHER]
        walkers = [f for f in foes if f["k"] != PIRATE]
        hp = sum(f["hp"] for f in foes)
        regen = sum(REGEN.get(f["k"], 0) for f in foes)
        sold = sum(DPS[SOLDIER] for d in mine if d["k"] == SOLDIER and any(f["x"] >= d["x"] for f in foes))
        edps = sum(FOE_DPS[f["k"]] for f in walkers)
        front = max(blockers, key=lambda d: d["x"]) if blockers else None
        r = {"row": y, "enemies": [[FOE_NAME[f["k"]], round(f["hp"]), f["x"]] for f in foes],
             "defences": [[DEF_NAME[d["k"]], round(d["hp"]), d["x"]] for d in mine],
             "enemy_hp": round(hp), "regen": regen, "soldier_dps": sold, "enemy_dps_on_front": edps,
             "front": [DEF_NAME[front["k"]], round(front["hp"]), front["x"]] if front else None}
        if foes:
            # Archers sweep rows lowest-first: this row has only its soldiers until lower rows clear.
            hp_left = max(0.0, hp - max(0.0, sold - regen) * t_prev)
            net = sold + archer_dps - regen
            t_clear = t_prev + hp_left / net if net > 0 else INF
            t_prev = t_clear if t_clear < INF else t_prev
            t_front = front["hp"] / edps if front and edps else INF
            t_line = sum(d["hp"] for d in blockers) / edps if edps else INF
            pir = [f for f in foes if f["k"] == PIRATE]
            pirate_hits_line = bool(pir) and front is not None and front["x"] <= T["soldier_x"] + 1
            # seconds until the first enemy walks off the board at x=0 if nothing stops it
            t_reach = min(f["x"] * 1.5 / FOE_SPEED[f["k"]] for f in foes)
            if not blockers and t_clear < 0.7 * t_reach:
                status = "ok"       # nothing in the way, but it dies long before it gets there
            elif not blockers:
                status = "BREACH"
            elif t_clear > t_line or pirate_hits_line:
                status = "critical"
            elif t_clear > t_front * T["weak_margin"]:
                status = "weak"
            else:
                status = "ok"
            first = y == min(f["y"] for f in B["foes"])  # archers are on this row right now
            # A penguin (250, single use) only for the rare SERIOUS case: 2+ wizards/clowns bunched within
            # 2 tiles in a row that is losing (breached/critical). One slide then kills them all.
            regen_x = sorted(f["x"] for f in walkers if f["k"] in (CLOWN, WIZARD))
            bunched = max((sum(1 for b in regen_x if 0 <= b - a <= 2) for a in regen_x), default=0)
            serious = status in ("BREACH", "critical") and bunched >= 2
            r.update(dps_here=sold + (archer_dps if first else 0), serious=serious,
                     t_clear=_r(t_clear), t_front=_r(t_front), t_line=_r(t_line), t_reach=_r(t_reach), status=status,
                     nearest_enemy_x=foes[0]["x"])
        else:
            r.update(status="open" if not mine else "quiet")  # an archer-only row is not empty
        rows[y] = r
    return {"rows": rows, "archer_dps_total": archer_dps,
            "max_t_clear": max([r.get("t_clear") or 0 for r in rows.values()] + [0])}


def _pick(xs, y, occ, before=MAX_X + 1):
    return next((x for x in xs if 0 <= x < before and x <= MAX_X and (x, y) not in occ), None)


def row_fixes(B, y, r, occ):
    """(critical, normal, why, archers_critical, archers_normal) for one row. Mutates occ.
    Strategy: dps comes from archers; a soldier only for a top row (archers reach it last) that is short
    on dps; shields only to tank (breach, dying front, pirate bait, busy row); penguins only for bunched
    wizards/clowns in a losing row."""
    crit, norm, why = [], [], []
    foes = [f for f in B["foes"] if f["y"] == y]
    nearest = min((f["x"] for f in foes), default=MAX_X + 1)
    walkers = [f for f in foes if f["k"] != PIRATE]
    types = {f["k"] for f in foes}
    mine = [d for d in B["defs"] if d["y"] == y]
    kinds = {d["k"] for d in mine}
    blockers = [d for d in mine if d["k"] != ARCHER]
    front_x = max((d["x"] for d in mine), default=-1)
    sx, shx = T["soldier_x"], T["shield_x"]
    arch = [0, 0]

    def add(lst, kind, x, note):
        if x is not None:
            lst.append(card(kind, x, y)); occ.add((x, y)); why.append(f"r{y} {note}")
            return True
        return False

    def dps_gap(lst, level, need):
        """Cover a dps shortfall: top row without a soldier -> one soldier first; the rest as archers."""
        if need > 0 and y >= T["top_row"] and SOLDIER not in kinds:
            if add(lst, SOLDIER, _pick((sx, 2, 0), y, occ, nearest), f"soldier (top row, need {need:.0f} dps)"):
                need -= DPS[SOLDIER]
        if need > 0:
            n = min(3, math.ceil(need / DPS[ARCHER]))
            arch[level] += n
            why.append(f"r{y} +{n} archer (need {need:.0f} dps)")

    status = r["status"]
    if status in ("open", "quiet", "ok"):
        return crit, norm, why, 0, 0
    t_line = r["t_line"] if r.get("t_line") is not None else INF
    need = (r["enemy_hp"] / max(min(t_line, r.get("t_reach") or INF), 1.0)) + r["regen"] - r.get("dps_here", 0)
    if status == "BREACH":
        if r.get("serious"):  # bunched wizards/clowns: one slide kills them all
            add(crit, PENGUIN, _pick((min(nearest - 1, shx), 3, 2, 1, 0), y, occ, nearest), "BREACH->penguin")
        else:  # tank: a shield stops them where archers can grind them down
            add(crit, SHIELD, _pick(range(min(nearest - 1, shx), -1, -1), y, occ, nearest), "BREACH->shield")
            dps_gap(crit, 0, need)
        return crit, norm, why, arch[0], arch[1]

    lst = crit if status == "critical" else norm
    if PIRATE in types and not any(d["x"] > sx + 1 for d in blockers):  # pirate would blow up on our units
        add(crit, SHIELD, _pick(range(min(shx, nearest - 1), sx + 1, -1), y, occ, nearest), "pirate->bait")
    penguin = False
    if PENGUIN not in kinds and r.get("serious"):
        # In front of the line (closer than the nearest enemy) so the first throw/beam hits it now.
        penguin = add(lst, PENGUIN, _pick([*range(front_x + 1, nearest), *range(front_x - 1, sx, -1)], y, occ, nearest),
                      f"penguin vs bunched {len(walkers)} ({status})")
    if not penguin:
        dps_gap(lst, 0 if status == "critical" else 1, need)
        if r.get("t_front") is not None and r["t_front"] < 6:  # tank: more wall, in front else behind
            add(lst, SHIELD, _pick([*range(min(shx, nearest - 1), front_x, -1), *range(front_x - 1, 2, -1)],
                                   y, occ, nearest), "shield (front dying)")
    walls = [d for d in blockers if d["x"] > sx]
    if (len(walkers) >= T["heavy_walkers"] or r["enemy_hp"] >= T["heavy_hp"]) and len(walls) <= 1 \
            and not any(c["defenceId"] in (SHIELD, PENGUIN) for c in crit + norm):  # busy row: second wall
        add(norm, SHIELD, _pick(range(max(front_x - 1, shx), sx + 1, -1), y, occ, nearest), f"2nd shield ({len(walkers)} foes)")
    return crit, norm, why, arch[0], arch[1]


def archer_slots(B, occ, threats=None):
    """Archer tiles: x=0 first, in rows no enemy can use (opening), then bottom rows up (archers fire at
    the lowest row first, so bottom rows defend themselves); then x=2 in rows walled at x>=3."""
    quiet = [y for y in ROWS if threats is not None and y not in threats]
    out = [(0, y) for y in quiet + [y for y in ROWS if y not in quiet] if (0, y) not in occ]
    out += [(2, y) for y in ROWS if y not in quiet and (2, y) not in occ
            and any((x, y) in occ for x in range(3, MAX_X + 1))]
    return out


def row_unit(y):
    """The unit that occupies an otherwise empty row: archers bottom, soldiers top."""
    return (SOLDIER, T["soldier_x"]) if y >= T["top_row"] else (ARCHER, 0)


def fill_empty(B, cards, rows, floor, units=True):
    """No row left empty (a pirate crosses faster than one decision): each row of `rows` with nothing in
    it gets its unit (archer bottom / soldier top) while coins stay above `floor`, else a 150 shield."""
    tiles, left, out = _tiles(B, cards), B["coins"] - cost_of(cards, B["costs"]), []
    empty = [y for y in rows if not any(t[1] == y for t in tiles)]
    shield_p = B["costs"].get(SHIELD, 150)
    for i, y in enumerate(empty):
        kind, x = row_unit(y)
        p = B["costs"].get(kind, 999)
        later = shield_p * (len(empty) - i - 1)  # covering every row beats a better unit in one row
        if units and left - p - later >= floor and (x, y) not in tiles:
            c = card(kind, x, y)
        else:
            sx = _pick((T["shield_x"], T["shield_x"] - 1, T["shield_x"] - 2), y, tiles)
            if sx is None:
                continue
            c, p = card(SHIELD, sx, y), B["costs"].get(SHIELD, 150)
        out.append(c); tiles.add(tuple(_kxy(c)[1:])); left -= p
    return out


def shields_for_empty(rows, occ_tiles):
    out = []
    for y in rows:
        if not any(t[1] == y for t in occ_tiles):
            x = _pick((T["shield_x"], T["shield_x"] - 1, T["shield_x"] - 2), y, occ_tiles)
            if x is not None:
                out.append(card(SHIELD, x, y))
    return out


def _tiles(B, cards):
    return set(B["occ"]) | {tuple(_kxy(c)[1:]) for c in cards}


def keep_reserve(cards, coins, floor, costs):
    out, left = [], coins
    for c in cards:
        p = costs.get(c["defenceId"], 999)
        if left - p >= floor:
            out.append(c); left -= p
    return out


# --- options: opening -------------------------------------------------------
def opening_options(B):
    """No shields. Each threatened row gets one unit: archer@0 in bottom rows (0-2), soldier@1 in top rows
    (3-5); at least 3 archers; spare coins above the reserve -> more archers (dps). A row the coins can't
    cover with its unit gets a 150 shield, so no row is ever open."""
    threats, costs, coins, res = B["threats"], B["costs"], B["coins"], B["reserve"]
    if not B["reinf"]:
        return opening_no_pauses(B)
    rows = sorted(threats)
    cover = fill_empty(B, [], rows, T["archer_floor_frac"] * res)
    occ = _tiles(B, cover)
    have = sum(1 for c in cover if c["defenceId"] == ARCHER)
    # minimum archers: behind the top-row soldiers (x=0), bottom rows first
    extra_min = [card(ARCHER, x, y) for x, y in archer_slots(B, occ, threats)]
    slots = [c for c in extra_min]
    wave_hp = sum(FOE_HP.get(k, 200) * n for t in threats.values() for k, n in t.items())
    n_base = max(T["min_archers_open"], min(6, round(wave_hp / T["wave_hp_per_archer"])))

    def build(n_total, floor):
        out, left = list(cover), coins - cost_of(cover, costs)
        n = have
        for c in slots:
            if n >= n_total:
                break
            lim = floor if n >= T["min_archers_open"] else T["archer_floor_frac"] * res
            if left - costs.get(ARCHER, 300) >= lim:
                out.append(c); left -= costs.get(ARCHER, 300); n += 1
        return out

    lean = build(T["min_archers_open"], res)
    opts = {
        "fortress": (build(T["max_archers"], res), f"one unit per threatened row (archers bottom, soldiers top), "
                     f"then every archer the coins allow above the {res:.0f} reserve; no shields"),
        "balanced": (build(n_base, res), f"one unit per row + {n_base} archers total, keeps the reserve"),
        "lean": (lean, f"one unit per row + the minimum {T['min_archers_open']} archers"),
    }
    return opts, "fortress", cover, {"wave_hp": wave_hp, "archers_suggested": n_base,
                                     "threats": {y: {FOE_NAME.get(k, k): round(v, 1) for k, v in t.items()}
                                                 for y, t in threats.items()}}


def opening_no_pauses(B):
    """Reinforcements off: no second decision, so build every threatened row fully now: soldier@1,
    shield@6, a penguin in wizard rows, then archers."""
    threats, costs, coins = B["threats"], B["costs"], B["coins"]
    sx, cards = T["soldier_x"], []
    for y in sorted(threats):
        t = threats[y]
        cards += [card(SOLDIER, sx, y), card(SHIELD, T["shield_x"], y)]
        if t.get(WIZARD, 0) >= 1:
            cards.append(card(PENGUIN, T["penguin_x"], y))
        if t.get(WIZARD, 0) >= 1.5:
            cards.append(card(PENGUIN, T["penguin_x"] + 1, y))
    arch = [card(ARCHER, x, y) for x, y in archer_slots(B, _tiles(B, cards), threats)][: T["max_archers"]]
    full = cards + keep_reserve(arch, coins - cost_of(cards, costs), 0, costs)
    opts = {"fortress": (full, "no pauses: full line in every threatened row + archers"),
            "lean": (cards, "no pauses: full line only")}
    return opts, "fortress", cards, {"threats": {y: {FOE_NAME.get(k, k): round(v, 1) for k, v in t.items()}
                                                 for y, t in threats.items()}}


# --- options: pauses --------------------------------------------------------
def fortify(B, planned, late):
    """Tank only where needed: a shield in front of archers/soldiers that have nothing ahead of them
    AND enemies coming in their row. Rows with archers first."""
    tiles = _tiles(B, planned)
    planned_walls = {(_kxy(c)[1], _kxy(c)[2]) for c in planned if c["defenceId"] in (SHIELD, PENGUIN)}
    rows = []
    for y in ROWS:
        foes = [f["x"] for f in B["foes"] if f["y"] == y]
        if not foes:
            continue
        units = [d["x"] for d in B["defs"] if d["y"] == y and d["k"] in (ARCHER, SOLDIER)] + \
            [_kxy(c)[1] for c in planned if _kxy(c)[2] == y and c["defenceId"] in (ARCHER, SOLDIER)]
        if not units:
            continue
        walls = [d["x"] for d in B["defs"] if d["y"] == y and d["k"] in (SHIELD, PENGUIN)] + \
            [x for x, yy in planned_walls if yy == y]
        if any(w > max(units) for w in walls):
            continue
        nearest = min(foes)
        x = _pick(range(min(T["shield_x"], nearest - 1), max(units) + 1, -1), y, tiles, nearest)
        if x is not None:
            has_archer = any(d["k"] == ARCHER for d in B["defs"] if d["y"] == y)
            rows.append(((not has_archer, y), card(SHIELD, x, y)))
    return [c for _, c in sorted(rows, key=lambda r: r[0])]


def affordable_fallback(B, A, cards, occ):
    """A breached/critical row whose fixes can't be paid for gets the cheapest card that still helps:
    a shield (150) in front of its line, else behind it. (TC02: 150 coins, a wizard row critical, the
    penguin+soldier plan unaffordable -> nothing was placed and the game was lost.)"""
    left, paid_rows = B["coins"], set()
    for c in cards:
        p = B["costs"].get(c["defenceId"], 999)
        if p <= left:
            left -= p; paid_rows.add(_kxy(c)[2])
    out = list(cards)
    for y, r in A["rows"].items():
        if r["status"] in ("BREACH", "critical") and y not in paid_rows and B["costs"].get(SHIELD, 150) <= left:
            nearest = r.get("nearest_enemy_x", MAX_X + 1)
            front = max((d["x"] for d in B["defs"] if d["y"] == y and d["k"] != ARCHER), default=-1)
            x = _pick([*range(min(T["shield_x"], nearest - 1), front, -1), *range(front - 1, -1, -1)], y, occ, nearest)
            if x is not None:
                out.insert(0, card(SHIELD, x, y)); occ.add((x, y)); left -= B["costs"].get(SHIELD, 150)
    return out


def pause_options(B):
    """Row fixes (dps via archers, soldiers only for short top rows, shields only to tank) -> archers ->
    fill rows left empty (archer bottom / soldier top above the reserve, else a shield)."""
    costs, res = B["costs"], B["reserve"]
    A = assess(B)
    occ = set(B["occ"])
    order = sorted(ROWS, key=lambda y: ["BREACH", "critical", "weak", "ok", "open", "quiet"].index(A["rows"][y]["status"]))
    critical, normal, why, ac, an = [], [], [], 0, 0
    for y in order:
        c, n, w, a1, a2 = row_fixes(B, y, A["rows"][y], occ)
        critical += c; normal += n; why += w; ac += a1; an += a2
    room = max(0, T["max_archers"] - sum(1 for d in B["defs"] if d["k"] == ARCHER))
    slots = [card(ARCHER, x, y) for x, y in archer_slots(B, occ)]
    crit_arch = slots[:min(ac, room)]
    norm_arch = slots[len(crit_arch): len(crit_arch) + min(an, room - len(crit_arch))]
    spare = slots[len(crit_arch) + len(norm_arch): room]
    critical += crit_arch
    critical = affordable_fallback(B, A, critical, occ)
    late = B["queued"] <= T["late_frac"] * B["total"]  # end of wave: critical fixes only
    base = critical + ([] if late else keep_reserve(normal + norm_arch, B["coins"] - cost_of(critical, costs), res, costs))
    base += keep_reserve(fortify(B, base, late), B["coins"] - cost_of(base, costs), res, costs)
    speed = base + ([] if late else keep_reserve(spare, B["coins"] - cost_of(base, costs), res, costs))

    def finish(cards, n_fixed, units=True):
        """Last: no row left empty. Optional cards (after the first n_fixed) give way, from the end,
        until the fillers are affordable."""
        cards = list(cards)
        while True:
            out = cards + fill_empty(B, cards, ROWS, res, units)
            if cost_of(out, costs) <= B["coins"] or len(cards) <= n_fixed:
                return out
            cards.pop()

    n_crit = len(critical)
    hold = finish(critical, n_crit, units=False)
    opts = {
        "speed": (finish(speed, n_crit), f"needed + archers with the coins above the {res:.0f} reserve (faster kills, all rows)"),
        "needed": (finish(base, n_crit), ("END OF WAVE: only rows that can't clear before the line falls. " if late else "")
                   + "fixes: " + (", ".join(why) or "nothing urgent")),
        "hold": (hold, "only critical fixes (breach / line about to fall) + a shield in any empty row, keep coins"),
    }
    default = "speed" if len(speed) > len(base) else "needed"
    return opts, default, hold, A


# --- LLM decision -----------------------------------------------------------
PROMPT = """You command a lane-defence game. Grid x 0-10 (base x=0; enemies enter at x=10 and walk left), rows y 0-5, 1 tile = 1.5 units. Enemies never change rows. ONE enemy crossing x=0 = LOSS.
DEFENCES (id name cost hp):
0 penguin 250 200hp: idle until hit, then slides forward through its row dealing 500 to EVERY enemy it touches, then gone. RARE: only for 2+ wizards/clowns bunched within 2 tiles in a losing row (status BREACH/critical; field "serious"). Never on a lone enemy, never at the opening. Penguins in other rows are dropped.
1 soldier 200 150hp: 40 dps at the first enemy anywhere ahead in its own row (unlimited range). One row only.
2 archer 300 100hp: 26 dps into ANY row; all archers focus the LOWEST row with enemies (row 0 first), nearest to base. Our main source of dps.
3 shield 150 700hp: no attack; enemies stop at the first defence in their row and their shots/beams hit it first. For tanking only.
ENEMIES: clown 200hp +15/s regen, stops 3.3 tiles before the first defence, throws 17 dps. monk 280hp, 2.3 tiles/s, melee 36 dps. wizard 500hp +15/s regen, stops 5.3 tiles before the first defence, beam 50 dps (switches to the next defence in range). pirate 30hp, 5.3 tiles/s (crosses in 1.9 s, faster than one decision), explodes for 200 on the first defence it reaches (or where it dies), radius 1.3 tiles: same tile, x+-1, and the tiles above/below.
STRATEGY: high dps from archers, few soldiers, shields only to tank, penguins rarely.
- No row may be EMPTY between decisions (a fast enemy would cross): an empty row gets its unit, archer@0 in bottom rows 0-2, soldier@1 in top rows 3-5, else a shield@6 if coins are short. No shields at the opening.
- More dps -> archers. A soldier only for a TOP row (3-5) that is short on dps (archers reach top rows last). Never two soldiers in one row.
- Tank with a shield (x<=6, closer than the nearest enemy) only when needed: breach, a front about to fall, a pirate about to reach our units, a row with 3+ enemies, exposed units with enemies coming in their row (these exposed-unit shields are added AUTOMATICALLY after your answer).
- Never put a shield past x=6 (clowns would stop off the board).
SCORE of a WON game /100: win 25, kills 25, defences still standing 25, speed 10, coins left 15. A LOST game scores only kills. Priorities: 1) never lose, 2) keep defences alive, 3) speed, 4) coins left.
PENGUINS go IN FRONT of the front defence (closer than the nearest enemy) so they fire now.
END OF WAVE (few enemies still to spawn): spend nothing on rows whose dps clears them before the line falls.
COINS: keep `reserve` coins banked for surprises; spend below it ONLY for rows with status BREACH or critical. Coins above the reserve should buy speed (archers) while enemies are still coming.
The user message has assess_board() output: per row enemy_hp, regen, our dps_here, enemy_dps_on_front, t_clear (s to kill the row), t_front / t_line (s our front / whole line survives), t_reach (s until the first enemy walks off the board if nothing stops it), status (BREACH: nothing stops them and they reach the base before dying; critical: line falls before they die; weak: front falls; ok; open: no defence and no enemy; quiet). Rule-based options are precomputed. Tools: row_detail(row) re-runs the model for one row with optional extra cards, check_plan(placements) validates cards and shows the new model.
"option" must be EXACTLY one of the keys of "options" in the message (anything else is ignored); for your own cards use "placements".
Answer with JSON only: {"option": "<name>", "reason": "<=12 words"} or {"placements": [[defenceId, x, y], ...], "reason": "..."}"""


def _from_triples(raw):
    cards = []
    for p in raw or []:
        try:
            if isinstance(p, dict):
                g = p.get("defenceGridPosition") or {}
                k, x, y = p.get("defenceId"), p.get("x", g.get("x")), p.get("y", g.get("y"))
            else:
                k, x, y = p
            cards.append(card(int(k), int(x), int(y)))
        except (TypeError, ValueError):
            continue
    return cards


def llm_decide(B, opts, default, briefing):
    if not USE_LLM or LLMClient is None:
        return None, "llm off"
    t0 = time.monotonic()
    deadline = min(T["llm_deadline"], B["budget_s"] - 3)
    if deadline < 1:
        return None, "no time"

    @tool
    def row_detail(row: int, extra: list = None) -> dict:
        """Combat model for one row, optionally with hypothetical extra cards [[defenceId, x, y], ...].

        Args:
            row: Board row, 0-5.
            extra: optional cards to add before modelling, as [defenceId, x, y].
        """
        ex = [tuple(_kxy(c)) for c in _from_triples(extra or [])]
        return assess(B, ex)["rows"][int(row)]

    @tool
    def check_plan(placements: list) -> dict:
        """Validate placements [[defenceId, x, y], ...]: accepted cards, cost, coins left vs reserve, new row statuses.

        Args:
            placements: list of [defenceId, x, y].
        """
        ok = check(_from_triples(placements), set(B["occ"]), B["coins"], B["costs"])
        left = B["coins"] - cost_of(ok, B["costs"])
        A = assess(B, [tuple(_kxy(c)) for c in ok]) if B["phase"] != "start" else None
        return {"accepted": [list(_kxy(c)) for c in ok], "cost": cost_of(ok, B["costs"]), "coins_left": left,
                "below_reserve": left < B["reserve"],
                "row_status": {y: r["status"] for y, r in A["rows"].items()} if A else None}

    msg = {
        "phase": B["phase"], "game_time": B["t"], "coins": B["coins"], "reserve": round(B["reserve"]),
        "enemies_still_to_spawn": B["queued"], "wave_size": B["total"], "reinforcements_later": bool(B["reinf"]),
        "assess_board": briefing,
        "options": {n: {"cost": cost_of(c, B["costs"]), "cards": [list(_kxy(x)) for x in c], "what": w}
                    for n, (c, w) in opts.items()},
        "rules_recommend": default,
    }
    client = LLMClient()
    # PUBLISH sandbox: any file open outside agent.py kills the call. requests' trust_env reads
    # ~/.netrc (and proxy/CA env files) before each request; the gateway is a local http relay.
    session = getattr(client, "_session", None)
    if session is not None:
        session.trust_env = False
    run = run_tool_loop(
        client,
        [{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(msg, separators=(",", ":"), default=str)}],
        [row_detail, check_plan],
        deadline_s=deadline,
        max_rounds=T["llm_rounds_start"] if B["phase"] == "start" else T["llm_rounds_pause"],
        max_tokens=T["llm_max_tokens"], temperature=0,
    )
    text = (run.content or "").strip()
    if "{" in text:
        text = text[text.index("{"): text.rindex("}") + 1]
    answer = json.loads(text)
    reason = str(answer.get("reason", ""))[:80]
    tools_used = [c.name for c in run.calls]
    if answer.get("option") in opts:
        cards, src = opts[answer["option"]][0], f"llm option={answer['option']}"
    elif isinstance(answer.get("placements"), list):
        cards, src = _from_triples(answer["placements"]), "llm custom"
    else:
        return None, f"llm unusable answer: {text[:80]}"
    return cards, f"{src} ({reason}) tools={tools_used} {time.monotonic() - t0:.1f}s"


def enforce_reserve(B, cards, critical_rows):
    """LLM plans keep the reserve too: below it, only cards for breach/critical rows survive.
    Also at most one soldier per row."""
    out, left = [], B["coins"]
    soldier_rows = {d["y"] for d in B["defs"] if d["k"] == SOLDIER}
    for c in cards:
        if c["defenceId"] == PENGUIN and _kxy(c)[2] not in B.get("penguin_rows", set(ROWS)):
            continue  # no penguins on a lone clown/monk
        if c["defenceId"] == SOLDIER:
            if _kxy(c)[2] in soldier_rows:
                continue
            soldier_rows.add(_kxy(c)[2])
        p = B["costs"].get(c["defenceId"], 999)
        if left - p >= B["reserve"] or _kxy(c)[2] in critical_rows:
            out.append(c); left -= p
    return out


# --- entry point ------------------------------------------------------------
def get_reinforcements(state):
    try:
        B = analyse(state)
        if B["phase"] == "start":
            try:
                log("STATE " + json.dumps(state, separators=(",", ":")))
            except Exception:
                pass
            opts, default, critical, briefing = opening_options(B)
            critical_rows = set(ROWS)
            B["penguin_rows"] = {y for y, t in B["threats"].items() if WIZARD in t}
        else:
            opts, default, critical, A = pause_options(B)
            briefing = A
            critical_rows = {y for y, r in A["rows"].items() if r["status"] in ("BREACH", "critical", "open")}
            B["penguin_rows"] = {y for y, r in A["rows"].items() if r.get("serious")}
            log("rows " + " | ".join(
                f"r{y} {r['status']} hp={r['enemy_hp']} dps={r.get('dps_here', 0)} clear={r.get('t_clear')}s "
                f"line={r.get('t_line')}s" for y, r in A["rows"].items() if r["status"] not in ("quiet",)))
        log(f"{B['phase']} t={B['t']} coins={B['coins']:.0f} reserve={B['reserve']:.0f} queued={B['queued']}/{B['total']} "
            + " ".join(f"{n}={cost_of(c, B['costs'])}" for n, (c, _) in opts.items()) + f" default={default}")
        if B["phase"] != "start" and not B["foes"] and not opts[default][0]:
            chosen, src = [], "nothing to do"
        else:
            try:
                chosen, src = llm_decide(B, opts, default, briefing)
            except Exception as exc:
                chosen, src = None, f"llm failed: {type(exc).__name__}: {str(exc)[:80]}"
            if chosen is None:
                chosen, src = opts[default][0], f"rules {default} ({src})"
            else:
                chosen = enforce_reserve(B, chosen, critical_rows)
            touched = {_kxy(c)[2] for c in chosen}
            # safety net, paid FIRST: fixes for rows the decision left untouched (TC02: the LLM's shield for a
            # quiet row ate the last 150 while the breached wizard row got nothing)
            chosen = [c for c in critical if _kxy(c)[2] not in touched] + chosen
            if B["phase"] == "start":  # minimum archers, whatever was chosen
                have = sum(1 for c in chosen if c["defenceId"] == ARCHER)
                tiles = {_kxy(c)[1:] for c in chosen}
                spare = [c for c in opts["lean"][0] if c["defenceId"] == ARCHER and _kxy(c)[1:] not in tiles]
                chosen = chosen + spare[: max(0, T["min_archers_open"] - have)]
        if B["phase"] != "start":
            # Mandatory, whatever was decided: shields in front of exposed archers/soldiers (incl. rows whose
            # shield was just destroyed), from coins above the reserve; not at the end of the wave unless
            # that row still has enemies.
            late = B["queued"] <= T["late_frac"] * B["total"]
            # a 150 shield guarding a 300 archer is worth dipping to half the reserve
            add = keep_reserve(fortify(B, chosen, late), B["coins"] - cost_of(chosen, B["costs"]),
                               T["archer_floor_frac"] * B["reserve"], B["costs"])
            if add:
                chosen = chosen + add
                src += f" +fortify {[_kxy(c)[1:] for c in add]}"
        out = check(chosen, set(B["occ"]), B["coins"], B["costs"])
        log(f"{src} -> placing {[_kxy(c) for c in out]}")
        return out
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        return []
