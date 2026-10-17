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
FOE_STOP = {CLOWN: 3.33, MONK: 0.5, WIZARD: 5.33, PIRATE: 0.0}   # tiles before the first defence where it stops
WIZARD_WINDUP = 1.75
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
    "min_archers_open": 3,    # pre-place at least this many archers at the opening when the reserve allows
    "archer_steal_s": 3.0,    # s a newcomer in a lower row pulls all archers away (focus model)
    "archer_share_min": 0.25, # never assume a row gets less than this share of the archers
    "max_archers_per_fix": 3, # archers bought to save one row before falling back to a shield
    "hold_margin": 0.85,      # a fix must clear the row within this share of the time its line holds
    "penguin_dense_hp": 500,  # 3+ bunched enemies holding at least this much hp: penguin
    "penguin_bunch_hp": 700,  # 2 bunched enemies: penguin once they hold this much hp (2 healthy wizards ~1000)
    "penguin_min_clear_s": 6, # ...unless our dps clears them within this anyway (don't waste it on the dying)
    "wizard_shield_min": 0.5, # opening: a row expecting at least this many wizards gets a shield@6 first
    "trap_wizard_min": 3.0,   # ...and with at least this many, a penguin@5 behind it (the trap)
    "wizard_reach": 5.33,     # tiles: a wizard beams defences this far in front of it (8 units)
    "urgent_s": 8.0,          # a row whose line falls within this is URGENT: may spend below the reserve. Others
                              # wait for the next decision (~interval + hold; unknown at a pause, assumed)
    "wall_dying_s": 8.0,      # a wall with less than this left may be replaced by a new shield
    "top_row": 3,             # rows >= this are "top": archers reach them last
    "fill_extra": 300,        # an empty, enemy-free row gets a blocker only with reserve + this much coin
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
def archer_share(B, y):
    """Share of the archers' time row y can count on. All archers fire at the LOWEST row with enemies, so
    while enemies are still queued, newcomers in lower rows keep pulling them away: the higher the row and
    the faster the spawns, the less archer time it gets. Spawn gap estimated as elapsed / spawned so far;
    rows unknown at a pause, so a newcomer lands below row y with chance y/6; each steals ~archer_steal_s."""
    if B["queued"] <= 0:
        return 1.0
    spawned = max(1, B["total"] - B["queued"])
    gap = max(0.5, float(B["t"] or 0) / spawned) if B["t"] else 3.0
    return max(T["archer_share_min"], 1.0 - (y / 6.0) * T["archer_steal_s"] / gap)


def assess(B, extra=()):
    """Per-row model of the live board (plus hypothetical `extra` cards [(kind, x, y)]; y=-1 adds an archer
    without a row). Every defence in a row blocks it (enemies stop at the first one, archers included)."""
    defs = B["defs"] + [{"k": k, "x": x, "y": y, "hp": {SHIELD: 700.0, ARCHER: 100.0, PENGUIN: 200.0}.get(k, 150.0),
                         "max": 1.0} for k, x, y in extra]
    archer_dps = DPS[ARCHER] * sum(1 for d in defs if d["k"] == ARCHER)
    rows, t_prev = {}, 0.0
    lowest = min((f["y"] for f in B["foes"]), default=None)
    for y in ROWS:
        foes = sorted((f for f in B["foes"] if f["y"] == y), key=lambda f: f["x"])
        mine = sorted((d for d in defs if d["y"] == y), key=lambda d: d["x"])
        walkers = [f for f in foes if f["k"] != PIRATE]
        hp = sum(f["hp"] for f in foes)
        regen = sum(REGEN.get(f["k"], 0) for f in foes)
        sold = sum(DPS[SOLDIER] for d in mine if d["k"] == SOLDIER and any(f["x"] >= d["x"] for f in foes))
        edps = sum(FOE_DPS[f["k"]] for f in walkers)
        front = max(mine, key=lambda d: d["x"]) if mine else None
        r = {"row": y, "enemies": [[FOE_NAME[f["k"]], round(f["hp"]), f["x"]] for f in foes],
             "defences": [[DEF_NAME[d["k"]], round(d["hp"]), d["x"]] for d in mine],
             "enemy_hp": round(hp), "regen": regen, "soldier_dps": sold, "enemy_dps_on_front": edps,
             "front": [DEF_NAME[front["k"]], round(front["hp"]), front["x"]] if front else None}
        if foes:
            share = archer_share(B, y)
            a_dps = archer_dps * share
            # archers sweep rows lowest-first: this row has only its soldiers until the lower rows clear
            hp_left = max(0.0, hp - max(0.0, sold - regen) * t_prev)
            net = sold + a_dps - regen
            t_clear = t_prev + hp_left / net if net > 0 else INF
            t_prev = t_clear if t_clear < INF else t_prev
            # before they hurt our front, enemies still have to walk into range (and wizards wind up)
            approach = min((max(0.0, f["x"] - front["x"] - FOE_STOP[f["k"]]) * 1.5 / FOE_SPEED[f["k"]]
                            + (WIZARD_WINDUP if f["k"] == WIZARD and not f["atk"] else 0.0)
                            for f in walkers), default=0.0) if front else 0.0
            t_front = approach + front["hp"] / edps if front and edps else INF
            t_line = approach + sum(d["hp"] for d in mine) / edps if edps else INF
            t_reach = min(f["x"] * 1.5 / FOE_SPEED[f["k"]] for f in foes)  # walk off the board if unblocked
            # a pirate blows up on our first defence unless our dps kills it on the way
            dps_now = sold + (a_dps if y == lowest else 0)
            pirate_hits = any(f["k"] == PIRATE and (dps_now <= 0 or f["hp"] / dps_now >
                                                   max(0.0, f["x"] - (front["x"] if front else 0)) * 1.5 / FOE_SPEED[PIRATE])
                              for f in foes) and front is not None and front["k"] != SHIELD
            if not mine:
                status = "ok" if t_clear < 0.7 * t_reach else "BREACH"
            elif t_clear > t_line or pirate_hits:
                status = "critical"
            elif t_clear > t_front * T["weak_margin"]:
                status = "weak"   # the front unit falls but the row holds: no spend
            else:
                status = "ok"
            # Penguin = density weapon (one slide: 500 to every enemy in the row): 3+ bunched within 2 tiles,
            # or 2 bunched holding real hp. Never for a lone enemy.
            wx = sorted((f["x"], f["hp"]) for f in walkers)
            bunch, bunch_hp = 0, 0.0
            for a, _ in wx:
                grp = [h for b, h in wx if 0 <= b - a <= 2]
                if (len(grp), sum(grp)) > (bunch, bunch_hp):
                    bunch, bunch_hp = len(grp), sum(grp)
            # Fire while the bunch is healthy and parked (wizards stop at the spawn in front of our x=6 shield):
            # waiting until the wall falls wastes the 500 on enemies our archers already chipped down.
            hold_t = t_line if mine else t_reach
            urgent = status == "BREACH" or (status == "critical" and hold_t < T["urgent_s"])
            # Go easy (aggressive penguins cost the winnable rounds): only a truly dense bunch, or a pair in a row
            # whose line falls before the next decision (incl. one our dps can never kill: TC01's stall).
            serious = t_clear > T["penguin_min_clear_s"] and (
                (bunch >= 3 and bunch_hp >= T["penguin_dense_hp"])
                or (bunch >= 2 and urgent and (bunch_hp >= T["penguin_bunch_hp"] or t_clear == INF)))
            r.update(dps_here=round(dps_now), archer_share=round(share, 2), serious=serious, bunch=bunch, urgent=urgent,
                     t_clear=_r(t_clear), t_front=_r(t_front), t_line=_r(t_line), t_reach=_r(t_reach),
                     status=status, nearest_enemy_x=foes[0]["x"])
        else:
            r.update(status="open" if not mine else "quiet")
        rows[y] = r
    return {"rows": rows, "archer_dps_total": archer_dps,
            "max_t_clear": max([r.get("t_clear") or 0 for r in rows.values()] + [0])}


def _pick(xs, y, occ, before=MAX_X + 1):
    return next((x for x in xs if 0 <= x < before and x <= MAX_X and (x, y) not in occ), None)


def has_wall(B, y):
    """The row already has a shield/penguin that will still stand for wall_dying_s against the enemies there.
    Then it never gets a second shield (stacking burned the coins on the "12 wizards" case); a wall about to
    fall may be replaced."""
    edps = sum(FOE_DPS[f["k"]] for f in B["foes"] if f["y"] == y and f["k"] != PIRATE)
    return any(d["k"] in (SHIELD, PENGUIN) and (edps <= 0 or d["hp"] / edps >= T["wall_dying_s"])
               for d in B["defs"] if d["y"] == y)


def _fixed(A, y):
    """Holds with a safety margin: the model's dps numbers are approximate (guide: rates come from animations)."""
    r = A["rows"][y]
    lim = min(x for x in (r.get("t_line"), r.get("t_reach") if not r.get("defences") else None, INF) if x is not None)
    return r["status"] not in ("BREACH", "critical") and (r.get("t_clear") or 0) <= T["hold_margin"] * lim


def row_plan(B, y, r, occ):
    """What one row needs. Returns (must, optional, why, dps) where dps is None or
    (archers_alone, shield_card, archers_with_shield): the fewest extra archers that make the row hold on
    their own (None if 3 aren't enough), else a blocking shield plus the archers still needed with it."""
    must, opt, why = [], [], []
    foes = [f for f in B["foes"] if f["y"] == y]
    nearest = min((f["x"] for f in foes), default=MAX_X + 1)
    kinds = {d["k"] for d in B["defs"] if d["y"] == y}
    front_x = max((d["x"] for d in B["defs"] if d["y"] == y), default=-1)
    status = r["status"]

    def penguin(lst):
        x = _pick([*range(nearest - 1, front_x, -1), *range(front_x - 1, 0, -1)], y, occ, nearest)
        if x is not None:  # just in front of the bunch: the first throw/beam/strike sets it off
            lst.append(card(PENGUIN, x, y)); occ.add((x, y)); why.append(f"r{y} penguin vs bunch of {r.get('bunch')}")
            return True
        return False

    wiz = [f for f in foes if f["k"] == WIZARD]
    trap_armed = PENGUIN in kinds and wiz
    if trap_armed:
        return must, opt, why, None  # armed: let it fire on its own, never wall it off further
    if wiz and r.get("serious"):  # the trap only for the same dense/urgent cases; else the normal dps logic
        # Wizard rows: archers (26 dps vs 500 hp + 15/s regen) and more shields only delay; a penguin kills
        # EVERY enemy in the row. So keep a TRAP: a shield in front, a penguin right behind it. Wizards park at
        # the spawn beaming the shield, the bunch grows, and when the shield dies the beam switches to the
        # penguin with no wind-up: it slides through the whole bunch by itself. (TC11: 5 shields died to stacked
        # wizards while 2 archers chipped at them.)
        reach = math.ceil(min(f["x"] for f in wiz) - T["wizard_reach"])  # leftmost tile the beam reaches
        walls = sorted(d["x"] for d in B["defs"] if d["y"] == y and d["k"] == SHIELD)
        lst = must if r.get("urgent") else opt  # below the reserve only when urgent
        if walls:
            px = _pick(range(walls[-1] - 1, max(reach, 1) - 1, -1), y, occ, nearest)
            if px is not None:
                lst.append(card(PENGUIN, px, y)); occ.add((px, y)); why.append(f"r{y} arm trap: penguin behind the shield")
            return must, opt, why, None
        sx = _pick(range(min(T["shield_x"], nearest - 1), 1, -1), y, occ, nearest)
        px = _pick(range((sx or nearest) - 1, max(reach, 1) - 1, -1), y, occ | ({(sx, y)} if sx else set()), nearest)
        trap = ([card(SHIELD, sx, y)] if sx is not None else []) + ([card(PENGUIN, px, y)] if px is not None else [])
        if r.get("bunch", 0) >= 2:
            trap.sort(key=lambda c: c["defenceId"] != PENGUIN)  # a bunch is already here: the penguin first
        for c in trap:
            lst.append(c); occ.add(tuple(_kxy(c)[1:]))
        why.append(f"r{y} rebuild trap (shield + penguin)")
        return must, opt, why, None
    if r.get("serious") and PENGUIN not in kinds:
        if penguin(must if r.get("urgent") else opt):  # below the reserve only when urgent
            return must, opt, why, None
    if status not in ("BREACH", "critical"):
        return must, opt, why, None   # blocked and our dps gets there first: nothing to buy
    if any(f["k"] == PIRATE for f in foes) and not any(f["k"] != PIRATE for f in foes) \
            and not any(d["k"] == SHIELD for d in B["defs"] if d["y"] == y):
        x = _pick(range(min(T["shield_x"], nearest - 1), front_x, -1), y, occ, nearest)
        if x is not None:  # only pirates: a shield in front takes the blast instead of an archer
            must.append(card(SHIELD, x, y)); occ.add((x, y)); why.append(f"r{y} pirate->shield")
        return must, opt, why, None
    alone = next((n for n in range(1, T["max_archers_per_fix"] + 1)
                  if _fixed(assess(B, [(ARCHER, 0, -1)] * n), y)), None)
    walled = has_wall(B, y)
    # one shield per row: a row that already has its wall gets dps (or a penguin above), never a 2nd shield
    sx = None if walled else _pick(range(min(T["shield_x"], nearest - 1), -1, -1), y, occ, nearest)
    shield = card(SHIELD, sx, y) if sx is not None else None
    with_shield = 0
    if shield is not None:
        with_shield = next((n for n in range(0, T["max_archers_per_fix"] + 1)
                            if _fixed(assess(B, [(SHIELD, sx, y)] + [(ARCHER, 0, -1)] * n), y)), T["max_archers_per_fix"])
    return must, opt, why, (alone, shield, with_shield)


def archer_slots(B, occ, threats=None):
    """Archer tiles: x=0 first, in rows no enemy can use (opening), then bottom rows up (archers fire at
    the lowest row first, so bottom rows defend themselves); then x=2 in rows walled at x>=3."""
    quiet = [y for y in ROWS if threats is not None and y not in threats]
    out = [(0, y) for y in quiet + [y for y in ROWS if y not in quiet] if (0, y) not in occ]
    # second column: TOP rows first (archers clear the bottom rows first, so extra hp is worth more up top)
    out += [(2, y) for y in reversed(ROWS) if y not in quiet and (2, y) not in occ
            and any((x, y) in occ for x in range(3, MAX_X + 1))]
    out += [(1, y) for y in reversed(ROWS) if y not in quiet and (1, y) not in occ]
    return out


def row_unit(y):
    """What occupies an otherwise empty row: archers in the bottom rows (archers fire at the lowest row
    first, so they guard their own), shields in the top rows (archers reach those last; a 700hp shield
    holds there and survives 3 pirate blasts, where a soldier dies to one)."""
    return (SHIELD, None) if y >= T["top_row"] else (ARCHER, 0)


def shield_x(B, y, tiles):
    """x=6 by default; if an enemy in this row is already at/inside 6, the furthest free tile in front of it."""
    nearest = min((f["x"] for f in B["foes"] if f["y"] == y), default=MAX_X + 1)
    return _pick(range(min(T["shield_x"], nearest - 1), 0, -1), y, tiles, nearest)


def fill_empty(B, cards, rows, floor, units=True):
    """No row left empty (a pirate crosses faster than one decision): each row of `rows` with nothing in
    it gets its unit (archer bottom rows while coins stay above `floor`; a shield on top, or when short)."""
    tiles, left, out = _tiles(B, cards), B["coins"] - cost_of(cards, B["costs"]), []
    # top rows first: when coins run short, the rows archers reach last get covered first
    empty = sorted((y for y in rows if not any(t[1] == y for t in tiles)), key=lambda y: -y)
    shield_p = B["costs"].get(SHIELD, 150)
    for i, y in enumerate(empty):
        kind, x = row_unit(y)
        p = B["costs"].get(kind, 999)
        later = shield_p * (len(empty) - i - 1)  # covering every row beats a better unit in one row
        if kind != SHIELD and units and left - p - later >= floor and (x, y) not in tiles:
            c = card(kind, x, y)
        else:
            sx = shield_x(B, y, tiles)
            if sx is None or left - shield_p < floor:  # fillers never dip into the reserve
                continue
            c, p = card(SHIELD, sx, y), shield_p
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
    """Archers only (dps first): an archer@0 in every threatened row. If the coins can't do that, the
    densest (then top) rows get a 150 shield instead, so no row is open. Spare coins above the reserve
    -> more archers. Shields for tanking come at the pauses, once enemies are seen."""
    threats, costs, coins, res = B["threats"], B["costs"], B["coins"], B["reserve"]
    if not B["reinf"]:
        return opening_no_pauses(B)
    pa, ps = costs.get(ARCHER, 300), costs.get(SHIELD, 150)
    # Wizard rows: a shield@6 first. Without it a wizard walks up to 5.3 tiles from the first defence (an
    # archer at x=0) and beams it dead in ~2 s; with it the wizard stops at the spawn (x~10), its beam can't
    # reach x<=4, and wizards pile up there for a penguin. (Public "12 wizards" case: lost an archer, then the row.)
    wizard_rows = [y for y in sorted(threats) if threats[y].get(WIZARD, 0) >= T["wizard_shield_min"]]
    walls = [card(SHIELD, T["shield_x"], y) for y in wizard_rows]
    # wizard-heavy rows: a penguin behind that shield (the trap), so a whole parked bunch dies in one slide
    walls += [card(PENGUIN, T["shield_x"] - 1, y) for y in wizard_rows if threats[y].get(WIZARD, 0) >= T["trap_wizard_min"]]
    rows = [y for y in sorted(threats) if y not in wizard_rows]
    budget = coins - res - cost_of(walls, costs)  # archers keep the full reserve
    # as many archers as possible while every threatened row still gets at least a shield
    k = max([k for k in range(len(rows) + 1) if k * pa + (len(rows) - k) * ps <= budget], default=0)
    dens = {y: sum(threats[y].values()) for y in rows}
    shield_rows = set(sorted(rows, key=lambda y: (-dens[y], -y))[: len(rows) - k])  # densest, then top
    cover = walls + [card(ARCHER, 0, y) for y in rows if y not in shield_rows]
    for y in sorted(shield_rows, key=lambda y: (-dens[y], -y)):
        if cost_of(cover, costs) + ps <= coins:
            cover.append(card(SHIELD, T["shield_x"], y))
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
            lim = floor  # the reserve holds at the opening: nothing is projected to cross yet
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
    # safety net for rows a decision leaves empty: the cheapest blocker only (archers there would break the reserve)
    blockers = [card(SHIELD, T["shield_x"], y) for y in sorted(threats)]
    return opts, "fortress", blockers, {"wave_hp": wave_hp, "archers_suggested": n_base,
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
def affordable_fallback(B, A, cards, occ, owner=None):
    """A breached/critical row whose fixes can't be paid for gets the cheapest card that still helps:
    a shield (150) in front of its line, else behind it. (TC02: 150 coins, a wizard row critical, the
    penguin+soldier plan unaffordable -> nothing was placed and the game was lost.)"""
    left, paid_rows = B["coins"], set()
    for c in cards:
        p = B["costs"].get(c["defenceId"], 999)
        if p <= left:
            left -= p; paid_rows.add((owner or {}).get(tuple(_kxy(c)[1:]), _kxy(c)[2]))
    out = list(cards)
    for y, r in A["rows"].items():
        walled = has_wall(B, y) or any(d["k"] == PENGUIN for d in B["defs"] if d["y"] == y)  # armed trap: leave it
        if r.get("urgent") and y not in paid_rows and not walled \
                and B["costs"].get(SHIELD, 150) <= left:
            nearest = r.get("nearest_enemy_x", MAX_X + 1)
            front = max((d["x"] for d in B["defs"] if d["y"] == y and d["k"] != ARCHER), default=-1)
            x = _pick([*range(min(T["shield_x"], nearest - 1), front, -1), *range(front - 1, -1, -1)], y, occ, nearest)
            if x is not None:
                out.insert(0, card(SHIELD, x, y)); occ.add((x, y)); left -= B["costs"].get(SHIELD, 150)
    return out


def pause_options(B):
    """For every row the model projects to fall: DPS first (archers from the coins above the reserve, if
    enough of them make the row hold), else a blocking shield (+ the archers still needed with it). Then
    penguins for dense bunches, and a blocker in any row left empty."""
    costs, res = B["costs"], B["reserve"]
    pa = costs.get(ARCHER, 300)
    A = assess(B)
    occ = set(B["occ"])
    order = sorted(ROWS, key=lambda y: ["BREACH", "critical", "weak", "ok", "open", "quiet"].index(A["rows"][y]["status"]))
    must, optional, why = [], [], []
    spare = B["coins"] - res
    need_arch, arch_for = 0, []  # arch_for: the row each bought archer is for (archers sit in other rows)
    for y in order:
        m, o, w, dps = row_plan(B, y, A["rows"][y], occ)
        must += m; optional += o; why += w
        spare -= cost_of(m, costs)
        if dps is None:
            continue
        alone, shield, with_shield = dps
        if alone is not None and alone * pa <= spare:  # dps from spare coins: always fine
            need_arch += alone; spare -= alone * pa; arch_for += [y] * alone
            why.append(f"r{y} +{alone} archer (enough dps, no shield needed)")
        elif not A["rows"][y].get("urgent"):
            # losing, but its line outlasts the next decision: keep the reserve, look again then
            # (public "12 wizards": an archer bought here left no coins for the shield that fell next)
            why.append(f"r{y} critical but holds past the next decision: wait")
        elif shield is not None:
            must.append(shield); occ.add(tuple(_kxy(shield)[1:])); spare -= costs.get(SHIELD, 150)
            need_arch += with_shield; spare -= with_shield * pa; arch_for += [y] * with_shield
            why.append(f"r{y} shield" + (f" + {with_shield} archer" if with_shield else "") + " (archers alone can't hold it)")
        elif alone is not None:
            need_arch += alone; arch_for += [y] * alone; why.append(f"r{y} +{alone} archer (no room for a shield)")
    room = max(0, T["max_archers"] - sum(1 for d in B["defs"] if d["k"] == ARCHER))
    slots = [card(ARCHER, x, y) for x, y in archer_slots(B, occ)]
    bought = slots[:min(need_arch, room)]
    owner = {tuple(_kxy(c)[1:]): y for c, y in zip(bought, arch_for)}
    must += bought
    spare_slots = slots[min(need_arch, room): room]
    must = affordable_fallback(B, A, must, occ, owner)
    late = B["queued"] <= T["late_frac"] * B["total"]  # end of wave: only what the projection demands
    # Empty rows get a blocker from coins above the reserve only (a pause can't know which rows the level
    # uses: the "12 wizards" case paid for a shield in a row no enemy could ever enter). Fillers come before
    # optional archers, so speed doesn't leave rows open.
    # Every threatened row was covered at the opening, so a row with nothing in it and no enemy is most likely
    # one the level never uses: fill it only with comfortable coins (reserve + fill_extra).
    fill = fill_empty(B, must, ROWS, res + T["fill_extra"], True)
    hold = must + fill_empty(B, must, ROWS, res + T["fill_extra"], False)
    base = must + fill + ([] if late else keep_reserve(optional, B["coins"] - cost_of(must + fill, costs), res, costs))
    speed = base + ([] if late else keep_reserve(spare_slots, B["coins"] - cost_of(base, costs), res, costs))
    opts = {
        "speed": (speed, f"needed + archers with the coins above the {res:.0f} reserve (faster kills, all rows)"),
        "needed": (base, ("END OF WAVE: only rows projected to fall. " if late else "")
                   + "fixes: " + (", ".join(why) or "nothing needed")),
        "hold": (hold, "only rows projected to fall + a blocker in empty rows if coins allow, keep the reserve"),
    }
    default = "speed" if len(speed) > len(base) else "needed"
    return opts, default, hold, A, must


# --- LLM decision -----------------------------------------------------------
PROMPT = """You command a lane-defence game. Grid x 0-10 (base x=0; enemies enter at x=10 and walk left), rows y 0-5, 1 tile = 1.5 units. Enemies never change rows. ONE enemy crossing x=0 = LOSS.
DEFENCES (id name cost hp):
0 penguin 250 200hp: idle until hit, then slides forward through its row dealing 500 to EVERY enemy it touches, then gone. A density weapon: for bunched enemies (field "serious"). Never on a lone enemy, never at the opening. Penguins in other rows are dropped.
1 soldier 200 150hp: 40 dps at the first enemy anywhere ahead in its own row (unlimited range). One row only.
2 archer 300 100hp: 26 dps into ANY row; all archers focus the LOWEST row with enemies (row 0 first), nearest to base. Our main source of dps.
3 shield 150 700hp: no attack; enemies stop at the first defence in their row and their shots/beams hit it first. For tanking only.
ENEMIES: clown 200hp +15/s regen, stops 3.3 tiles before the first defence, throws 17 dps. monk 280hp, 2.3 tiles/s, melee 36 dps. wizard 500hp +15/s regen, stops 5.3 tiles before the first defence, beam 50 dps (switches to the next defence in range). pirate 30hp, 5.3 tiles/s (crosses in 1.9 s, faster than one decision), explodes for 200 on the first defence it reaches (or where it dies), radius 1.3 tiles: same tile, x+-1, and the tiles above/below.
STRATEGY, in priority order: (1) maximise dps with archers; (2) shields so no row is ever empty; (3) penguins for density; (4) shields to tank for time while the archers at the back work. Soldiers only hit their own row: avoid them.
- Opening: archers only, one @0 in every threatened row; only if coins can't cover that, shields in the densest/top rows. No other shields at the opening.
- No row may be EMPTY between decisions (a fast enemy would cross): an empty row gets archer@0 in bottom rows 0-2, a shield in top rows 3-5.
- Shield position: x=6 by default; if an enemy in that row is already at/inside x=6, the furthest free tile in front of it. Cards at or behind an enemy are moved in front of it automatically.
- A row with any defence is BLOCKED (enemies stop at the first defence, archers included). If our dps clears it before that line falls, it needs nothing: no shield.
- Archer focus: all archers fire at the lowest row with enemies; a higher row only gets them once the rows below are clear, and while enemies are still queued newcomers in lower rows keep pulling them away (field archer_share). So shields, when needed, are mostly needed in the TOP rows.
- A row projected to fall (BREACH/critical): first buy archers with the coins above the reserve if enough of them make it hold; only otherwise a shield (+ the archers still needed).
- Penguins are expensive: use them only for a dense bunch (3+) or a pair in a row whose line falls before the next decision (field "serious"). For wizards that is a TRAP: a penguin right behind the front shield; when the shield dies the beam jumps to it and it slides through the bunch. Never wall off an armed penguin.
- Penguin: dense bunches only (3+ within 2 tiles, or 2+ in a row projected to fall; field "serious"). Place it just in front of the bunch.
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


def in_front(B, cards):
    """A card at/behind the nearest enemy of its row is useless (enemies walk past it, soldiers ignore
    what is behind them): shields/penguins move to the furthest free tile in front of that enemy,
    archers/soldiers to the free tile nearest the base."""
    tiles, out = set(B["occ"]), []
    for c in cards:
        k, x, y = _kxy(c)
        nearest = min((f["x"] for f in B["foes"] if f["y"] == y), default=MAX_X + 1)
        if x >= nearest:
            xs = range(nearest - 1, 0, -1) if k in (SHIELD, PENGUIN) else range(0, nearest)
            nx = _pick(xs, y, tiles, nearest)
            if nx is None:
                continue
            c, x = card(k, nx, y), nx
        tiles.add((x, y)); out.append(c)
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
            must = []
            critical_rows = set()  # opening: nothing is projected to cross, the reserve holds
            B["penguin_rows"] = {y for y, t in B["threats"].items() if WIZARD in t}
        else:
            opts, default, critical, A, must = pause_options(B)
            briefing = A
            # only rows the model projects to fall may spend below the reserve
            critical_rows = {y for y, r in A["rows"].items() if r["status"] in ("BREACH", "critical")}
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
            # Safety net, paid FIRST: (1) fixes for rows projected to fall, in their own priority order (the
            # blocking shield before the archers bought for that row); (2) blockers for rows the decision left
            # empty; then the decision. (Public TC04: archers meant for a breached row were paid first and its
            # shield was dropped; the wizard crossed.)
            tiles = {tuple(_kxy(c)[1:]) for c in must}
            chosen = [c for c in chosen if tuple(_kxy(c)[1:]) not in tiles]
            touched = {_kxy(c)[2] for c in must + chosen}
            tiles |= {tuple(_kxy(c)[1:]) for c in chosen}
            fill = [c for c in critical if _kxy(c)[2] not in touched and tuple(_kxy(c)[1:]) not in tiles]
            chosen = list(must) + fill + chosen
            if B["phase"] == "start":  # minimum archers, whatever was chosen
                have = sum(1 for c in chosen if c["defenceId"] == ARCHER)
                tiles = {_kxy(c)[1:] for c in chosen}
                spare = [c for c in opts["lean"][0] if c["defenceId"] == ARCHER and _kxy(c)[1:] not in tiles]
                chosen = chosen + spare[: max(0, T["min_archers_open"] - have)]
        out = check(in_front(B, chosen), set(B["occ"]), B["coins"], B["costs"])
        log(f"{src} -> placing {[_kxy(c) for c in out]}")
        return out
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        return []
