"""Rules agent for the GDG NSUT Autopilot lane-defence game.

IMPORTANT (read from the SDK source, autopilot_sdk/runtime/reinforcements.py):
every call runs in a FRESH subprocess loaded from a temp copy of this file, and
a PUBLISH run's sandbox forbids opening any other file (a violation ends the run).
So nothing survives between calls: no module memory, no file cache. The opening
builds every row that cannot wait (Pirates ~1.9 s and Monks ~2.1 s cross the board,
faster than the 3 s pause interval); pauses react to the live board only.

Plan per threatened row (x = column, 0 = base):
  pirate row: Shield bait @3/@5 (blast radius 2 u, kept away from everything) + Soldier @1
  wizard row: Penguin @2 (the beam triggers the slide: 500 dmg kills a full Wizard) + Soldier @1
  other rows: Shield @2 in front of Soldier @1
Pauses: emergency blockers, replace used Penguins vs live Wizards, re-shield worn fronts.
"""
import json
import sys

# --- game facts -------------------------------------------------------------
PENGUIN, SOLDIER, ARCHER, SHIELD = 0, 1, 2, 3
CLOWN, MONK, WIZARD, PIRATE = 0, 1, 2, 3
ENEMY_IDS = {"EnemClown": CLOWN, "EnemyMonk": MONK, "EnemyWizard": WIZARD, "EnemyPirate": PIRATE}
DEF_IDS = {"defencePenguin": PENGUIN, "defenceSolider": SOLDIER, "defenceArcher": ARCHER, "defenceShield": SHIELD}
DEFAULT_COSTS = {PENGUIN: 250, SOLDIER: 200, ARCHER: 300, SHIELD: 150}
ROWS = range(6)
MAX_X = 10

# --- tuning -----------------------------------------------------------------
T = {
    "soldier_x": 1,
    "front_x": 2,               # shield / penguin in front of the soldier
    "pirate_bait_x": (3, 5),    # even rows, odd rows: adjacent pirate rows' blasts never overlap
    "reshield_frac": 0.35,      # add a shield when the front defence drops below this
    "archer_spare": 1200,       # coins left after the core plan before archers are worth it
    "max_archers": 2,
    "shield_per_melee": 3,      # reinforcements off: one extra shield per N clowns/monks in a row
}


def log(msg):
    try:
        print(f"[agent] {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _xy(u):
    return int(round(u["position"]["x"])), int(round(u["position"]["y"]))


def _card(kind, x, y):
    return {"defenceId": kind, "defenceGridPosition": {"x": x, "y": y}}


def check(cards, taken, coins, costs):
    """Keep cards that are on the board, on a free tile, not duplicated and affordable in order."""
    out = []
    for c in cards:
        kind, p = c["defenceId"], c["defenceGridPosition"]
        x, y = p["x"], p["y"]
        if kind in costs and 0 <= x <= MAX_X and 0 <= y <= 5 and (x, y) not in taken and costs[kind] <= coins:
            taken.add((x, y))
            coins -= costs[kind]
            out.append(c)
    return out


def _board(state):
    costs = {int(k): int(v) for k, v in (state.get("unitCosts") or {}).items()} or dict(DEFAULT_COSTS)
    coins = float((state.get("budget") or {}).get("remaining", 0))
    defences = [d for d in state.get("defences") or [] if d.get("health", 0) > 0]
    enemies = [e for e in state.get("enemies") or [] if not e.get("isDead") and e.get("health", 0) > 0]
    return costs, coins, defences, enemies


# --- opening ----------------------------------------------------------------
def row_threats(state):
    """{row: {enemyId: expected count}}. Missing allowedRows means any row."""
    threats = {y: {} for y in ROWS}
    for g in state.get("upcomingEnemies") or []:
        eid, n = int(g.get("enemyId", 0)), int(g.get("count", 0))
        rows = [r for r in (g.get("allowedRows") or ROWS) if r in threats]
        for r in rows:
            threats[r][eid] = threats[r].get(eid, 0) + n / len(rows)
    for e in state.get("enemies") or []:  # the enemy already on the field
        eid = ENEMY_IDS.get(e.get("type"))
        if eid is not None:
            r = _xy(e)[1]
            threats[r][eid] = max(threats[r].get(eid, 0), 1)
    return {y: t for y, t in threats.items() if t}


def opening(state, coins, costs):
    threats = row_threats(state)
    reinf = (state.get("gameSettings") or {}).get("reinforcementsEnabled", True)
    sx, fx = T["soldier_x"], T["front_x"]
    # Pirate rows first (they outrun a pause), then monk rows, then wizards.
    order = sorted(threats, key=lambda y: (PIRATE not in threats[y], MONK not in threats[y],
                                           WIZARD not in threats[y], y))

    first, second, extra = [], [], []
    for y in order:
        t = threats[y]
        if PIRATE in t:
            first.append(_card(SOLDIER, sx, y))
            second.append(_card(SHIELD, T["pirate_bait_x"][y % 2], y))
        elif WIZARD in t and MONK not in t:
            first.append(_card(PENGUIN, fx, y))
            second.append(_card(SOLDIER, sx, y))
        else:
            first.append(_card(SOLDIER, sx, y))
            second.append(_card(SHIELD, fx, y))
        if not reinf:  # no second chance: stack what each row will need
            nx = fx + 1
            wiz = int(t.get(WIZARD, 0) + 0.99) - (1 if (WIZARD in t and MONK not in t and PIRATE not in t) else 0)
            for _ in range(max(0, wiz)):
                extra.append(_card(PENGUIN, nx if PIRATE not in t else 2, y)); nx += 1
            melee = t.get(CLOWN, 0) + t.get(MONK, 0)
            for _ in range(int(melee // T["shield_per_melee"])):
                extra.append(_card(SHIELD, nx if PIRATE not in t else 0, y)); nx += 1

    plan = first + second + extra
    cost = sum(costs.get(c["defenceId"], 999) for c in plan)
    # Archers sit at x=0 in rows no enemy can use; they focus whatever is lowest/nearest.
    safe = [y for y in ROWS if y not in threats]
    wizards = sum(t.get(WIZARD, 0) for t in threats.values())
    if safe and (not reinf or wizards >= 2) and coins - cost >= T["archer_spare"]:
        plan += [_card(ARCHER, 0, y) for y in safe[: T["max_archers"]]]
    log(f"opening threats={ {y: {k: round(v, 1) for k, v in t.items()} for y, t in threats.items()} } "
        f"reinf={reinf} plan_cost={cost} coins={coins:.0f}")
    return plan


# --- pauses -----------------------------------------------------------------
def pause(state, coins, defences, enemies, taken):
    queued = (state.get("counts") or {}).get("enemiesQueued", 0)
    plan, why = [], []
    by_row = {y: [] for y in ROWS}
    for e in enemies:
        by_row.setdefault(_xy(e)[1], []).append(e)
    for y in ROWS:
        foes = by_row[y]
        if not foes:
            continue
        mine = sorted((d for d in defences if _xy(d)[1] == y), key=lambda d: _xy(d)[0])
        kinds = {DEF_IDS.get(d["type"]) for d in mine}
        nearest = min(_xy(e)[0] for e in foes)
        types = {ENEMY_IDS.get(e["type"]) for e in foes}

        def free(x):
            return 0 <= x < nearest and (x, y) not in taken

        if not mine:  # emergency: nothing stands between this row and the base
            px = next((x for x in (2, 1, 0, 3) if free(x)), None)
            if px is not None:
                plan.append(_card(PENGUIN, px, y)); taken.add((px, y))
            sx = next((x for x in (1, 0) if free(x)), None)
            if sx is not None:
                plan.append(_card(SOLDIER, sx, y)); taken.add((sx, y))
            why.append(f"row{y} EMPTY vs {sorted(types)} @x{nearest}")
            continue
        if WIZARD in types and PENGUIN not in kinds:
            front = _xy(mine[-1])[0]
            px = next((x for x in range(front + 1, min(nearest, 5)) if free(x)), None)
            if px is None:
                px = next((x for x in range(front - 1, -1, -1) if free(x)), None)
            if px is not None:
                plan.append(_card(PENGUIN, px, y)); taken.add((px, y))
                why.append(f"row{y} penguin vs wizard")
        if not kinds & {SOLDIER, ARCHER, PENGUIN}:
            sx = next((x for x in (1, 0, 2) if free(x)), None)
            if sx is not None:
                plan.append(_card(SOLDIER, sx, y)); taken.add((sx, y))
                why.append(f"row{y} no killer")
        front = mine[-1]
        if (queued or len(foes) > 1) and DEF_IDS.get(front["type"]) != PENGUIN and PIRATE not in types \
                and front["health"] < T["reshield_frac"] * front.get("maxHealth", front["health"]):
            fx = _xy(front)[0] + 1
            if free(fx):
                plan.append(_card(SHIELD, fx, y)); taken.add((fx, y))
                why.append(f"row{y} front worn")
    log(f"pause t={state.get('elapsedSeconds')} coins={coins:.0f} queued={queued} "
        f"enemies={[[e['type'][4:9], round(e['health']), *_xy(e)] for e in enemies]} "
        f"defs={[[d['type'][7:12], round(d['health']), *_xy(d)] for d in defences]} -> {why or 'save coins'}")
    return plan


def get_reinforcements(state):
    try:
        costs, coins, defences, enemies = _board(state)
        taken = {_xy(d) for d in defences}
        if state.get("phase") == "start":
            try:
                log("STATE " + json.dumps(state, separators=(",", ":")))
            except Exception:
                pass
            plan = opening(state, coins, costs)
        else:
            plan = pause(state, coins, defences, enemies, set(taken))
        out = check(plan, taken, coins, costs)
        log(f"placing {[(c['defenceId'], c['defenceGridPosition']['x'], c['defenceGridPosition']['y']) for c in out]}")
        return out
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        return []
