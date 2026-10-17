"""Offline harness: python3 test_agent.py  (valid, affordable, fast, never crashes; LLM path faked)."""
import json
import time

import agent

COSTS = {"0": 250, "1": 200, "2": 300, "3": 150}


def u(kind, x, y, hp, maxhp=None, **kw):
    return {"id": f"{kind}_{x}{y}", "type": kind, "health": hp, "maxHealth": maxhp or hp,
            "position": {"x": x, "y": y}, **kw}


def foe(kind, x, y, hp):
    return u(kind, x, y, hp, isAttacking=False, isDead=False)


def opening(wave, coins=3000, reinf=True, first=("EnemClown", 10, 2, 200)):
    return {"phase": "start", "timeBudgetSeconds": 7, "sessionID": "t", "elapsedSeconds": 0.4,
            "counts": {"activeEnemyCount": 1, "enemiesQueued": 9, "totalEnemiesInWave": 10},
            "defences": [], "enemies": [foe(*first)], "unitCosts": COSTS,
            "budget": {"remaining": coins, "spent": 0, "unlocked": coins, "total": coins + 500},
            "upcomingEnemies": wave,
            "spawnSettings": {"interArrivalTime": 6, "timeDecreaseRate": 0.25, "minInterArrivalTime": 1},
            "gameSettings": {"reinforcementsEnabled": reinf, "reinforcementIntervalSeconds": 3}}


def pause(defences, enemies, coins=1000, queued=5):
    return {"phase": "pause", "timeBudgetSeconds": 7, "sessionID": "t", "elapsedSeconds": 12,
            "counts": {"activeEnemyCount": len(enemies), "enemiesQueued": queued, "totalEnemiesInWave": 10},
            "defences": defences, "enemies": enemies, "unitCosts": COSTS,
            "budget": {"remaining": coins, "spent": 0, "unlocked": coins, "total": 5000}}


TC01 = opening([{"enemyId": 0, "count": 14, "allowedRows": [0, 1, 2, 3, 4, 5]},
                {"enemyId": 1, "count": 4, "allowedRows": [0, 1, 2, 3, 4, 5]}], coins=5000, first=("EnemClown", 10, 3, 200))
TC01_STALL = pause([u("defenceShield", 2, 3, 160, 700), u("defenceSolider", 1, 3, 150)],
                   [foe("EnemClown", 6, 3, 117), foe("EnemClown", 6, 3, 80), foe("EnemClown", 6, 3, 192)], coins=2900, queued=0)
CASES = {
    "tc01-opening": TC01,
    "wizard-heavy": opening([{"enemyId": 2, "count": 6, "allowedRows": [0, 1, 2]}], first=("EnemyWizard", 10, 1, 500)),
    "pirate-heavy": opening([{"enemyId": 3, "count": 8, "allowedRows": [2, 3, 4, 5]}], first=("EnemyPirate", 10, 3, 30)),
    "all-rows-unknown": opening([{"enemyId": i, "count": 3, "allowedRows": None} for i in range(4)]),
    "low-coins": opening([{"enemyId": i, "count": 3, "allowedRows": [i, 5]} for i in range(4)], coins=400),
    "reinf-off": opening([{"enemyId": 0, "count": 4, "allowedRows": [1, 2, 3]}, {"enemyId": 2, "count": 3, "allowedRows": [0, 1]},
                          {"enemyId": 1, "count": 5, "allowedRows": [4]}], coins=3900, reinf=False),
    "missing-keys": {"phase": "start", "budget": {"remaining": 1300}},
    "garbage": {"phase": "pause", "enemies": [{"type": "EnemyMonk"}]},
    "pause-empty-row": pause([], [foe("EnemyMonk", 6, 3, 280)]),
    "pause-wizard": pause([u("defenceSolider", 1, 0, 150)], [foe("EnemyWizard", 7, 0, 500)]),
    "pause-worn": pause([u("defenceShield", 2, 4, 100, 700), u("defenceSolider", 1, 4, 150)],
                        [foe("EnemClown", 5, 4, 200), foe("EnemClown", 6, 4, 200)]),
    "pause-broke": pause([], [foe("EnemyMonk", 6, 3, 280)], coins=100),
    "tc01-stall": TC01_STALL,
}


def triples(out):
    return [(c["defenceId"], c["defenceGridPosition"]["x"], c["defenceGridPosition"]["y"]) for c in out]


def valid(state, out):
    assert isinstance(out, list)
    coins = (state.get("budget") or {}).get("remaining", 0)
    taken = {(d["position"]["x"], d["position"]["y"]) for d in state.get("defences", []) if d.get("health", 0) > 0}
    spent = 0
    for k, x, y in triples(out):
        assert k in (0, 1, 2, 3) and 0 <= x <= 10 and 0 <= y <= 5
        assert (x, y) not in taken, (k, x, y)
        taken.add((x, y))
        spent += int(COSTS[str(k)])
    assert spent <= coins, (spent, coins)


class FakeLLM:
    """Stands in for LLMClient: replies with scripted messages, in order."""
    replies = []

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, messages, timeout=None, **kw):
        msg = FakeLLM.replies.pop(0)
        return {"choices": [{"message": msg}]}


def say(obj):
    return {"role": "assistant", "content": json.dumps(obj)}


if __name__ == "__main__":
    agent.USE_LLM = False
    for name, state in CASES.items():
        t0 = time.perf_counter()
        out = agent.get_reinforcements(state)
        dt = time.perf_counter() - t0
        valid(state, out)
        assert dt < 0.5, dt
        print(f"ok  {name:17} {dt*1000:5.1f} ms  {triples(out)}")

    # TC01 opening (5000 coins): archers only, one @0 in every row, extras above the reserve.
    out = triples(agent.get_reinforcements(TC01))
    assert {k for k, *_ in out} == {2}, out
    assert {y for k, x, y in out if x == 0} == set(range(6)), out
    # Never two soldiers in one row, anywhere.
    for s in CASES.values():
        rows = [y for k, _, y in triples(agent.get_reinforcements(s)) if k == 1]
        assert len(rows) == len(set(rows)), rows
    # Pause: a dps gap gets archers (never soldiers); empty rows get their unit.
    gap = pause([u("defenceShield", 6, 0, 100, 700), u("defenceShield", 6, 4, 100, 700)],
                [foe("EnemyMonk", 7, 0, 280), foe("EnemyMonk", 7, 4, 280)], coins=2500, queued=14)
    gap["counts"]["totalEnemiesInWave"] = 18
    out = triples(agent.get_reinforcements(gap))
    assert not any(k == 1 for k, *_ in out), out
    assert sum(1 for k, *_ in out if k == 2) >= 2, out
    # (empty enemy-free rows are only filled with reserve + fill_extra coins: not asserted here)
    # TC01 stall: three regenerating clowns -> a penguin just in front of the line.
    assert any(k == 0 and y == 3 and 3 <= x <= 5 for k, x, y in triples(agent.get_reinforcements(TC01_STALL)))
    # Pirate rows: a blast on any shield never reaches a soldier, archer or penguin.
    out = triples(agent.get_reinforcements(CASES["pirate-heavy"]))
    zone = lambda x, y: {(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1), (x + 1, y - 1), (x + 1, y + 1)}
    soft = {(x, y) for k, x, y in out if k != 3}
    for k, x, y in out:
        if k == 3:
            assert not (zone(x, y) & soft), (x, y, out)
    # Shields never past x=6 (clowns would stop off the board).
    for s in CASES.values():
        assert all(not (k == 3 and x > 6) for k, x, _ in triples(agent.get_reinforcements(s)))
    # With comfortable coins every empty row ends a pause obstructed (its unit, or a shield).
    out = triples(agent.get_reinforcements(pause([u("defenceSolider", 1, 0, 150), u("defenceShield", 4, 0, 700)], [], coins=3000, queued=5)))
    assert all(any(yy == y for _, _, yy in out) for y in range(1, 6)), out
    # A lone clown in front of a healthy line: no penguin.
    lone = pause([u("defenceSolider", 1, 2, 150), u("defenceShield", 4, 2, 700)], [foe("EnemClown", 7, 2, 150)])
    assert not any(k == 0 for k, *_ in triples(agent.get_reinforcements(lone)))
    # Reserve: start 1500 -> reserve 300. A merely weak row (front low, line holds) gets nothing at 400 coins...
    full = [u(t, x, y, h) for y in (0, 1, 2, 4, 5) for t, x, h in (("defenceSolider", 1, 150), ("defenceShield", 6, 700))]
    poor = pause(full + [u("defenceSolider", 1, 3, 150), u("defenceShield", 6, 3, 100, 700)],
                 [foe("EnemClown", 9, 3, 200)], coins=400, queued=0)
    poor["budget"]["unlocked"] = 1500 + 50 * 9  # 10 in wave, 0 queued, 1 alive -> 9 kills
    assert agent.analyse(poor)["reserve"] == 300
    assert triples(agent.get_reinforcements(poor)) == [], agent.get_reinforcements(poor)
    # ...a critical row (3 regenerating clowns on a worn shield) may spend below it.
    crit = pause(full + [u("defenceSolider", 1, 3, 150), u("defenceShield", 6, 3, 160, 700)],
                 [foe("EnemClown", 9, 3, 117), foe("EnemClown", 9, 3, 80), foe("EnemClown", 9, 3, 192)], coins=400, queued=0)
    crit["budget"]["unlocked"] = 1500 + 50 * 7
    assert any(k == 0 and y == 3 for k, _, y in triples(agent.get_reinforcements(crit)))
    # ...but a breach is still answered below the floor.
    breach = pause([], [foe("EnemyMonk", 6, 3, 280)], coins=400, queued=0)
    breach["budget"]["unlocked"] = 1500
    assert any(y == 3 for _, _, y in triples(agent.get_reinforcements(breach)))
    for s in (CASES["pause-empty-row"], CASES["pause-wizard"]):
        assert triples(agent.get_reinforcements(s)), "must react"

    # Opening: at least 3 archers whenever coins allow (3000 coins, all six rows threatened).
    assert sum(1 for k, *_ in triples(agent.get_reinforcements(CASES["all-rows-unknown"])) if k == 2) >= 3
    # Wizard parked in front of our x=6 shield: arm the trap, a penguin right BEHIND the shield (x=5).
    full = [u(t, x, y, h) for y in range(6) for t, x, h in (("defenceSolider", 1, 150), ("defenceShield", 6, 700))]
    wiz = pause([d for d in full if not (d["position"]["y"] == 0 and d["type"] == "defenceShield")]
                + [u("defenceShield", 6, 0, 300, 700)], [foe("EnemyWizard", 10, 0, 500), foe("EnemClown", 10, 0, 200)])
    out = triples(agent.get_reinforcements(wiz))
    assert (0, 5, 0) in out, out
    # End of wave: nothing queued, rows clearing in time -> spend nothing.
    calm = pause(full + [u("defenceArcher", 0, y, 100) for y in range(3)],
                 [foe("EnemClown", 9, 2, 120), foe("EnemyMonk", 7, 4, 100)], coins=3000, queued=0)
    assert triples(agent.get_reinforcements(calm)) == []

    # Mid-game row short on dps (already has its soldier): buy an archer, never a second soldier.
    short = pause(full + [u("defenceArcher", 0, 5, 100)], [foe("EnemyMonk", 7, 3, 280)], coins=3000, queued=6)
    short["defences"] = [d for d in short["defences"] if not (d["position"] == {"x": 6, "y": 3})] + [u("defenceShield", 6, 3, 100, 700)]
    out = triples(agent.get_reinforcements(short))
    assert any(k == 2 for k, *_ in out) and not any(k == 1 for k, *_ in out), out

    # A blocked row whose dps clears the enemy in time needs no shield (spare coins go to dps instead).
    exposed = pause([u("defenceArcher", 0, y, 100) for y in range(6)] + [u("defenceSolider", 1, 2, 150)],
                    [foe("EnemClown", 9, 2, 200)], coins=3000, queued=8)
    exposed["counts"]["totalEnemiesInWave"] = 12
    out = triples(agent.get_reinforcements(exposed))
    assert not any(k == 3 for k, *_ in out), out

    # Row 4 lost its shield (archer + soldier left). A lone clown that our dps kills in time: no shield...
    lost = pause([d for d in full if d["position"] != {"x": 6, "y": 4}] + [u("defenceArcher", 0, 4, 100)],
                 [foe("EnemClown", 9, 4, 200)], coins=2500, queued=6)
    lost["counts"]["totalEnemiesInWave"] = 12
    assert not any(k == 3 and y == 4 for k, _, y in triples(agent.get_reinforcements(lost)))
    # ...two wizards there while the archers are busy on row 0 and the coins are below the reserve: it falls
    # -> a blocking shield (archers can't be afforded above the reserve).
    lost["enemies"] = [foe("EnemyMonk", 3, 0, 280), foe("EnemyWizard", 8, 4, 500), foe("EnemyWizard", 10, 4, 500)]
    lost["budget"]["remaining"] = 500
    out = triples(agent.get_reinforcements(lost))
    assert any(y == 4 and k in (0, 3) for k, _, y in out), out

    # A single monk walking into an archer-only row with 6 archers on it: NEVER a penguin, and no shield.
    monk = pause([u("defenceArcher", 0, y, 100) for y in range(6)], [foe("EnemyMonk", 6, 2, 280)], coins=2500, queued=8)
    monk["counts"]["totalEnemiesInWave"] = 12
    out = triples(agent.get_reinforcements(monk))
    assert not any(k == 0 for k, *_ in out), out
    assert not any(k == 3 and y == 2 for k, x, y in out), out  # 6 archers on that row kill it first: no shield
    # Same board, the LLM tries a penguin anyway: dropped.
    agent.USE_LLM, agent.LLMClient = True, FakeLLM
    FakeLLM.replies = [say({"placements": [[0, 5, 2]], "reason": "panic"})]
    assert not any(k == 0 for k, *_ in triples(agent.get_reinforcements(monk)))
    agent.USE_LLM = False
    # Calm board, every row has a unit: no shields wasted.
    mid = pause([u("defenceArcher", 0, y, 100) for y in range(6)], [], coins=1000, queued=8)
    mid["counts"]["totalEnemiesInWave"] = 12
    mid["budget"]["unlocked"] = 5000 + 50 * 4
    assert not any(k == 3 for k, *_ in triples(agent.get_reinforcements(mid)))

    # TC02 loss: wizard row critical, only 150 coins -> a shield, not nothing.
    tc02 = pause([u("defenceShield", 6, 4, 130, 700), u("defenceArcher", 0, 2, 100), u("defenceArcher", 0, 3, 100)],
                 [foe("EnemyWizard", 10, 4, 437), foe("EnemClown", 9, 1, 25)], coins=150, queued=3)
    tc02["counts"]["totalEnemiesInWave"] = 11
    tc02["budget"]["unlocked"] = 1700 + 50 * 6
    out = triples(agent.get_reinforcements(tc02))
    assert any(k == 3 and y == 4 for k, _, y in out), out

    # Public TC04 opening (2100 coins, wizards+pirates in rows 1/3/5): 3 archers, no penguins.
    tc04 = opening([{"enemyId": 0, "count": 2, "allowedRows": [0, 2, 4]}, {"enemyId": 1, "count": 7, "allowedRows": [0, 2, 4]},
                    {"enemyId": 2, "count": 4, "allowedRows": [1, 3, 5]}, {"enemyId": 3, "count": 2, "allowedRows": [1, 3, 5]}],
                   coins=2100, first=("EnemyMonk", 10, 0, 280))
    tc04["gameSettings"]["reinforcementIntervalSeconds"] = 10
    out = triples(agent.get_reinforcements(tc04))
    assert not any(k == 0 for k, *_ in out) and sum(1 for k, *_ in out if k == 2) >= 3, out

    # Shield x follows the enemy: a monk already at x=4 in an empty top row -> shield in front of it, not at 6.
    deep = pause([u("defenceArcher", 0, y, 100) for y in range(3)] + [u("defenceShield", 6, y, 700) for y in (3, 5)],
                 [foe("EnemyMonk", 4, 4, 280)], coins=2000, queued=5)
    out = triples(agent.get_reinforcements(deep))
    assert any(k == 3 and y == 4 and x < 4 for k, x, y in out) and not any(y == 4 and x >= 4 for _, x, y in out), out
    # ...and an LLM card behind the enemy is moved in front of it.
    agent.USE_LLM, agent.LLMClient = True, FakeLLM
    FakeLLM.replies = [say({"placements": [[3, 6, 4]], "reason": "x"})]
    out = triples(agent.get_reinforcements(deep))
    assert any(k == 3 and y == 4 and x < 4 for k, x, y in out) and not any(y == 4 and x >= 4 for _, x, y in out), out
    agent.USE_LLM = False
    # Public TC03 (2000 coins): 6 archers don't fit above half the reserve -> 5 archers + a shield in the
    # densest row (row 5: 2 wizards + 2 pirates expected). No soldiers.
    tc03 = opening([{"enemyId": 0, "count": 2, "allowedRows": [2, 3]}, {"enemyId": 1, "count": 2, "allowedRows": [0, 1]},
                    {"enemyId": 2, "count": 4, "allowedRows": [0, 5]}, {"enemyId": 3, "count": 4, "allowedRows": [4, 5]}], coins=2000)
    out = triples(agent.get_reinforcements(tc03))
    spent = sum({0: 250, 1: 200, 2: 300, 3: 150}[k] for k, *_ in out)
    assert 2000 - spent >= agent.analyse(tc03)["reserve"], (spent, out)          # the reserve holds
    assert (3, 6, 5) in out and sum(1 for k, *_ in out if k == 2) >= 3, out       # densest row shielded first
    # Density: 3 clowns bunched within 2 tiles (600 hp) -> penguin right in front of them, whatever the dps.
    bunch = pause([u("defenceArcher", 0, y, 100) for y in range(6)] + [u("defenceShield", 6, 4, 700)],
                  [foe("EnemClown", 9, 4, 200), foe("EnemClown", 9, 4, 200), foe("EnemClown", 10, 4, 200)], coins=2000, queued=6)
    assert (0, 8, 4) in triples(agent.get_reinforcements(bunch))
    # A lone wizard the archers are killing: never a penguin (public TC03 lesson).
    lone_w = pause([u("defenceArcher", 0, y, 100) for y in range(6)], [foe("EnemyWizard", 10, 3, 500)], coins=2000, queued=0)
    assert not any(k == 0 for k, *_ in triples(agent.get_reinforcements(lone_w)))
    # Two wizards our dps can't clear at a worn shield: the trap penguin behind it.
    dense = pause([u("defenceArcher", 0, 0, 100), u("defenceShield", 6, 4, 200, 700)],
                  [foe("EnemClown", 1, 0, 200)] + [foe("EnemyWizard", 9, 4, 500), foe("EnemyWizard", 10, 4, 500)], coins=2000, queued=6)
    out = triples(agent.get_reinforcements(dense))
    assert (0, 5, 4) in out, out  # trap: right behind the worn shield, fires when it falls
    # Public TC04 pause 2: a wizard breaching row 1 (archer only), 700 coins -> the row-1 shield is paid first.
    tc04p = pause([u("defenceArcher", 0, y, 100) for y in (0, 1, 3, 5)] + [u("defenceShield", 6, 4, 700), u("defenceShield", 6, 2, 700)],
                  [foe("EnemyWizard", 5, 1, 462)], coins=700, queued=12)
    tc04p["counts"]["totalEnemiesInWave"] = 15
    tc04p["budget"]["unlocked"] = 2100 + 50 * 2
    for llm in (False, True):
        agent.USE_LLM, agent.LLMClient = llm, FakeLLM
        FakeLLM.replies = [say({"option": "needed", "reason": "x"})]
        out = triples(agent.get_reinforcements(tc04p))
        assert out and out[0][0] == 3 and out[0][2] == 1, (llm, out)
    agent.USE_LLM = False
    # ...but two spread-out clowns in a holding row: no penguin.
    spread = pause([u("defenceArcher", 0, y, 100) for y in range(6)] + [u("defenceShield", 6, 4, 700)],
                   [foe("EnemClown", 7, 4, 200), foe("EnemClown", 10, 4, 200)], coins=2000, queued=6)
    assert not any(k == 0 for k, *_ in triples(agent.get_reinforcements(spread)))

    # Reserve leaks closed: an LLM opening that spends everything is trimmed back to the reserve...
    agent.USE_LLM, agent.LLMClient = True, FakeLLM
    FakeLLM.replies = [say({"placements": [[2, x, y] for y in range(6) for x in (0, 1)], "reason": "all in"})]
    out = triples(agent.get_reinforcements(tc03))
    assert sum(1 for k, *_ in out if k == 2) <= (2000 - agent.analyse(tc03)["reserve"]) // 300, out  # archers within the reserve
    assert {y for *_, y in out} == set(range(6)), out                                               # rows left empty get shields
    agent.USE_LLM = False
    # ...and an exposed archer with an enemy coming but the row holding: no shield below the reserve.
    hold = pause([u("defenceArcher", 0, y, 100) for y in range(6)], [foe("EnemClown", 9, 2, 200)], coins=700, queued=6)
    hold["counts"]["totalEnemiesInWave"] = 12
    hold["budget"]["unlocked"] = 2100 + 50 * 6   # start 2100 -> reserve 600
    assert triples(agent.get_reinforcements(hold)) == [], agent.get_reinforcements(hold)

    # DPS before shields: a top row with 1 archer losing to a clown. Spare coins above the reserve ->
    # one more archer fixes it (model: t_clear 5.4 s < line 5.9 s), no shield...
    top = pause([u("defenceArcher", 0, y, 100) for y in (0, 1, 2, 3, 4, 5)], [foe("EnemClown", 7, 5, 200)], coins=3000, queued=0)
    top["defences"] = [d for d in top["defences"] if d["position"]["y"] == 5]
    top["defences"] += [u("defenceShield", 6, y, 700) for y in range(5)]
    out = triples(agent.get_reinforcements(top))
    assert not any(k == 3 for k, *_ in out) and sum(1 for k, *_ in out if k == 2) >= 1, out
    # ...no spare coins (below the reserve) and the clown already close (urgent): a blocking shield in row 5.
    top["enemies"] = [foe("EnemClown", 5, 5, 200)]
    top["budget"]["remaining"] = 400
    top["budget"]["unlocked"] = 2000 + 50 * 9
    out = triples(agent.get_reinforcements(top))
    assert any(k == 3 and y == 5 for k, _, y in out), out
    # Archer focus + queue: 4 archers clear a top-row clown easily when nothing is queued...
    focus = pause([u("defenceArcher", 0, y, 100) for y in (0, 1, 2, 5)] + [u("defenceShield", 6, y, 700) for y in (3, 4)],
                  [foe("EnemClown", 7, 5, 200)], coins=3000, queued=0)
    assert agent.assess(agent.analyse(focus))["rows"][5]["status"] == "ok"
    # ...but with 10 enemies still queued and spawning every 2 s, lower rows keep stealing the archers: row 5 falls.
    focus["counts"].update(enemiesQueued=10, totalEnemiesInWave=15)
    focus["elapsedSeconds"] = 10
    r5 = agent.assess(agent.analyse(focus))["rows"][5]
    assert r5["status"] == "critical" and r5["archer_share"] < 0.5, r5

    # Public "12 wizards + 6 clowns, rows 2-3, 1400 coins" case.
    wz = opening([{"enemyId": 2, "count": 12, "allowedRows": [2, 3]}, {"enemyId": 0, "count": 6, "allowedRows": [2, 3]}],
                 coins=1400, first=("EnemyWizard", 10, 2, 500))
    out = triples(agent.get_reinforcements(wz))
    spent = sum({0: 250, 1: 200, 2: 300, 3: 150}[k] for k, *_ in out)
    assert (3, 6, 2) in out and (3, 6, 3) in out, out                 # shields keep the beams off the archers
    assert (0, 5, 2) in out and (0, 5, 3) in out, out                 # wizard-heavy rows: the trap penguins
    assert 1400 - spent >= agent.analyse(wz)["reserve"], out
    # Two wizards parked at a HEALTHY x=6 shield (~9 s of wall): not urgent -> no penguin yet, no 2nd shield.
    base_wz = [u("defenceShield", 6, 2, 700), u("defenceShield", 6, 3, 700), u("defenceArcher", 0, 0, 100), u("defenceArcher", 0, 1, 100)]
    park = pause(base_wz, [foe("EnemyWizard", 10, 2, 500), foe("EnemyWizard", 10, 2, 480)], coins=500, queued=14)
    park["counts"]["totalEnemiesInWave"] = 18
    park["budget"]["unlocked"] = 1400 + 50 * 2
    out = triples(agent.get_reinforcements(park))
    assert not any(k == 0 for k, *_ in out), out
    assert not any(k == 3 and y == 2 for k, _, y in out), out
    # Rows 1/4/5 are empty but coins are near the reserve: no fillers bought there.
    assert not any(y in (1, 4, 5) for _, _, y in out), out
    # A critical row whose shield is healthy, 100 coins: no second shield stacked behind it.
    stack = pause(base_wz, [foe("EnemyWizard", 10, 3, 443)], coins=100, queued=13)
    stack["counts"]["totalEnemiesInWave"] = 18
    assert not any(k == 3 for k, *_ in triples(agent.get_reinforcements(stack)))

    # Public "12 wizards" run 2, replayed. Pause 2: one wizard per row beaming a still-solid x=6 shield
    # (~10 s of wall left): nothing urgent -> keep the coins (the run bought an archer here and went broke).
    wz_defs = lambda h2, h3: [u("defenceShield", 6, 2, h2, 700), u("defenceShield", 6, 3, h3, 700),
                              u("defenceArcher", 0, 0, 100), u("defenceArcher", 0, 1, 100)]
    p2 = pause(wz_defs(510, 510), [foe("EnemyWizard", 10, 2, 481), foe("EnemyWizard", 10, 3, 372)], coins=500, queued=16)
    p2["counts"]["totalEnemiesInWave"] = 18; p2["elapsedSeconds"] = 5.5; p2["budget"]["unlocked"] = 1400
    assert triples(agent.get_reinforcements(p2)) == [], agent.get_reinforcements(p2)  # nothing urgent: keep the coins
    # Pause 3: two healthy wizards parked in row 2 -> the trap penguin behind the shield, even near the reserve.
    p3 = pause(wz_defs(300, 260), [foe("EnemyWizard", 10, 2, 480), foe("EnemyWizard", 10, 2, 469), foe("EnemyWizard", 10, 3, 450)],
               coins=550, queued=14)
    p3["counts"]["totalEnemiesInWave"] = 18; p3["elapsedSeconds"] = 10.7; p3["budget"]["unlocked"] = 1450
    out = triples(agent.get_reinforcements(p3))
    assert (0, 5, 2) in out, out
    # Pause 4: row 3's shield is about to fall (20 hp) under a wizard -> it gets replaced now.
    p4 = pause(wz_defs(500, 20) + [u("defenceArcher", 0, 2, 100)], [foe("EnemyWizard", 10, 3, 369)], coins=300, queued=13)
    p4["counts"]["totalEnemiesInWave"] = 18; p4["elapsedSeconds"] = 16; p4["budget"]["unlocked"] = 1600
    out = triples(agent.get_reinforcements(p4))
    assert any(y == 3 and k in (0, 2, 3) for k, _, y in out), out

    # TC11: an armed trap whose shield is about to fall is left alone (no shield stacked over the penguin).
    armed = pause([u("defenceShield", 6, 3, 60, 700), u("defenceShield", 6, 2, 700), u("defencePenguin", 5, 3, 200),
                   u("defencePenguin", 5, 2, 200), u("defenceArcher", 0, 0, 100)],
                  [foe("EnemyWizard", 10, 3, 500), foe("EnemyWizard", 10, 3, 500), foe("EnemClown", 9, 3, 200)], coins=400, queued=10)
    armed["counts"]["totalEnemiesInWave"] = 18
    assert not any(y == 3 for _, _, y in triples(agent.get_reinforcements(armed)))
    # ...and after it fired (row empty, a new wizard there): rebuild the trap.
    fired = pause([u("defenceShield", 6, 2, 700), u("defencePenguin", 5, 2, 200), u("defenceArcher", 0, 0, 100)],
                  [foe("EnemyWizard", 10, 3, 500), foe("EnemyWizard", 10, 3, 480)], coins=500, queued=8)
    fired["counts"]["totalEnemiesInWave"] = 18
    out = triples(agent.get_reinforcements(fired))
    assert any(k == 0 and y == 3 for k, _, y in out) and any(k == 3 and y == 3 for k, _, y in out), out

    # --- LLM path, with a fake client ---------------------------------------
    agent.USE_LLM, agent.LLMClient = True, FakeLLM
    FakeLLM.replies = [say({"option": "lean", "reason": "test"})]
    out = triples(agent.get_reinforcements(TC01))
    assert sorted(k for k, *_ in out) == [2] * 6, out  # lean: an archer in every row
    tool_call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "lane_report", "arguments": '{"row": 3}'}}]}
    FakeLLM.replies = [tool_call, say({"placements": [[0, 3, 3], [2, 0, 3]], "reason": "custom"})]
    out = triples(agent.get_reinforcements(TC01_STALL))
    assert out[-2:] == [(0, 3, 3), (2, 0, 3)] and len({y for *_, y in out}) == 6, out  # fillers for empty rows first
    FakeLLM.replies = [say({"option": "hold", "reason": "save coins"})]
    assert not any(k == 3 for k, *_ in triples(agent.get_reinforcements(exposed)))
    # Public TC02 pause 5: wizard breaching row 3, 150 coins, the LLM spends them on row 1 -> row 3 still gets its shield.
    tc02b = pause([u("defenceArcher", 0, 2, 100), u("defenceSolider", 1, 2, 150), u("defenceShield", 6, 0, 700),
                   u("defenceShield", 6, 4, 400, 700), u("defenceShield", 6, 5, 700), u("defenceSolider", 1, 4, 150)],
                  [foe("EnemyWizard", 5, 3, 389), foe("EnemClown", 8, 2, 12)], coins=150, queued=0)
    tc02b["counts"]["totalEnemiesInWave"] = 11
    tc02b["budget"]["unlocked"] = 1700 + 50 * 9
    FakeLLM.replies = [say({"placements": [[3, 6, 1]], "reason": "penguin in front of wizard"})]
    out = triples(agent.get_reinforcements(tc02b))
    assert out and out[0][2] == 3, out
    FakeLLM.replies = [{"role": "assistant", "content": "I think... maybe"}]
    assert any(k == 0 and y == 3 and 3 <= x <= 5 for k, x, y in triples(agent.get_reinforcements(TC01_STALL)))       # garbage -> rules
    FakeLLM.replies = [say({"placements": [[1, 5, 0]]})]                      # LLM skips row 3's breach
    out = triples(agent.get_reinforcements(CASES["pause-empty-row"]))
    assert any(y == 3 for _, _, y in out), out                                # safety net adds it
    print("all passed")
