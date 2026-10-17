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

    # TC01 opening: every row gets soldier@1 + a shield at x 4..6, archers sit at x=0 behind them.
    out = triples(agent.get_reinforcements(TC01))
    assert all((1, 1, y) in out and any(k == 3 and 4 <= x <= 6 and yy == y for k, x, yy in out) for y in range(6)), out
    assert sum(1 for k, x, _ in out if k == 2 and x == 0) >= 4, out
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
    # Every row ends a pause obstructed: an empty, quiet row gets a blocker shield.
    out = triples(agent.get_reinforcements(pause([u("defenceSolider", 1, 0, 150), u("defenceShield", 4, 0, 700)], [], queued=5)))
    assert all(any(yy == y and k == 3 for k, _, yy in out) for y in range(1, 6)), out
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

    # --- LLM path, with a fake client ---------------------------------------
    agent.USE_LLM, agent.LLMClient = True, FakeLLM
    FakeLLM.replies = [say({"option": "lean", "reason": "test"})]
    out = triples(agent.get_reinforcements(TC01))
    assert not any(k == 2 for k, *_ in out) and len(out) == 12, out          # lean = line only
    tool_call = {"role": "assistant", "content": None, "tool_calls": [
        {"id": "1", "type": "function", "function": {"name": "lane_report", "arguments": '{"row": 3}'}}]}
    FakeLLM.replies = [tool_call, say({"placements": [[0, 3, 3], [2, 0, 3]], "reason": "custom"})]
    out = triples(agent.get_reinforcements(TC01_STALL))
    assert out[:2] == [(0, 3, 3), (2, 0, 3)] and all(k == 3 for k, *_ in out[2:]), out  # + open-row blockers
    FakeLLM.replies = [{"role": "assistant", "content": "I think... maybe"}]
    assert any(k == 0 and y == 3 and 3 <= x <= 5 for k, x, y in triples(agent.get_reinforcements(TC01_STALL)))       # garbage -> rules
    FakeLLM.replies = [say({"placements": [[1, 5, 0]]})]                      # LLM skips row 3's breach
    out = triples(agent.get_reinforcements(CASES["pause-empty-row"]))
    assert any(y == 3 for _, _, y in out), out                                # safety net adds it
    print("all passed")
