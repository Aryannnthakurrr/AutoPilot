"""Offline harness: python3 test_agent.py  (asserts valid, affordable, fast, never crashes)."""
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


CASES = {
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
}


def valid(state, out):
    assert isinstance(out, list)
    coins = (state.get("budget") or {}).get("remaining", 0)
    taken = {(d["position"]["x"], d["position"]["y"]) for d in state.get("defences", []) if d.get("health", 0) > 0}
    spent = 0
    for c in out:
        x, y = c["defenceGridPosition"]["x"], c["defenceGridPosition"]["y"]
        assert c["defenceId"] in (0, 1, 2, 3) and 0 <= x <= 10 and 0 <= y <= 5, c
        assert (x, y) not in taken, c
        taken.add((x, y))
        spent += int(COSTS[str(c["defenceId"])])
    assert spent <= coins, (spent, coins)


if __name__ == "__main__":
    for name, state in CASES.items():
        t0 = time.perf_counter()
        out = agent.get_reinforcements(state)
        dt = time.perf_counter() - t0
        valid(state, out)
        assert dt < 0.5, dt
        print(f"ok  {name:18} {dt*1000:5.1f} ms  {[(c['defenceId'], c['defenceGridPosition']['x'], c['defenceGridPosition']['y']) for c in out]}")
    out = agent.get_reinforcements(CASES["pause-empty-row"])
    assert any(c["defenceGridPosition"]["y"] == 3 for c in out), "empty row must get a blocker"
    out = agent.get_reinforcements(CASES["pause-wizard"])
    assert any(c["defenceId"] == 0 for c in out), "live wizard needs a penguin"
    print("all passed")
