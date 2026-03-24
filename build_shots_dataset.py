import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

EVENTS_DIR = Path("data/events")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOAL_X = 120.0
GOAL_LEFT_POST = np.array([120.0, 36.0])
GOAL_RIGHT_POST = np.array([120.0, 44.0])
GOAL_CENTER = np.array([120.0, 40.0])

def euclid(a, b):
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))

def dist_to_goal(x, y):
    return euclid([x, y], GOAL_CENTER)

def angle_to_goal(x, y):
    p = np.array([x, y], dtype=float)
    v1 = GOAL_LEFT_POST - p
    v2 = GOAL_RIGHT_POST - p
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return float(math.atan2(abs(cross), dot))

def point_in_triangle(pt, a, b, c):
    pt = np.array(pt, dtype=float)
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    v0 = c - a
    v1 = b - a
    v2 = pt - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False

    inv = 1.0 / denom
    u = (dot11 * dot02 - dot01 * dot12) * inv
    v = (dot00 * dot12 - dot01 * dot02) * inv

    return (u >= 0) and (v >= 0) and (u + v <= 1)

def extract_freeze_frame_features(shot_x, shot_y, freeze_frame):
    if not freeze_frame or freeze_frame is None:
        return {
            "keeper_distance": np.nan,
            "nearest_defender_distance": np.nan,
            "blocker_count": 0,
            "defenders_count": 0
        }

    shot_pt = [shot_x, shot_y]

    keeper_locs = []
    defender_locs = []

    for p in freeze_frame:
        loc = p.get("location")
        if not loc or len(loc) < 2:
            continue

        x, y = float(loc[0]), float(loc[1])
        is_teammate = p.get("teammate", None)

        pos_name = (p.get("position", {}) or {}).get("name", "")
        is_keeper = (pos_name == "Goalkeeper") or bool(p.get("keeper", False))

        if is_keeper:
            keeper_locs.append([x, y])
            continue

        if is_teammate is False:
            defender_locs.append([x, y])

    keeper_distance = np.nan
    if keeper_locs:
        keeper_distance = min(euclid(shot_pt, k) for k in keeper_locs)

    nearest_defender_distance = np.nan
    if defender_locs:
        nearest_defender_distance = min(euclid(shot_pt, d) for d in defender_locs)

    blocker_count = 0
    tri_a = shot_pt
    tri_b = GOAL_LEFT_POST
    tri_c = GOAL_RIGHT_POST
    for d in defender_locs:
        if d[0] < shot_x or d[0] > GOAL_X:
            continue
        if point_in_triangle(d, tri_a, tri_b, tri_c):
            blocker_count += 1

    return {
        "keeper_distance": keeper_distance,
        "nearest_defender_distance": nearest_defender_distance,
        "blocker_count": int(blocker_count),
        "defenders_count": int(len(defender_locs))
    }

rows = []
event_files = sorted(EVENTS_DIR.glob("*.json"))
if not event_files:
    raise RuntimeError("data/events içinde json yok.")

for fp in event_files:
    match_id = int(fp.stem)
    events = json.loads(fp.read_text(encoding="utf-8"))

    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue

        loc = ev.get("location")
        shot = ev.get("shot", {}) or {}
        outcome = (shot.get("outcome", {}) or {}).get("name")
        if not loc or outcome is None:
            continue

        shot_type = (shot.get("type", {}) or {}).get("name")
        if shot_type == "Penalty":
            continue

        shot_x, shot_y = float(loc[0]), float(loc[1])
        is_goal = 1 if outcome == "Goal" else 0

        freeze_frame = shot.get("freeze_frame", None)
        ff_features = extract_freeze_frame_features(shot_x, shot_y, freeze_frame)

        rows.append({
            "match_id": match_id,
            "event_id": ev.get("id"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),

            "shot_x": shot_x,
            "shot_y": shot_y,
            "distance_to_goal": dist_to_goal(shot_x, shot_y),
            "shot_angle_rad": angle_to_goal(shot_x, shot_y),

            "is_goal": int(is_goal),
            "outcome": outcome,

            "shot_type": shot_type,
            "body_part": (shot.get("body_part", {}) or {}).get("name"),
            "technique": (shot.get("technique", {}) or {}).get("name"),
            "play_pattern": (ev.get("play_pattern", {}) or {}).get("name"),
            "under_pressure": bool(ev.get("under_pressure", False)),

            **ff_features,

            "freeze_frame_json": json.dumps(freeze_frame, ensure_ascii=False),
        })

df = pd.DataFrame(rows)

df = df[
    (df["shot_x"] >= 0) & (df["shot_x"] <= 120) &
    (df["shot_y"] >= 0) & (df["shot_y"] <= 80)
]

df = df.dropna(subset=[
    "distance_to_goal",
    "shot_angle_rad"
])

cat_cols = ["shot_type", "body_part", "technique", "play_pattern"]
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = df[col].replace("None", "unknown")
    df[col] = df[col].fillna("unknown")

df["under_pressure"] = df["under_pressure"].astype(bool)

df["keeper_distance"] = df["keeper_distance"].fillna(-1)
df["nearest_defender_distance"] = df["nearest_defender_distance"].fillna(-1)

df = df.dropna(subset=["shot_x", "shot_y", "distance_to_goal", "shot_angle_rad", "is_goal"])
df["is_goal"] = df["is_goal"].astype(int)

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(df, groups=df["match_id"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)

df.to_parquet(OUT_DIR / "shots_features.parquet", index=False)
df.to_csv(OUT_DIR / "shots_features.csv", index=False)
train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
test_df.to_parquet(OUT_DIR / "test.parquet", index=False)

report = []
report.append(f"matches_unique={df.match_id.nunique()}")
report.append(f"shots_total={len(df)}")
report.append(f"goals_total={int(df.is_goal.sum())}")
report.append(f"goal_rate={float(df.is_goal.mean()):.4f}")
report.append(f"freeze_frame_null_ratio={(df.freeze_frame_json=='null').mean():.4f}")
report.append(f"train_shots={len(train_df)}")
report.append(f"test_shots={len(test_df)}")
(OUT_DIR / "data_report.txt").write_text("\n".join(report), encoding="utf-8")

print("Dataset hazır")
print("\n".join(report))
print("Çıktı:", OUT_DIR.resolve())
