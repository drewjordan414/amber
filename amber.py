"""Amber v0 — cannabis grow advisor: read sensors + take a photo every hour, print recommendations.

Run on the Pi:   python amber.py
Change stage:    python amber.py --stage flower     (labels every photo from now on)
Log an action:   python amber.py --event "fed 5ml/gal bloom"
Run anywhere:    python amber.py --fake --once
Self-check:      python amber.py --test
"""
import argparse, csv, math, time
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).parent / "data"

PLANT_TYPE = "photoperiod"  # or "auto" (autoflowers flower on age, not light schedule)

# Cannabis stages, in order. preflower = the 2-3 week "stretch" after the flip to 12/12, when pistils
# appear at the nodes; flower = bulking; ripening = last 2-3 weeks, trichomes turning cloudy/amber.
# Targets: (lo, hi), common cannabis VPD-chart ranges; tune for strain and medium.
TARGETS = {
    "seedling":  {"temp_c": (20, 25), "rh": (65, 75), "vpd": (0.4, 0.8)},
    "veg":       {"temp_c": (22, 28), "rh": (55, 70), "vpd": (0.8, 1.1)},
    "preflower": {"temp_c": (22, 27), "rh": (50, 60), "vpd": (1.0, 1.3)},
    "flower":    {"temp_c": (20, 26), "rh": (45, 50), "vpd": (1.2, 1.5)},
    "ripening":  {"temp_c": (18, 24), "rh": (35, 45), "vpd": (1.3, 1.6)},
}
FLOWERING = ("preflower", "flower", "ripening")
LIGHT_HOURS = {"photoperiod": {"seedling": 18, "veg": 18, **dict.fromkeys(FLOWERING, 12)},
               "auto": dict.fromkeys(TARGETS, 18)}[PLANT_TYPE]
FEED = {
    "seedling":  "Plain pH'd water (soil 6.0-7.0, coco/hydro 5.5-6.5). No nutrients until 2-3 sets of true leaves.",
    "veg":       "Grow formula (high N), ramp 1/2 -> full strength. Add Cal-Mag if in coco or under LEDs.",
    "preflower": "Transition feed: keep N up through the stretch (plants can double in height), start adding bloom P/K.",
    "flower":    "Bloom formula (high P/K), cut N. Peak feed strength during the bulking weeks.",
    "ripening":  "Taper feed; plain water the last 1-2 weeks if you flush. Check trichomes: harvest at mostly cloudy, ~10-20% amber.",
}
# Calibration knobs. Read the soil sensor in bone-dry and just-watered soil and put the values here.
SOIL_DRY, SOIL_WET = 350, 1000
WATER_BELOW = 0.35  # water when moisture drops below 35% of the dry->wet range
LIGHTS_ON_LUX = 50  # below this the tent is dark; skip the photo (useless for training)
LEAK_LUX = 1        # anything above this during lights-off in flower is a light leak


def vpd_kpa(temp_c, rh):
    """Leaf VPD estimate (leaf assumed 2C cooler than air)."""
    svp = lambda t: 0.6108 * math.exp(17.27 * t / (t + 237.3))
    return svp(temp_c - 2) - svp(temp_c) * rh / 100


def recommend(stage, r, lit_hours=None):
    t, out = TARGETS[stage], []
    for key, label in (("temp_c", "Temperature"), ("rh", "Humidity")):
        lo, hi = t[key]
        if r[key] < lo: out.append(f"{label} low ({r[key]:.1f}, target {lo}-{hi}).")
        if r[key] > hi: out.append(f"{label} high ({r[key]:.1f}, target {lo}-{hi}).")
    if stage in ("flower", "ripening") and r["rh"] > t["rh"][1]:
        out.append("Bud rot (botrytis) risk in dense colas: dehumidify, increase airflow, defoliate the interior.")
    if PLANT_TYPE == "photoperiod" and stage in FLOWERING and LEAK_LUX < r["lux"] < LIGHTS_ON_LUX:
        out.append(f"Light leak during dark period ({r['lux']:.1f} lux): interrupted dark can cause hermaphrodites (nanners). Seal the tent.")
    want = LIGHT_HOURS[stage]
    if lit_hours is not None and abs(lit_hours - want) >= 2:
        out.append(f"Lights on ~{lit_hours:.0f}h/day, expected {want}h for {stage}. Check the timer.")
    v, (lo, hi) = vpd_kpa(r["temp_c"], r["rh"]), t["vpd"]
    if not lo <= v <= hi:
        out.append(f"VPD {v:.2f} kPa outside {lo}-{hi}: {'raise RH / lower temp' if v > hi else 'lower RH / raise temp'}.")
    moisture = (r["soil_raw"] - SOIL_DRY) / (SOIL_WET - SOIL_DRY)
    if moisture < WATER_BELOW:
        out.append(f"Soil at {moisture:.0%}: water now. Feed: {FEED[stage]}")
    return out or ["All in range. No action."]


def read_sensors():
    # ponytail: assumes Adafruit STEMMA soil (seesaw), SHT4x and TSL2591 on I2C; swap imports for your actual boards.
    import board, adafruit_sht4x, adafruit_tsl2591
    from adafruit_seesaw.seesaw import Seesaw
    i2c = board.I2C()
    temp_c, rh = adafruit_sht4x.SHT4x(i2c).measurements
    return {"temp_c": temp_c, "rh": rh,
            "soil_raw": Seesaw(i2c, addr=0x36).moisture_read(),
            "lux": adafruit_tsl2591.TSL2591(i2c).lux}


def snapshot(ts, stage):
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.start(); time.sleep(2)  # let auto-exposure settle
    path = DATA / "photos" / stage / f"{ts}.jpg"  # folder = label, which is what train.py reads
    path.parent.mkdir(parents=True, exist_ok=True)
    cam.capture_file(str(path)); cam.close()
    return path


def now():
    return datetime.now().strftime("%Y%m%d-%H%M")


def append_csv(name, header, row):
    path = DATA / name
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(header)
        w.writerow(row)


def lit_hours():
    """Hours of light in the last 24h, from the log. None until there's a full day of readings."""
    log = DATA / "log.csv"
    if not log.exists(): return None
    with log.open() as f:
        rows = [r for r in csv.DictReader(f)
                if (datetime.now() - datetime.strptime(r["ts"], "%Y%m%d-%H%M")).total_seconds() < 86400]
    if len(rows) < 20: return None  # ponytail: assumes hourly readings; resolution is +/- one interval
    return 24 * sum(float(r["lux"]) >= LIGHTS_ON_LUX for r in rows) / len(rows)


def current_stage():
    f = DATA / "stage.txt"
    return f.read_text().strip() if f.exists() else "seedling"


def tick(fake):
    ts, stage = now(), current_stage()
    r = {"temp_c": 29.0, "rh": 35.0, "soil_raw": 400, "lux": 20000} if fake else read_sensors()
    photo = "" if fake or r["lux"] < LIGHTS_ON_LUX else snapshot(ts, stage)
    append_csv("log.csv", ["ts", "stage", *r, "photo"], [ts, stage, *r.values(), photo])
    print(ts, stage, r)
    for line in recommend(stage, r, lit_hours()): print("  -", line)


def test():
    assert 1.0 < vpd_kpa(25, 50) < 1.3, vpd_kpa(25, 50)
    ok = {"temp_c": 25, "rh": 62, "soil_raw": 900, "lux": 20000}
    assert recommend("veg", ok, 18) == ["All in range. No action."], recommend("veg", ok, 18)
    hot_dry = recommend("flower", {"temp_c": 30, "rh": 30, "soil_raw": SOIL_DRY, "lux": 20000})
    assert any("Temperature high" in s for s in hot_dry)
    assert any("water now" in s and "Bloom" in s for s in hot_dry)
    dark_wet = recommend("ripening", {"temp_c": 20, "rh": 60, "soil_raw": 900, "lux": 5}, 18)
    assert any("Bud rot" in s for s in dark_wet)
    assert any("Light leak" in s for s in dark_wet)
    assert any("expected 12h" in s for s in dark_wet)
    assert not any("Light leak" in s for s in recommend("veg", {**ok, "lux": 5}))  # veg dark is fine
    print("ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=TARGETS, help="set the current stage and exit")
    p.add_argument("--event", help="log a grower action and exit")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--test", action="store_true")
    p.add_argument("--every", type=int, default=3600, help="seconds between readings")
    a = p.parse_args()
    if a.test: raise SystemExit(test())
    DATA.mkdir(exist_ok=True)
    if a.stage:
        (DATA / "stage.txt").write_text(a.stage)
        a.event = f"stage -> {a.stage}"
    if a.event:
        raise SystemExit(append_csv("events.csv", ["ts", "stage", "event"], [now(), current_stage(), a.event]))
    while True:
        tick(a.fake)
        if a.once: break
        time.sleep(a.every)
