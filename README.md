# Amber

A cannabis grow advisor running on a Raspberry Pi. Amber watches one tent with a camera and a few sensors, and tells the grower what to do next: water, feed, fix the climate, check the light timer. It's named after amber trichomes, the classic harvest signal.

It's built only for cannabis, not as a general plant monitor. Stages, targets, feed advice and warnings all follow cannabis grow practice.

## How it works

Every hour `amber.py`:

1. Reads soil moisture, air temperature/humidity and light level
2. Takes a photo, filed under the current stage (`data/photos/<stage>/`), and skips it when the lights are off
3. Logs the readings to `data/log.csv`
4. Prints recommendations for the current stage

The photos become training data. Once there's enough of them, `train.py` fine-tunes a pretrained image model (EfficientNet-B0) to recognize the stage from the camera, so it no longer has to be set by hand.

## What it checks

| Check | Why it matters for cannabis |
|---|---|
| Temperature, humidity and VPD by stage | Each stage has its own range. Late flower needs low humidity. |
| Soil moisture | Says when to water, with that stage's feed advice |
| Hours of light per day | Photoperiod plants need about 18/6 in veg and 12/12 in flower |
| Light during the dark period in flower | Light leaks can turn plants hermaphrodite (nanners) |
| High humidity in flower and ripening | Bud rot (botrytis) risk in dense colas |

Stages: `seedling` → `veg` → `preflower` (the 2–3 week stretch after the flip) → `flower` → `ripening`

## Hardware

- Raspberry Pi + Pi camera module
- [Adafruit STEMMA soil sensor](https://www.adafruit.com/product/4026) (seesaw, I2C)
- [Adafruit SHT4x](https://www.adafruit.com/product/4885): temperature and humidity
- [Adafruit TSL2591](https://www.adafruit.com/product/1980): light

Everything connects over I2C (STEMMA QT cables daisy-chain).

## Setup (Pi)

```sh
sudo apt install python3-picamera2
pip install adafruit-circuitpython-seesaw adafruit-circuitpython-sht4x adafruit-circuitpython-tsl2591
python amber.py --test          # self-check, no hardware needed
```

Before the first grow, edit the settings at the top of `amber.py`:

- `PLANT_TYPE`: `"photoperiod"` or `"auto"`
- `SOIL_DRY` / `SOIL_WET`: read the soil sensor in bone-dry soil and right after watering
- `LIGHTS_ON_LUX` / `LEAK_LUX`: adjust if your tent reads differently

## Usage

```sh
python amber.py                              # run hourly (use systemd to keep it running)
python amber.py --stage preflower            # change stage; labels every photo from now on
python amber.py --event "fed 5ml/gal bloom"  # log what you did
python amber.py --fake --once                # try it without hardware
```

Grower actions go to `data/events.csv` and are kept as future training labels for the recommendations.

## Training the stage model

Run on a machine with a GPU or Apple Silicon, after copying `data/photos/` over from the Pi:

```sh
pip install torch torchvision
python train.py --test   # self-check
python train.py          # writes stage_model.pt
```

Train and validation photos are split by day rather than at random. Hourly shots from the same day are near-duplicates, so a random split would make accuracy look far better than it is. The reported accuracy is averaged across stages, so a model that always guesses "flower" can't score well.

## Roadmap

- [x] Sensor + photo collection, cannabis rule-based advice
- [x] Stage model training pipeline
- [ ] Collect one full grow of labeled photos
- [ ] Load the trained model on the Pi to set the stage automatically
- [ ] Hermaphrodite / pollen sac detection
- [ ] Trichome ripeness (clear / cloudy / amber) with a macro lens
- [ ] Deficiency and bud rot / powdery mildew spotting
- [ ] EC/pH probe for real nutrient dosing
- [ ] Phone alerts

## History

Amber revives [growtent](https://github.com/drewjordan414/growtent) (2023), a React/Flask/MySQL dashboard that never got to the model.

## Legal

Check your local laws before growing cannabis. Amber gives advice; the grower is responsible for the grow.
