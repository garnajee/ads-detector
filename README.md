# Ads detector (and automatic deletion)

This repository aims at detecting and deleting ads for a specifiv tv channel and tv show.

```
ads-detector/
│
├── detect_ads.py               # Main script to run the detector
│
├── src/                        # Core library for the ad detection logic
│   ├── __init__.py
│   ├── logo_detector.py        # Contains the LogoDetector class
│   ├── video_processing.py     # Functions for frame extraction, precise boundary detection
│   ├── data_models.py          # Dataclasses
│   ├── config.py               # Central place for constants and thresholds
│   └── utils.py                # Utility functions
│
├── logo_dataset/               # folder with logo PNGs
│   └── ... (logo_A.png, etc.)
│
├── README.md
└── requirements.txt
```

Other interresting scripts:

```
├── extract-frame-logo.py       # extract logo png from video with given coordinates
├── keep-ads-only.py            # keep only ads from the video with a given list of range
├── delete_ads.py               # delete ads with a given list of ranges
├── detect-and-delete-ads.py    # obvious.
```

## How to run

- prepare env:

```bash
python3 -m venv venv
pip install -r requirements.txt
source venv/bin/activate
```

- detect ads:

```bash
python3 detect_ads.py video.mkv logo_dataset
```

- delete ads:

```bash
# modify the if __name__ ... part, especially the input/output file
# and the original_remove_ranges
python3 delete_ads.py
```

- detect and delete ads

```bash
python detect-and-delete-ads.py --help
python detect-and-delete-ads.py video.mkv logo_dataset
```
