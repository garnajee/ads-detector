# Architecture fichiers

This repository aims at detecting and deleting ads for a specifiv tv channel and tv show.

```
ads-detector/
│
├── main.py                     # Main script to run the detector
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

## How to run

- detect ads:

```
python3 -m venv venv
pip install -r requirements.txt
python3 main.py video.mkv logo_dataset
```

