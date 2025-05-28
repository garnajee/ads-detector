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

## upload it to [gofile](gofile.io)

> This programm will automatically find today's video and upload it to gofile

First need to install pip package:

```bash
pip install python-dotenv
pip install gofilepy-0.4.0-py3-none-any.whl
```

This package was built from [here](https://github.com/garnajee/Gofile).

Then, modify `.env` file, and add your token and/of gofile's folder id

```bash
cp .env.example .env
vim .env
```

- run

```bash
python up-gofile.py
```

You can upload it to an existing folder or if `folder_id` is not set, it will create one automatically.

> [!NOTE]
> If you want to upload to an existing folder, you need to give the id which is something like this `a08f31e7-d478-4097-5673-50g8391c2e8d` (uuid4 style)

