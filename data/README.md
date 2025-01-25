# FPL Data

This directory contains Fantasy Premier League (FPL) data.

## Data Structure 

## How to Get Data

1. Download raw data from [FPL Archive](URL_TO_DATA_SOURCE)
2. Place the downloaded files in the corresponding season directories
3. Run preprocessing script:
   ```bash
   python src/preprocess.py
   ```

## Data Format

Each gameweek file contains:
- Player statistics
- Match results
- etc.

The processed file combines all gameweeks and adds:
- Position information
- Team information
- etc.