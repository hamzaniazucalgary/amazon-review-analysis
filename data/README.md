# Data

This directory stores the Amazon Reviews Polarity dataset.

## Source

- **Dataset:** Amazon Reviews Polarity (Zhang et al., 2015)
- **HuggingFace:** [amazon_polarity](https://huggingface.co/datasets/amazon_polarity)
- **Size:** 3,600,000 training / 400,000 test reviews
- **Classes:** Negative (1) / Positive (2)

## Schema

| Column   | Type   | Description                    |
|----------|--------|--------------------------------|
| polarity | int    | 1 = negative, 2 = positive    |
| title    | string | Review title                   |
| text     | string | Review body                    |

## Download

```bash
python data/download_data.py --source huggingface --format parquet
```

Files will be saved as `train.parquet` and `test.parquet` (~300MB compressed).
