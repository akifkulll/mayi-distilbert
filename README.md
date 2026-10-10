# DistilBERT

Fine-tunes `distilbert-base-uncased` to label an agent's proposed tool call **allow (0)** or
**deny (1)**, in two input formats from the cleaned dataset:

- **A** (`input_a`): the tool call alone.
- **B** (`input_b`): the tool call, the user's request and the agent's earlier steps.

Comparing A and B measures how much the context helps.

## Files

| file | what it does |
|---|---|
| `distilbert_colab.ipynb` | Runs everything in Colab (free T4): clone the data, train A and B, score them. |
| `train_distilbert.py` | Trains one model and writes `model/`, `predictions/<split>.csv` (`id, label, p_deny`) and `run.json` (settings, time, speed). |
| `evaluate_predictions.py` | Scores any `id, label, p_deny` file: metrics overall and by length, difficulty and framework, a confusion matrix and the 200 most confident mistakes. Works for every model in the project. |
| `push_to_hub.py` | Uploads a trained model and its model card to the Hugging Face Hub. |
| `predict.py` | Loads a model from the Hub (or a folder) and scores one text or a whole CSV. |
| `Dockerfile`, `requirements.txt` | A CPU image that runs `predict.py`; the model is downloaded at run time. |

## Training setup

| | A | B |
|---|---|---|
| train rows (stratified sample) | 100,000 (and a 60,000 run) | 60,000 |
| max tokens | 128 | 512 |
| batch size | 64 | 16 |
| learning rate | 3e-5, 6% linear warm-up, linear decay | same |
| epochs | 1 | 1 |
| model selection | best macro-F1 on a 5,000-row val sample, every 500 steps | same |

## Results (test split, 71,613 rows, threshold 0.5)

| model | accuracy | macro-F1 | false allow rate | false deny rate | AUROC |
|---|---:|---:|---:|---:|---:|
| A | 0.8995 | 0.8995 | 0.0991 | 0.1018 | 0.9595 |
| B | 0.9399 | 0.9399 | 0.0605 | 0.0597 | 0.9846 |

**Metric choice.** The data is balanced (about 50% deny), so accuracy is not misleading, but we
rank by **macro-F1**, which weighs both classes equally. The two errors do not cost the same:
letting a harmful call through is worse than blocking a safe one, so we also report the **false
allow rate** (share of deny rows predicted allow). **AUROC** shows how well the scores separate
the classes before any threshold is chosen.

Full breakdowns are in `results/`.

## Reproduce

Training: open `distilbert_colab.ipynb` in Colab with a T4 GPU and run all cells (A ~5 min,
B ~16 min of training).

Scoring with the published model, no GPU needed:

```bash
docker build -t mayi-distilbert .
echo "### PROPOSED TOOL CALL ..." | docker run -i --rm -v hf-cache:/cache mayi-distilbert --text -
docker run --rm -v hf-cache:/cache -v "$PWD/data:/data" mayi-distilbert \
    --input /data/test.csv.gz --out /data/predictions/test.csv
```

Use the A model with `-e MODEL_ID=akifkulll/mayi-distilbert-a ... --text-col input_a`.
Checked on a laptop CPU with Docker Desktop: the outside test set `ood_approve_or_deny`
(1,448 rows) gives accuracy 0.8722, the same as on the Colab GPU. CPU speed for B is about
9 rows/s (GPU T4: about 265 rows/s), so score a sample before a whole split.
