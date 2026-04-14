# Theory And Explanations — Model Training In This Project

This document provides short theory and practical explanations for the three ML training tracks used in the project, with examples and training details.

**5. Model Training (Baselines)**

**Theory (Why This Works)**
- **TF-IDF** converts text into numeric vectors where each term is weighted by how important it is in a document and how rare it is across the corpus. This highlights discriminative words like "urgent" or "unsubscribe".
- **N-grams (1–2)** capture both single words and short phrases (e.g., "action required"). This improves class separation for emails that have common short phrases.
- **Structured features** (keyword counts, reply prefix, deadline flag) inject domain-specific signals that text models can miss.
- **Linear classifiers** are fast, stable, and perform strongly on sparse TF‑IDF vectors.

**How It Is Implemented**
- Text and structured features are combined using a `ColumnTransformer` pipeline.
- TF‑IDF is applied to the `text` field (subject + body).
- Structured signals are extracted from `subject`, `body`, `sender`, and `has_reply_prefix`.
- Models are trained and evaluated on a train/validation split.

**Training Details**
- Data source: unified dataset from Enron + TREC 2007 (+ optional BC3).
- Split: `train.csv` and `val.csv` created by `datasets/build_dataset.py`.
- Vectorizer: TF‑IDF with `min_df=2`, `max_df=0.9`, `ngram_range=(1,2)`.
- Models:
- Logistic Regression (class‑weighted, max_iter=1000).
- Linear SVC with calibration (for probabilities).
- SGDClassifier with log‑loss.

**Example (Baseline Training Command)**
```bash
python models/train.py
```

**Example (Input Row And Features)**
Input:
- Subject: "Re: Action required"
- Body: "Please resolve this by EOD."

Signals extracted (illustrative):
- `urgency_keyword_count = 1`
- `followup_keyword_count = 0`
- `action_phrase_count = 1`
- `deadline_flag = 1`
- `has_reply_prefix = 1`

TF‑IDF features include tokens like:
- `re`, `action`, `required`, `resolve`, `eod` and bigrams like `action required`.

**Why It Helps The Project**
- Fast to train and run for production.
- Works well with weakly-labeled data.
- Provides a strong baseline for comparison.

---

**6. Embedding-Based Baseline (MiniLM)**

**Theory (Why This Works)**
- Sentence‑Transformers generate dense vector embeddings that capture semantic similarity, not just word overlap.
- MiniLM is lightweight and fast while retaining strong semantic representation.
- Combining embeddings with structured signals captures both meaning and domain heuristics.

**How It Is Implemented**
- `SentenceTransformer` encodes the merged email text into a dense vector.
- The embedding vector is concatenated with structured features.
- Logistic Regression is trained on this combined feature set.

**Training Details**
- Embeddings model: `sentence-transformers/all-MiniLM-L6-v2`.
- Batch size: `64` (configurable).
- Device: GPU if available, otherwise CPU.

**Example (MiniLM Training Command)**
```bash
python models/train.py
```

**Example (Semantic Benefit)**
- Email A: "Please fix this today, it is urgent"
- Email B: "This needs immediate attention"

TF‑IDF might treat these as different because words differ, but MiniLM embeddings place them close in vector space, enabling better recall for `urgent`.

**Why It Helps The Project**
- Better at paraphrases and semantic cues.
- Adds a higher‑quality baseline without full transformer fine‑tuning.

---

**7. Transformer Fine-Tuning (DistilBERT)**

**Theory (Why This Works)**
- Transformers learn contextual representations of text, capturing meaning based on surrounding words and long‑range dependencies.
- Fine‑tuning adapts a pretrained language model (DistilBERT) to the email classification task.
- Compared to TF‑IDF, transformers can model subtle intent and context.

**How It Is Implemented**
- Training uses `AutoTokenizer` and `AutoModelForSequenceClassification` from Hugging Face.
- Inputs are tokenized with padding and truncation to `max_length`.
- Cross‑entropy loss is optimized with AdamW and a linear warmup schedule.
- Checkpoints are saved each epoch, along with metrics and plots.

**Training Details**
- Base model: `distilbert-base-uncased`.
- Default epochs: `4`.
- Batch size: `16` (number of samples processed together before each optimizer update; larger batches use more memory but can improve throughput).
- Max length: `256` (maximum number of tokens per email; longer emails are truncated, shorter ones are padded).
- Checkpoints: `models/artifacts/distilbert/checkpoints/epoch_*.pt`.
- Metrics saved: `models/artifacts/distilbert/metrics.json`.
- Plots saved: `loss_curve.png`, `accuracy_curve.png`.

**Example (Transformer Training Command)**
```bash
python models/train_transformer.py --model distilbert-base-uncased --epochs 4 --batch-size 16 --max-length 256
```

**Example (Why Context Matters)**
- Email: "I did not say this was urgent".
- TF‑IDF might highlight "urgent" and push the class toward `urgent`.
- The transformer uses context and may reduce urgency because of negation.

**Why It Helps The Project**
- Highest potential accuracy.
- Handles nuanced intent and phrasing.
- Produces a strong model for real inbox data.

---

**General Training Process (Applies To All Models)**
- Build dataset: `python datasets/build_dataset.py`.
- Train models: `python models/train.py` for baselines or `python models/train_transformer.py` for transformer.
- Evaluate: `python models/evaluate.py`.
- Store artifacts and metrics under `models/artifacts/` for reproducibility and comparison.

**Dataset Split And Samples Per Epoch (Current Build)**
- Total samples: `517,402`.
- Train samples per epoch (baselines): `413,921`.
- Validation samples per epoch: `103,481`.
- Test split: not created in this project; only train/val are generated.
- Transformer training default: uses `--max-samples 100000`, so it trains on `100,000` samples per epoch unless overridden.
