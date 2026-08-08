# LoRA Fine-Tuning Results: RBI Circular Department Classifier

## Task
Classify a circular's department (DOR, DOS, FIDD, OTHER) from its opening text.

## Why fine-tuning instead of RAG/prompting
Closed-set, high-volume classification. A fine-tuned model gives
deterministic, millisecond, near-zero-cost inference per document --
the right tool when the answer space is fixed and the task repeats at
scale, unlike open-ended question answering, which needs RAG's grounding
in retrieved evidence.

## Setup
- Base model: distilbert-base-multilingual-cased
- Method: LoRA (peft), r=8, alpha=16, dropout=0.1, target_modules=[q_lin, v_lin]
- Trainable params: 741,124 / 136,068,872 (0.54%)
- Training data: 109 examples (first chunk of each circular + department label)
- Validation data: 27 examples
- Hardware: Google Colab free T4 GPU
- Training time: 35 seconds on GPU (vs 35 minutes on CPU -- same final metrics,
  confirming GPU only affects speed, not correctness)

## Label distribution
| Class | Train+Val count |
|---|---|
| DOR | 70 |
| DOS | 57 |
| FIDD | 5 |
| OTHER (merged: DCM, FMRD, CO, DPSS -- 1 example each) | 4 |

## Results (validation set, n=27)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| DOR | 0.88 | 1.00 | 0.93 | 14 |
| DOS | 1.00 | 0.91 | 0.95 | 11 |
| FIDD | 1.00 | 1.00 | 1.00 | 1 |
| OTHER | 0.00 | 0.00 | 0.00 | 1 |
| **Accuracy** | | | **0.93** | 27 |
| **Macro F1** | | | **0.72** | |

## Known limitation
The OTHER class (3 training examples spanning 4 unrelated departments)
failed entirely. This is a direct consequence of corpus size -- 4
departments had only 1 example each in the full 150-document corpus,
not enough to learn or validate a class on. The fix is more historical
data per rare department, not a different model or approach. DOR and
DOS, which cover 93% of labeled circulars, classify strongly (93-95% F1).

## Adapter
Pushed to: https://huggingface.co/Aditideo/rbi-circular-department-classifier-lora
