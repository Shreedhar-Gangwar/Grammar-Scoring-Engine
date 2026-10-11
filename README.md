# Grammar Scoring Engine for spoken English

Predicts a grammar score from 1 to 5 for a 45 to 60 second spoken answer, built for SHL's Kaggle
challenge (769 training recordings, 216 test recordings, scored on RMSE and Pearson correlation).

Everything is in one notebook, `grammar_scoring_final.ipynb`, with a written report at the top.

## Results

| | Value |
|---|---|
| Public leaderboard (about 60% of the test set) | 0.3358 |
| Out of fold, speaker folds (no speaker in both training and validation) | mix 0.3630, RMSE 0.547, Pearson 0.822 |
| Out of fold, topic folds (a whole topic held out at a time) | mix 0.3882, RMSE 0.578, Pearson 0.801 |
| Training RMSE (in sample, 732 files) | 0.441, Pearson 0.901 |

`mix` = 0.5 x RMSE + 0.5 x (1 - Pearson), lower is better. The leaderboard behaved like this mix.
Out of fold numbers are weighted to the test mix of short and long recordings. All numbers are printed in
sections 6 and 7 of the notebook.

## Approach

| Step | What | Output |
|---|---|---|
| 1 | Whisper large-v3-turbo transcribes each recording | transcript and an audio embedding |
| 2 | Two sentence encoders (MPNet, bge-large) read the transcript; topic effects removed | two text embeddings |
| 3 | Simple and grammar-specific measurements (CoLA acceptability, grammar-correction edit rate, GPT-2 surprise, pauses, vocabulary) | about 34 small features |
| 4 | Two LLMs (Qwen2.5-7B, Phi-3.5-mini) are asked to grade the grammar with the rubric | their hidden state at the answer position |
| 5 | DeBERTa-v3-base fine-tuned on the transcripts | one predicted score |
| 6 | One model per block (ridge, or boosted trees for the small features) | one prediction per block |
| 7 | A linear model with non-negative weights blends the blocks | the final score |

## Validation

What the data showed, and how validation was built around it:

* **Speakers repeat in training but not in test.** Folds keep each speaker (found by voice similarity) on
  one side only. Random folds were too optimistic.
* **The short test recordings mostly answer questions that are rare in training.** Text embeddings are
  made topic neutral, and every result is also checked with topic folds.
* **Short recordings are 24% of training and 69% of test.** Scores are weighted to the test mix.
* **37 training files have score 0**, outside the 1 to 5 rubric. They are left out of training.

The largest single gain came from fine-tuning DeBERTa instead of using frozen embeddings
(leaderboard 0.3440 to 0.3358). The notebook's section 8 lists every step and the ideas that did not
help.

## How to run

1. Create a Kaggle notebook and import `grammar_scoring_final.ipynb`.
2. Add the competition data. Set the accelerator to **GPU T4 x2** (the 7B model needs both GPUs) and
   turn Internet on to download the models.
3. Run all. From scratch this takes about 1.5 to 2 hours. Slow steps save their results to
   `/kaggle/working`; if those files are attached as an input dataset, they are reused and the notebook
   finishes in about 15 minutes.
4. `submission.csv` is written to `/kaggle/working`.

Main libraries: PyTorch, Hugging Face Transformers, scikit-learn, librosa. Versions are in
`requirements.txt`.

## Repository contents

* `grammar_scoring_final.ipynb`: code, results, plots and report
* `requirements.txt`: library versions used on Kaggle
* The competition data and cached features are not included.
