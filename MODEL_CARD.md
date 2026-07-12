# Model Card: A2 MONDO Band Reranker

Draft Hugging Face model card for the planned LoRA adapter release.

This is an optional future model surface. The primary public-release object for
the current repository is the Stage A tool-use trajectory benchmark and its
validated data artifacts.

## Model Summary

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Adaptation: LoRA SFT
- Task: closed-set disease entity reranking for ambiguous MONDO matches
- Input: clinical-trial condition string plus SapBERT top-8 MONDO candidates
- Output: candidate number `1`-`8`, or `0` for none of these
- Status: draft card; model repository not yet published

## Intended Use

This adapter is intended for biomedical entity-resolution research where a
retriever has already produced a small candidate set and the model must choose a
matching disease concept or abstain. It is not a diagnosis model, not a clinical
decision system, and not a general medical question-answering model.

## Evaluation

De-leaked ambiguous-band evaluation. The held-out query string was removed from
displayed candidate synonym lists before evaluation.

| Method | band_cases pick accuracy |
|---|---:|
| String-match floor | 0.285 |
| SapBERT top-1 retrieval | 0.750 |
| Qwen2.5-7B zero-shot rerank | 0.810 |
| Qwen2.5-1.5B SFT LoRA | 0.875 |
| Claude-haiku rerank reference | 0.900 |
| Gold-in-top8 ceiling | 0.970 |

Full pick-or-abstain band-eval accuracy:

| Model | Base | SFT |
|---|---:|---:|
| Qwen2.5-1.5B | 0.371 | 0.773 |

## Training Data

Training examples are generated from MONDO labels/synonyms and SapBERT top-8
candidate retrieval in the `a2_freetext/` pipeline. Splits are concept-disjoint
by gold MONDO identifier. Candidate synonym displays are de-leaked by removing
the query string itself.

## Limitations

- The model reranks only candidate IDs supplied by the retriever. It cannot
  recover if the correct MONDO concept is absent from the candidate set.
- Evaluation uses ontology and registry-surface variation, not a full public
  stream of real clinical-trial search logs.
- The adapter improves the open zero-API stack but does not fully match the
  proprietary reference on the corrected band evaluation.
- The current release is a research artifact and should not be used for clinical
  diagnosis or treatment decisions.

## Reproducibility

Public-safe evaluator demo:

```bash
python examples/run_public_demo.py
```

Representative commands after preparing local A2 artifacts:

```bash
python a2_freetext/precompute_band.py
python a2_freetext/build_band_rl_dataset.py
python a2_freetext/sft_train.py Qwen/Qwen2.5-1.5B-Instruct 2
```

## Citation

See `CITATION.cff` in the repository.
