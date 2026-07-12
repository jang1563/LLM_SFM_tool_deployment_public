## Summary

Describe the benchmark, artifact, or documentation change.

## Public-Surface Checklist

- [ ] Model-visible fields remain separated from hidden evaluator metadata.
- [ ] New public artifacts are listed in `release/public_release_manifest.json`
      when appropriate.
- [ ] JSONL record counts and checksums are regenerated for changed artifacts.
- [ ] No private databases, local paths, API keys, tokens, raw logs, or cluster
      breadcrumbs are included.
- [ ] Biological claims are tied to explicit artifacts, commands, or
      limitations.

## Validation

- [ ] `python scripts/check_public_release.py`
- [ ] `python scripts/check_public_git_history.py`
- [ ] `python post_training/validate_post_training_data.py`
- [ ] `python examples/run_public_demo.py`
- [ ] Public-safe pytest subset or full `python -m pytest -q`
