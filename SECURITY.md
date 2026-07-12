# Security Policy

## Supported Surface

This repository is a public research benchmark package. Security and safety
reports are most relevant when they affect:

- public JSONL/JSON artifacts;
- release manifests and checksums;
- prompt/hidden-label separation;
- leakage of local paths, credentials, private database references, or
  infrastructure breadcrumbs;
- scripts that could accidentally publish private artifacts.

## Reporting

Please do not open a public issue for suspected secrets, private data exposure,
or sensitive provenance leakage.

Instead, contact the repository owner through GitHub with a brief description
of:

- affected file or artifact;
- suspected exposure type;
- whether the issue appears in current files, git history, or both;
- suggested minimal reproduction, if available.

## Handling

Reports will be triaged as benchmark integrity issues. The expected response is
to:

- remove or redact the affected artifact;
- update the relevant checker where possible;
- regenerate checksums and manifests;
- document the correction in the release audit.

## Non-Security Scientific Issues

For benchmark design concerns, evaluator disagreements, or documentation
clarity, please use a normal GitHub issue.
