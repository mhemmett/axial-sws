# `site/` — Axial SWS project site

A small hand-written static site summarising the shear-wave-splitting results.
Live at <https://mhemmett.github.io/axial-sws/>.

## ⚠️ The password gate is cosmetic — this repo is PUBLIC

The gate is **client-side JavaScript only**. The password, the page content, and
every figure are plain files in a public GitHub repository, readable straight
from the repo or from the deployed asset URLs without ever seeing the prompt. It
keeps out casual visitors and search engines. That is the entire extent of the
protection — it is not access control.

So: **do not commit anything under `site/` that would be damaging to leak** — no
unpublished manuscript text, no pre-publication results you are not prepared to
have scooped, nothing embargoed. Assume everything here is already public,
because it is.

If that has to change, the hosting model changes with it: ship an encrypted
bundle the password actually decrypts, or move to a host with real server-side
auth. More JavaScript in the gate does not help.

## Layout — no build step

```
index.html            page shell, styles, gate  — hand-written, served as-is
content.js            prose/section content consumed by index.html
figures.json          figure manifest: source file, page, caption, tab
build_figures.py      renders figures.json -> assets/figures/<tab>/*.webp
assets/figures/<tab>/ generated figure assets (tracked — see below)
.nojekyll             stops Pages running Jekyll, so files are served
                      byte-for-byte and nothing underscore-prefixed is skipped
```

No npm, no bundler, no `package.json`. `index.html` and `content.js` are edited
by hand and served exactly as committed.

## Regenerating figures

```bash
conda run -n seismo python site/build_figures.py
```

Driven entirely by `figures.json`. Its `source` paths point at figures elsewhere
in the repo (`scripts/`, `lqt_pykonal_combined_results/`, …) — those are
**read-only inputs**, never modified, moved, or committed by the build. Only
`site/assets/figures/` is written.

## Previewing locally

```bash
python3 -m http.server -d site 8000   # then open http://localhost:8000/
```

Use a server, not `file://` — the fetch/module loads fail on a `file:` origin.

## Deployment

`.github/workflows/pages.yml` publishes `site/` on every push to `main` touching
`site/**` or the workflow file, plus manual dispatch from the Actions tab.

**One-time manual step:** enable Pages in **Settings → Pages → Build and
deployment → Source: "GitHub Actions"**. The workflow cannot do this itself;
until then runs fail at "Setup Pages". If you bump `upload-pages-artifact` past
v3, add `include-hidden-files: true` — v4+ drops dotfiles, i.e. `.nojekyll`.

## Tracking

The repo's `.gitignore` blanket-ignores `*.png`/`*.pdf`/etc. and any directory
named `results/`, which would otherwise swallow `site/assets/figures/results/`.
A negation block at the end of `.gitignore` re-includes the whole `site/` tree.
If a new site file mysteriously will not `git add`, look there first.
