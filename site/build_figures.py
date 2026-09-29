#!/usr/bin/env python3
"""
Build the web-ready figure assets for the Axial SWS static site.

Reads site/figures.json, renders every entry into site/assets/figures/<tab>/<id>.<ext>,
and emits site/figures.js for the page to consume.

SOURCE FILES ARE READ ONLY. This script opens them and nothing else: it never writes,
moves, renames, or deletes anything outside site/.

Rendering rules (per manifest entry):
  * copy_as_is: true        -> byte-for-byte copy (the .svg pipeline diagram)
  * source endswith .png    -> PIL resize to `width` (aspect preserved) -> WebP
  * otherwise (.pdf)        -> pdftocairo renders 1-indexed `page` to PNG -> PIL -> WebP

Usage:
    conda run -n seismo python site/build_figures.py [--check] [--force]

    --check   verify every source file exists; render nothing, write nothing
    --force   re-render even when the output is newer than its source
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from PIL import Image

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

SITE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SITE_DIR)
MANIFEST = os.path.join(SITE_DIR, "figures.json")
ASSETS_DIR = os.path.join(SITE_DIR, "assets", "figures")
FIGURES_JS = os.path.join(SITE_DIR, "figures.js")

PDFTOCAIRO = shutil.which("pdftocairo") or "/opt/homebrew/bin/pdftocairo"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def human_mb(nbytes):
    return nbytes / (1024.0 * 1024.0)


def svg_intrinsic_size(path):
    """Return (w, h) floats from an SVG's viewBox, falling back to width/height attrs."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        head = fh.read(4096)
    m = re.search(r'viewBox\s*=\s*["\']\s*([\d.eE+-]+)[\s,]+([\d.eE+-]+)[\s,]+'
                  r'([\d.eE+-]+)[\s,]+([\d.eE+-]+)\s*["\']', head)
    if m:
        return float(m.group(3)), float(m.group(4))
    mw = re.search(r'\bwidth\s*=\s*["\']([\d.eE+-]+)', head)
    mh = re.search(r'\bheight\s*=\s*["\']([\d.eE+-]+)', head)
    if mw and mh:
        return float(mw.group(1)), float(mh.group(1))
    return None


def is_stale(src, out, force):
    """True if `out` needs (re)building."""
    if force:
        return True
    if not os.path.exists(out):
        return True
    return os.path.getmtime(out) < os.path.getmtime(src)


def render_pdf_page(src, page, width, out_webp, quality):
    """pdftocairo the given 1-indexed page to PNG in a temp dir, then save as WebP."""
    with tempfile.TemporaryDirectory(prefix="axial_figbuild_") as tmpdir:
        base = os.path.join(tmpdir, "page")
        cmd = [
            PDFTOCAIRO, "-png",
            "-f", str(page), "-l", str(page),
            "-singlefile",
            "-scale-to-x", str(width),
            "-scale-to-y", "-1",
            src, base,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "pdftocairo failed (%d) on %s p%s: %s"
                % (proc.returncode, src, page, (proc.stderr or "").strip())
            )
        tmp_png = base + ".png"
        if not os.path.exists(tmp_png):
            raise RuntimeError("pdftocairo produced no output for %s p%s" % (src, page))
        with Image.open(tmp_png) as im:
            im = im.convert("RGB")
            im.save(out_webp, "WEBP", quality=quality, method=6)
            return im.size
    # temp dir (and the intermediate PNG) removed here


def render_png(src, width, out_webp, quality):
    """Load a PNG, downscale to `width` preserving aspect, save as WebP."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        if im.width != width:
            height = max(1, round(im.height * (width / float(im.width))))
            im = im.resize((width, height), Image.LANCZOS)
        im.save(out_webp, "WEBP", quality=quality, method=6)
        return im.size


def dims_of_existing(path, fallback_width):
    """Pixel dimensions of an already-built output (skip path)."""
    if path.lower().endswith(".svg"):
        size = svg_intrinsic_size(path)
        if size is None:
            return (fallback_width, fallback_width)
        w, h = size
        return (fallback_width, max(1, round(fallback_width * h / w)))
    with Image.open(path) as im:
        return im.size


def js_literal(obj):
    """Pretty-printed JSON that is also valid JS."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(argv):
    check_only = "--check" in argv
    force = "--force" in argv

    with open(MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    defaults = manifest.get("defaults", {})
    default_width = int(defaults.get("width", 1600))
    quality = int(defaults.get("quality", 88))
    figures = manifest.get("figures", [])

    missing = []
    records = []
    per_tab_bytes = {}
    outputs = []  # (id, out_path, nbytes)

    # ---- source existence pass -------------------------------------------------------
    resolved = []
    for entry in figures:
        src = os.path.join(REPO_ROOT, entry["source"])
        if not os.path.exists(src):
            missing.append((entry["id"], entry["source"]))
            continue
        resolved.append((entry, src))

    if check_only:
        print("Checked %d manifest entries against %s" % (len(figures), REPO_ROOT))
        if missing:
            print("\nMISSING SOURCES (%d):" % len(missing))
            for fid, rel in missing:
                print("  %-24s %s" % (fid, rel))
            return 1
        print("All %d source files present." % len(figures))
        return 0

    if not os.path.exists(PDFTOCAIRO):
        print("ERROR: pdftocairo not found at %s" % PDFTOCAIRO, file=sys.stderr)
        return 2

    # ---- render pass -----------------------------------------------------------------
    failed = []
    for entry, src in resolved:
        fid = entry["id"]
        tab = entry["tab"]
        width = int(entry.get("width", default_width))
        # Per-entry quality override. Photographic sources (the bathymetry basemaps)
        # compress far better at a lower quality than they do at a smaller width.
        entry_quality = int(entry.get("quality", quality))
        out_dir = os.path.join(ASSETS_DIR, tab)
        os.makedirs(out_dir, exist_ok=True)

        copy_as_is = bool(entry.get("copy_as_is", False))
        ext = os.path.splitext(src)[1].lower()

        if copy_as_is:
            out = os.path.join(out_dir, fid + ext)
        else:
            out = os.path.join(out_dir, fid + ".webp")

        try:
            if not is_stale(src, out, force):
                dims = dims_of_existing(out, width)
                action = "skip"
            elif copy_as_is:
                shutil.copyfile(src, out)          # verbatim; source untouched
                dims = dims_of_existing(out, width)
                action = "copy"
            elif ext == ".png":
                dims = render_png(src, width, out, entry_quality)
                action = "png "
            else:
                page = int(entry.get("page", 1))
                dims = render_pdf_page(src, page, width, out, entry_quality)
                action = "pdf "
        except Exception as exc:  # keep going; one bad entry must not block the build
            failed.append((fid, str(exc)))
            print("  FAIL %-26s %s" % (fid, exc), file=sys.stderr)
            continue

        nbytes = os.path.getsize(out)
        per_tab_bytes[tab] = per_tab_bytes.get(tab, 0) + nbytes
        outputs.append((fid, out, nbytes))

        rel_src = os.path.relpath(out, SITE_DIR).replace(os.sep, "/")
        badge = entry.get("badge", [])
        if isinstance(badge, str):
            badge = [badge]
        records.append({
            "id": fid,
            "tab": tab,
            "group": entry.get("group"),
            "title": entry.get("title"),
            "caption": entry.get("caption"),
            "badge": list(badge),
            "featured": bool(entry.get("featured", False)),
            "src": rel_src,
            "width": int(dims[0]),
            "height": int(dims[1]),
        })

        print("  %s %-26s %5dx%-5d %7.0f KB  %s"
              % (action, fid, dims[0], dims[1], nbytes / 1024.0, rel_src))

    # ---- figures.js ------------------------------------------------------------------
    meta = {
        "badges": manifest.get("badges", {}),
        "groups": manifest.get("groups", {}),
    }
    js = (
        "// Generated by site/build_figures.py - do not edit by hand.\n"
        "window.AXIAL_FIGURE_META = " + js_literal(meta) + ";\n\n"
        "window.AXIAL_FIGURES = " + js_literal(records) + ";\n"
    )
    with open(FIGURES_JS, "w", encoding="utf-8") as fh:
        fh.write(js)

    # ---- summary ---------------------------------------------------------------------
    total = sum(n for _, _, n in outputs)
    print("\n%-14s %s" % ("figures.js", os.path.relpath(FIGURES_JS, SITE_DIR)))
    print("%d figures rendered, %d entries in figures.js" % (len(outputs), len(records)))

    print("\nPer-tab payload:")
    for tab in sorted(per_tab_bytes, key=lambda t: -per_tab_bytes[t]):
        print("  %-10s %7.2f MB" % (tab, human_mb(per_tab_bytes[tab])))
    print("  %-10s %7.2f MB" % ("TOTAL", human_mb(total)))

    print("\nLargest 5 outputs:")
    for fid, out, nbytes in sorted(outputs, key=lambda r: -r[2])[:5]:
        print("  %7.0f KB  %-26s %s"
              % (nbytes / 1024.0, fid, os.path.relpath(out, SITE_DIR)))

    if missing:
        print("\nMISSING SOURCES (%d) - these entries were skipped:" % len(missing),
              file=sys.stderr)
        for fid, rel in missing:
            print("  %-26s %s" % (fid, rel), file=sys.stderr)
    if failed:
        print("\nRENDER FAILURES (%d):" % len(failed), file=sys.stderr)
        for fid, err in failed:
            print("  %-26s %s" % (fid, err), file=sys.stderr)

    if human_mb(total) > 6.0:
        print("\nWARNING: total payload %.2f MB exceeds the ~6 MB budget." % human_mb(total),
              file=sys.stderr)

    return 1 if (missing or failed) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
