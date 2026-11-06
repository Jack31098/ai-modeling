"""Fetch six small, pinned MDN files. No model or mutable branch downloads."""

import argparse
import json
from pathlib import Path
import urllib.request

from training.word_boundary_v0.io_utils import json_bytes, sha, write_identical_or_new

BASE = Path(__file__).parent
REPOS = {
    "en-us": ("mdn/content", "05b21c9303964f221febe77f053e5636f230afe9"),
    "zh-cn": ("mdn/translated-content", "79c5df5882bc51786844b8fa72a06a80a24f6404"),
}


def acquire():
    entries = []
    for lang, (repo, revision) in REPOS.items():
        for path in ("LICENSE.md", *(f"files/{lang}/glossary/{topic}/index.md" for topic in ("unicode", "utf-8"))):
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
            request = urllib.request.Request(url, headers={"User-Agent": "ai-modeling-small-source-audit"})
            with urllib.request.urlopen(request, timeout=25) as response:
                raw = response.read(262145)
            if len(raw) > 262144:
                raise ValueError("source exceeds small-file budget")
            local = "sources/" + lang + "/" + ("LICENSE.md" if path == "LICENSE.md" else path.split("/")[-2] + ".md")
            write_identical_or_new(BASE / local, raw)
            entries.append({"repository": repo, "revision": revision, "path": path, "url": url,
                            "local_path": local, "bytes": len(raw), "sha256": sha(raw),
                            "retrieved_on": "2026-11-03", "timezone": "America/Los_Angeles"})
    write_identical_or_new(BASE / "sources.lock.json", json_bytes({"files": entries}))
    return entries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps({"files": len(acquire())}))
