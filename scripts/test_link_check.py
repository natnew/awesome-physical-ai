"""Offline regression tests for the link-check configuration and catalog links."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VERIFIED_URLS = (
    *(f"https://www.iso.org/standard/{number}.html"
      for number in ("73934", "81230", "53820", "43464")),
    "https://www.anybotics.com/robotics/anymal/",
    *(f"https://x.com/{handle}" for handle in (
        "ylecun", "drfeifei", "pabbeel", "svlevine", "chelseabfinn",
        "russtedrake", "grok_", "rodneyabrooks", "petitegeek", "robotlaunch",
        "adcock_brett", "austinsemis", "soumithchintala", "karpathy",
    )),
)
# Destinations that returned 404 or timed out in link-check runs (2026-10) and
# were replaced with verified pages, plus X handles that resolved to the wrong
# account (403 in CI, so never whitelisted). They must not be reintroduced.
KNOWN_BROKEN_URLS = (
    "https://docs.ros.org/en/rolling/Tutorials/",
    "https://sapien.ucsd.edu/",
    "https://x.com/dieterfox",
    "https://x.com/robohub",
)
CATALOG_FILES = (
    ROOT / "README.md",
    *sorted((ROOT / "website" / "docs").rglob("*.md")),
    *sorted((ROOT / "website" / "docs").rglob("*.mdx")),
)


class LinkCheckExclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patterns = [
            re.compile(line.strip())
            for line in (ROOT / ".lycheeignore").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def is_excluded(self, url: str) -> bool:
        return any(pattern.search(url) for pattern in self.patterns)

    def test_verified_pages_are_excluded(self) -> None:
        for url in VERIFIED_URLS:
            with self.subTest(url=url):
                self.assertTrue(self.is_excluded(url), url)

    def test_exceptions_do_not_cover_neighbouring_urls(self) -> None:
        for url in VERIFIED_URLS:
            for other in (
                url.rstrip("/") + "/not-a-verified-page",
                url.rstrip("/") + "-not-a-verified-page",
                url + "?not-a-verified-page",
                "https://example.org/" + url,
            ):
                with self.subTest(url=other):
                    self.assertFalse(self.is_excluded(other), other)

    def test_unverified_and_broken_links_remain_checked(self) -> None:
        for url in (
            "https://www.iso.org/standard/00000.html",
            "https://www.anybotics.com/robotics/not-a-verified-robot/",
            "https://x.com/not_a_verified_profile",
            "https://x.com/dieterfox",
            "https://x.com/robohub",
            "https://docs.ros.org/en/rolling/Tutorials/",
            "https://docs.ros.org/en/rolling/First-Steps.html",
            "https://sapien.ucsd.edu/",
            "https://sapien-sim.github.io/docs",
            "https://example.org/missing",
            "http://127.0.0.1:8000/missing",
            "file:///missing.md",
        ):
            with self.subTest(url=url):
                self.assertFalse(self.is_excluded(url), url)


class CatalogLinkRegressionTests(unittest.TestCase):
    def test_known_broken_urls_are_not_reintroduced(self) -> None:
        self.assertTrue(CATALOG_FILES, "no catalog files found")
        for path in CATALOG_FILES:
            content = path.read_text(encoding="utf-8")
            for url in KNOWN_BROKEN_URLS:
                with self.subTest(file=path.relative_to(ROOT).as_posix(), url=url):
                    self.assertNotIn(url, content)


if __name__ == "__main__":
    unittest.main()
