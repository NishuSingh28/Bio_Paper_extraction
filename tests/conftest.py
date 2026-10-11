"""Shared test setup: import path, a fake embedder, a fake model, fake NCBI.

Nothing here touches the network or downloads a model, except the column
tests that use the real sentence-transformers router (see test_columns.py).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import embedder as embedder_module  # noqa: E402
import fetch  # noqa: E402
from llm import LLMResult  # noqa: E402


class FakeEmbedder(embedder_module.Embedder):
    """Deterministic hashed bag-of-words vectors. Counts its model calls.

    `salt` stands in for different weights under the same model name.
    """

    provider = "fake"

    def __init__(self, model_name: str = "fake-a", revision: str = "",
                 **options: Any):
        super().__init__(model_name, revision, **options)
        self.salt = str(options.get("salt", ""))
        self.load_count = 0

    def _load(self) -> None:
        self.load_count += 1
        self.dim = 12 if self.model_name.endswith("-b") else 8

    def _encode(self, texts: List[str], batch_size: int) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                digest = hashlib.md5(
                    f"{self.salt}{self.model_name}{word}".encode()).digest()
                out[row, digest[0] % self.dim] += 1.0 + digest[1] / 255.0
        return out


embedder_module.register_provider("fake", FakeEmbedder)


class FakeLLM:
    """Answers every column with a fixed value and counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: Dict[str, Any],
                      *, max_tokens: int = 0) -> LLMResult:
        self.calls += 1
        payload = {
            key: {"value": "42", "evidence_span": "We tested 42 voles",
                  "source_section": "methods", "confidence": 0.9}
            for key in schema["properties"]
        }
        return LLMResult(text=json.dumps(payload), input_tokens=100,
                         output_tokens=50, model="fake")

    def warmup(self) -> None:
        return None

    def describe(self) -> str:
        return "fake"


def pmc_xml(number: int) -> str:
    """A small but complete JATS article for PMC id PMC1000<number>."""
    return f"""<pmc-articleset><article article-type="research-article">
      <front><journal-meta><journal-title>Journal {number}</journal-title></journal-meta>
      <article-meta>
        <article-id pub-id-type="pmid">{2000000 + number}</article-id>
        <article-id pub-id-type="pmc">PMC{1000000 + number}</article-id>
        <title-group><article-title>Rodent survey number {number}</article-title></title-group>
        <pub-date><year>20{10 + number % 10}</year></pub-date>
        <abstract><p>We tested {40 + number} voles and found {number} positive by PCR.</p></abstract>
      </article-meta></front>
      <body>
        <sec sec-type="methods"><title>Methods</title>
          <p>Animals were trapped at {number + 3} sites in Poland and screened by RT-PCR for hantavirus RNA. {"Trapping followed a standard grid protocol over several nights. " * 30}</p></sec>
        <sec sec-type="results"><title>Results</title>
          <p>Of {40 + number} animals, {number} were positive, a prevalence that varied by site and season across the study period.</p></sec>
        <sec><title>Discussion</title><p>The sample size limits inference about seasonal trends in prevalence.</p></sec>
      </body>
      <back><notes><title>Funding</title><p>Funded by Council {number}.</p></notes></back>
    </article></pmc-articleset>"""


class FakeNCBI:
    """Stands in for Fetcher._get. Serves PMC XML and idconv answers."""

    def __init__(self) -> None:
        self.requests = 0
        self.fail = False

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeNCBI":
        ncbi = self

        def fake_get(self_fetcher, url, params=None, *, attempts=None):  # noqa: ANN001
            if not self_fetcher.allow_network:
                return None
            ncbi.requests += 1
            stats = getattr(self_fetcher.raw_cache, "stats", None)
            if stats is not None:
                stats.add("network_requests")
            if ncbi.fail:
                raise KeyboardInterrupt  # stands in for a crash mid-run
            params = params or {}
            if url == fetch.IDCONV:
                records = []
                for token in params["ids"].split(","):
                    number = int(token) - 2000000
                    records.append({"requested-id": token, "pmid": token,
                                    "pmcid": f"PMC{1000000 + number}"})
                return json.dumps({"records": records})
            if url.endswith("efetch.fcgi") and params.get("db") == "pmc":
                number = int(params["id"].replace("PMC", "")) - 1000000
                return pmc_xml(number)
            return None

        monkeypatch.setattr(fetch.Fetcher, "_get", fake_get)
        return self


@pytest.fixture
def fake_ncbi(monkeypatch: pytest.MonkeyPatch) -> FakeNCBI:
    return FakeNCBI().install(monkeypatch)


def paper_links(count: int) -> List[str]:
    """Half the papers as PMIDs (need resolving), half as PMCIDs."""
    links = []
    for number in range(1, count + 1):
        if number % 2:
            links.append(str(2000000 + number))
        else:
            links.append(f"PMC{1000000 + number}")
    return links
