from __future__ import annotations

import pytest
import numpy as np

from app import NumpyInnerProductIndex, ROOT, Retriever, Settings


@pytest.fixture(scope="module")
def retriever() -> Retriever:
    return Retriever(Settings(_env_file=None), ROOT / "knowledge_base")


def test_carc_query_excludes_unrelated_lcd(retriever: Retriever) -> None:
    result = retriever.retrieve_local("CO-16 denial missing information companion remark code")
    assert result.source == "hybrid"
    assert result.confidence >= 0.70
    assert any("Claim Adjustment Reason Codes.md" in citation for citation in result.citations)
    assert not any("L33947" in citation for citation in result.citations)


def test_exact_rarc_code_routes_to_rarc_source(retriever: Retriever) -> None:
    result = retriever.retrieve_local("What does RARC N563 mean?")
    assert result.source == "hybrid"
    assert result.confidence >= 0.70
    assert result.citations == ["knowledge_base/sources/Remittance Advice Remarks Codes.md"]


def test_lcd_screening_query_prioritizes_policy_page(retriever: Retriever) -> None:
    result = retriever.retrieve_local("Does LCD L33947 cover CCTA screening in Kentucky or Ohio?")
    assert result.source == "hybrid"
    assert result.confidence >= 0.70
    assert result.citations[0].endswith("#page=7")
    assert not any("Claim Adjustment Reason Codes" in citation for citation in result.citations)


def test_numpy_vector_fallback_returns_inner_product_order() -> None:
    index = NumpyInnerProductIndex(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32"))

    scores, indices = index.search(np.asarray([[1.0, 0.0]], dtype="float32"), 3)

    assert indices.tolist() == [[0, 2, 1]]
    assert scores[0, 0] == 1.0
