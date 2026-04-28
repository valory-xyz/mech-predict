# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#   Copyright 2026 Valory AG
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ------------------------------------------------------------------------------
"""Tests for benchmark/categories.py — shared per-platform category taxonomies."""

from benchmark.categories import (
    ACTIVE_CATEGORIES,
    OMEN_CATEGORIES,
    PLATFORM_ALLOWED_CATEGORIES,
    POLYMARKET_ACTIVE_CATEGORIES,
)


class TestCategoryTaxonomies:
    """Frozen-set taxonomies mirror the upstream platforms' own lists."""

    def test_omen_categories_immutable(self) -> None:
        """OMEN_CATEGORIES is a frozenset — accidental mutation is rejected."""
        assert isinstance(OMEN_CATEGORIES, frozenset)

    def test_polymarket_categories_immutable(self) -> None:
        """POLYMARKET_ACTIVE_CATEGORIES is a frozenset."""
        assert isinstance(POLYMARKET_ACTIVE_CATEGORIES, frozenset)

    def test_active_is_union(self) -> None:
        """ACTIVE_CATEGORIES is the union — used by fleet-level filters."""
        assert ACTIVE_CATEGORIES == OMEN_CATEGORIES | POLYMARKET_ACTIVE_CATEGORIES

    def test_polymarket_is_subset_of_omen(self) -> None:
        """Every polystrat tag is also a market-creator topic.

        The trader's tag list ``POLYMARKET_CATEGORY_TAGS`` is a strict
        subset of market-creator's ``DEFAULT_TOPICS`` today; this test
        catches an accidental drift introduction (e.g. adding a
        polymarket-only tag without matching coverage on the omen side).
        """
        assert POLYMARKET_ACTIVE_CATEGORIES.issubset(OMEN_CATEGORIES)


class TestPlatformAllowedCategories:
    """``PLATFORM_ALLOWED_CATEGORIES`` exposes the platform→set mapping."""

    def test_omen_key_maps_to_omen_set(self) -> None:
        """The mapping is keyed by scorer platform string."""
        assert PLATFORM_ALLOWED_CATEGORIES["omen"] is OMEN_CATEGORIES

    def test_polymarket_key_maps_to_polymarket_set(self) -> None:
        """The mapping is keyed by scorer platform string."""
        assert PLATFORM_ALLOWED_CATEGORIES["polymarket"] is POLYMARKET_ACTIVE_CATEGORIES

    def test_only_known_platforms(self) -> None:
        """Mapping contains exactly omen and polymarket — guards typos."""
        assert set(PLATFORM_ALLOWED_CATEGORIES.keys()) == {"omen", "polymarket"}


class TestAnalyzeReExports:
    """analyze.py re-exports the constants for backward compatibility."""

    def test_omen_constant_reexported(self) -> None:
        """``analyze.OMEN_CATEGORIES`` is the same object as in ``categories``."""
        # pylint: disable=import-outside-toplevel
        from benchmark import analyze, categories

        assert analyze.OMEN_CATEGORIES is categories.OMEN_CATEGORIES

    def test_polymarket_constant_reexported(self) -> None:
        """``analyze.POLYMARKET_ACTIVE_CATEGORIES`` is the same object as in ``categories``."""
        # pylint: disable=import-outside-toplevel
        from benchmark import analyze, categories

        assert (
            analyze.POLYMARKET_ACTIVE_CATEGORIES
            is categories.POLYMARKET_ACTIVE_CATEGORIES
        )

    def test_active_constant_reexported(self) -> None:
        """``analyze.ACTIVE_CATEGORIES`` is the same object as in ``categories``."""
        # pylint: disable=import-outside-toplevel
        from benchmark import analyze, categories

        assert analyze.ACTIVE_CATEGORIES is categories.ACTIVE_CATEGORIES
