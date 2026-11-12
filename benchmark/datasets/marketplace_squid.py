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
"""Adapt benchmark Marketplace reads to OpenReader and hydrate IPFS payloads.

Only the explicitly configured Polygon endpoint uses this adapter. Other
indexers keep their existing query and retry paths.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from graphql import parse, print_ast
from graphql.language import ast

# isort: split

from benchmark.datasets.marketplace_config import (
    is_squid,  # re-export for existing callers
)
from benchmark.datasets.marketplace_ipfs import fetch_payload

__all__ = [
    "is_squid",
    "squid_query",
    "hydrate",
    "graph_delivery_id",
    "squid_delivery_id",
]


def graph_delivery_id(value: str) -> str:
    """Preserve production row ids and dedup across the indexer cutover."""
    match = re.fullmatch(r"(0x[0-9a-fA-F]{64})-(0x[0-9a-fA-F]{64}|[0-9]+)", value)
    if not match:
        return value
    tx, suffix = match.groups()
    if suffix.startswith("0x"):
        return tx + suffix[2:]
    return tx + int(suffix).to_bytes(4, "little").hex()


def squid_delivery_id(value: str) -> str:
    """Resolve existing pending/shard Graph delivery ids on the squid."""
    if re.fullmatch(r"0x[0-9a-fA-F]{72}", value):
        return (
            value[:66] + "-" + str(int.from_bytes(bytes.fromhex(value[66:]), "little"))
        )
    if re.fullmatch(r"0x[0-9a-fA-F]{128}", value):
        return value[:66] + "-0x" + value[66:]
    return value


def _name(value: str) -> ast.NameNode:
    return ast.NameNode(value=value)


def _selection(source: str) -> tuple[ast.SelectionNode, ...]:
    definition = parse("{" + source + "}").definitions[0]
    if not isinstance(definition, ast.OperationDefinitionNode):
        raise ValueError("Expected a query operation")
    return definition.selection_set.selections


def squid_query(query: str) -> str:
    """Translate the benchmark's collection queries, preserving response keys."""
    document = parse(query)

    def fields(selection: ast.SelectionSetNode) -> ast.SelectionSetNode:
        out: list[ast.SelectionNode] = []
        for node in selection.selections:
            if not isinstance(node, ast.FieldNode):
                raise ValueError("Marketplace adapter supports field selections only")
            field = node.name.value
            if field == "parsedRequest":
                out.extend(
                    _selection("_squidRequest: marketplaceRequest { ipfsHashBytes } id")
                )
                continue
            if field == "parsedDeliveries":
                node.alias = node.alias or _name("parsedDeliveries")
                node.name = _name("delivers")
                node.selection_set = ast.SelectionSetNode(
                    selections=_selection(
                        "id request { id } _squidDelivery: marketplaceDelivery { ipfsHashBytes requestIdBytes }"
                    )
                )
            args = {a.name.value: a for a in node.arguments}
            direction = args.pop("orderDirection", None)
            for argument in args.values():
                if argument.name.value == "first":
                    argument.name = _name("limit")
                elif argument.name.value == "skip":
                    argument.name = _name("offset")
                elif argument.name.value == "orderBy":
                    if not isinstance(argument.value, ast.EnumValueNode):
                        raise ValueError("Marketplace orderBy must be a literal field")
                    order = argument.value.value
                    suffix = "ASC"
                    if direction is not None:
                        if not isinstance(direction.value, ast.EnumValueNode):
                            raise ValueError(
                                "Marketplace orderDirection must be a literal"
                            )
                        suffix = direction.value.value.upper()
                    # Deterministic tie order for timestamp pages.
                    values = [ast.EnumValueNode(value=f"{order}_{suffix}")]
                    if order != "id":
                        values.append(ast.EnumValueNode(value=f"id_{suffix}"))
                    argument.value = ast.ListValueNode(values=tuple(values))
            node.arguments = tuple(args.values())
            if node.selection_set:
                node.selection_set = fields(node.selection_set)
            out.append(node)
        return ast.SelectionSetNode(selections=tuple(out))

    def identifiers(node: ast.Node) -> None:
        if isinstance(node, ast.StringValueNode):
            node.value = squid_delivery_id(node.value)
        for key in getattr(node, "keys", ()):
            value = getattr(node, key, None)
            if isinstance(value, ast.Node):
                identifiers(value)
            elif isinstance(value, tuple):
                for child in value:
                    if isinstance(child, ast.Node):
                        identifiers(child)

    for definition in document.definitions:
        if not isinstance(definition, ast.OperationDefinitionNode):
            raise ValueError("Marketplace adapter expects query operations")
        definition.selection_set = fields(definition.selection_set)
    identifiers(document)
    return print_ast(document)


def hydrate(data: dict[str, Any]) -> dict[str, Any]:
    """Restore the parsed field shape consumed by existing benchmark code."""

    references: set[tuple[str, str | None]] = set()

    def collect(value: Any) -> None:
        if isinstance(value, list):
            for child in value:
                collect(child)
        elif isinstance(value, dict):
            for key, reference in value.items():
                if key in ("_squidRequest", "_squidDelivery") and reference:
                    digest = reference.get("ipfsHashBytes")
                    request_id = (
                        reference.get("requestIdBytes")
                        if key == "_squidDelivery"
                        else None
                    )
                    if digest and (key == "_squidRequest" or request_id):
                        references.add((digest, request_id))
                collect(reference)

    collect(data)
    with ThreadPoolExecutor(max_workers=8) as pool:
        keys = list(references)
        payloads = dict(zip(keys, pool.map(lambda key: fetch_payload(*key), keys)))

    def walk(value: Any) -> Any:
        if isinstance(value, list):
            return [walk(row) for row in value]
        if not isinstance(value, dict):
            return value
        row = {key: walk(child) for key, child in value.items()}
        if isinstance(row.get("id"), str):
            row["id"] = graph_delivery_id(row["id"])
        has_request = "_squidRequest" in row
        reference = row.pop("_squidRequest", None) or {}
        if has_request:
            digest = reference.get("ipfsHashBytes")
            payload = payloads.get((digest, None)) if digest else None
            row["parsedRequest"] = payload
        has_delivery = "_squidDelivery" in row
        reference = row.pop("_squidDelivery", None) or {}
        if has_delivery:
            digest = reference.get("ipfsHashBytes")
            payload = (
                payloads.get((digest, reference.get("requestIdBytes")))
                if digest
                else None
            )
            if payload:
                row.update(payload)
            else:
                row["_missing_payload"] = True
        return row

    result = walk(data)
    if "parsedDeliveries" in result:
        result["parsedDeliveries"] = [
            row for row in result["parsedDeliveries"] if not row.get("_missing_payload")
        ]
    return result
