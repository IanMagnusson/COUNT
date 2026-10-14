"""Read COUNT n-gram tables of token histories, continuations, and scores.

Each log10 score is stored in one byte, with a scale for each n-gram order
and mask bits marking absent candidates. Lookup decodes natural-log scores
without renormalizing the retained candidates.
"""

from __future__ import annotations

import mmap
import os
import struct
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

_MAGIC = b"COUN"  # four-byte tag written by the table packer
_FORMAT_ID = 2
_SCORE_ENCODING_LINEAR_UINT8 = 1
_HEADER_SIZE = 64
_SECTION_HEADER_SIZE = 64


class _OrderSection:
    """Mmap-backed arrays for one n-gram order."""

    def __init__(
        self,
        order: int,
        prefixes_2d: np.ndarray,
        topk_tokens: np.ndarray,
        *,
        topk_score_codes: np.ndarray,
        topk_finite_mask: np.ndarray,
        score_min: float,
        score_max: float,
    ):
        self.order = int(order)
        self.plen = self.order - 1
        self.prefixes_2d = prefixes_2d
        self.topk_tokens = topk_tokens
        self.topk_score_codes = topk_score_codes
        self.topk_finite_mask = topk_finite_mask
        self.score_min = float(score_min)
        self.score_max = float(score_max)

        width = topk_score_codes.shape[1]
        self._mask_byte_indices = np.arange(width, dtype=np.intp) // 8
        self._mask_bits = np.left_shift(np.uint8(1), np.arange(width, dtype=np.uint8) % 8)

        if self.plen > 0:
            struct_dtype = np.dtype([(f"f{column}", "<u4") for column in range(self.plen)])
            self.prefixes_struct = prefixes_2d.view(struct_dtype).reshape(-1)
        else:
            self.prefixes_struct = None
        self._key_bytes: Optional[np.ndarray] = None

    def _big_endian_keys(self, histories: np.ndarray) -> np.ndarray:
        """View ``(N, plen)`` histories as fixed-width byte strings.

        Big-endian fields make byte-wise string order equal to the numeric
        field-by-field order the table is sorted in, and NumPy compares these
        strings several times faster than it compares structured records.
        """
        keys = np.ascontiguousarray(histories, dtype=np.uint32).reshape(-1, self.plen)
        keys = np.ascontiguousarray(keys.astype(">u4"))
        return keys.view(f"S{4 * self.plen}").reshape(-1)

    def sorted_key_bytes(self) -> np.ndarray:
        """Byte-string keys of this order, built once on first use."""
        if self._key_bytes is None:
            self._key_bytes = self._big_endian_keys(self.prefixes_2d)
        return self._key_bytes

    def find_rows(self, histories: np.ndarray) -> np.ndarray:
        """Return the row of each exact history in ``histories`` (N, plen), or -1.

        One binary search over the sorted history keys serves every query.
        The first call for an order copies its keys into byte-string form,
        which costs about a tenth of the table's size in memory.
        """
        count = len(histories)
        if self.plen == 0:
            return np.zeros(count, dtype=np.intp)
        assert self.prefixes_struct is not None
        if count == 0 or len(self.prefixes_struct) == 0:
            return np.full(count, -1, dtype=np.intp)
        sorted_keys = self.sorted_key_bytes()
        keys = self._big_endian_keys(histories)
        rows = np.searchsorted(sorted_keys, keys)
        clipped = np.minimum(rows, len(sorted_keys) - 1)
        return np.where(sorted_keys[clipped] == keys, clipped, -1)

    def logprobs_at(self, rows: Union[int, np.ndarray]) -> np.ndarray:
        """Return stored log10-score rows decoded as float32."""
        codes = self.topk_score_codes[rows]
        values = codes.astype(np.float64)
        values /= 255.0
        values *= self.score_max - self.score_min
        values += self.score_min
        out = values.astype(np.float32)
        packed = self.topk_finite_mask[rows]
        finite = (
            np.bitwise_and(
                packed[..., self._mask_byte_indices],
                self._mask_bits,
            )
            != 0
        )
        out[~finite] = -np.inf
        return out


class CountTable:
    """Read stored score rows by their explicit token histories."""

    def __init__(self, path: str):
        self.path = path
        self._mm: Optional[mmap.mmap] = None
        self._fd: Optional[int] = None
        self.K = 0
        self.vocab_size = 0
        self.sections_by_order: dict[int, _OrderSection] = {}

    def _ensure_open(self) -> None:
        if self._mm is not None:
            return
        fd = os.open(self.path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        self._mm = mm

        if mm[:4] != _MAGIC:
            raise ValueError(
                f"unexpected magic {bytes(mm[:4])!r} in {self.path}; expected {_MAGIC!r}"
            )
        format_id, width, num_orders, vocab_size = struct.unpack("<IIII", mm[4:20])
        if format_id != _FORMAT_ID:
            raise ValueError("unsupported COUNT n-gram table format")
        if width <= 0 or vocab_size <= 0:
            raise ValueError("COUNT table has a non-positive width or vocabulary size")
        self.K = int(width)
        self.vocab_size = int(vocab_size)

        section_header_offset = _HEADER_SIZE
        for _ in range(num_orders):
            section_header = mm[
                section_header_offset : section_header_offset + _SECTION_HEADER_SIZE
            ]
            order, _padding, num_prefixes, prefixes_offset, tokens_offset, scores_offset = (
                struct.unpack("<II QQQQ", section_header[:40])
            )
            section_header_offset += _SECTION_HEADER_SIZE
            if order <= 0:
                raise ValueError("COUNT table contains an invalid order section")
            if order in self.sections_by_order:
                raise ValueError(f"COUNT table contains duplicate order {order}")

            prefix_length = order - 1
            if prefix_length > 0:
                prefixes = np.frombuffer(
                    mm,
                    dtype=np.uint32,
                    count=num_prefixes * prefix_length,
                    offset=prefixes_offset,
                ).reshape(num_prefixes, prefix_length)
            else:
                if num_prefixes != 1:
                    raise ValueError("COUNT table order 1 must contain exactly one row")
                prefixes = np.zeros((1, 0), dtype=np.uint32)

            topk_tokens = np.frombuffer(
                mm,
                dtype=np.uint32,
                count=num_prefixes * self.K,
                offset=tokens_offset,
            ).reshape(num_prefixes, self.K)
            score_encoding, score_min, score_max, finite_mask_offset = struct.unpack(
                "<IffQ", section_header[40:60]
            )
            if score_encoding != _SCORE_ENCODING_LINEAR_UINT8:
                raise ValueError(
                    f"COUNT table order {order} uses unsupported score encoding "
                    f"{score_encoding}"
                )
            if not np.isfinite(score_min) or not np.isfinite(score_max):
                raise ValueError(f"COUNT table order {order} has non-finite score calibration")
            if score_min > score_max:
                raise ValueError(f"COUNT table order {order} has score_min greater than score_max")
            topk_score_codes = np.frombuffer(
                mm,
                dtype=np.uint8,
                count=num_prefixes * self.K,
                offset=scores_offset,
            ).reshape(num_prefixes, self.K)
            mask_width = (self.K + 7) // 8
            topk_finite_mask = np.frombuffer(
                mm,
                dtype=np.uint8,
                count=num_prefixes * mask_width,
                offset=finite_mask_offset,
            ).reshape(num_prefixes, mask_width)

            self.sections_by_order[int(order)] = _OrderSection(
                order=order,
                prefixes_2d=prefixes,
                topk_tokens=topk_tokens,
                topk_score_codes=topk_score_codes,
                topk_finite_mask=topk_finite_mask,
                score_min=score_min,
                score_max=score_max,
            )

        if 1 not in self.sections_by_order:
            raise ValueError("COUNT table is missing the required order-1 fallback row")

    def lookup_order(
        self, context: Sequence[int], order: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return the stored row for this exact history order, if present."""
        self._ensure_open()
        section = self.sections_by_order.get(order)
        if section is None or len(context) < section.plen:
            return None
        if section.plen == 0:
            row = 0
        else:
            assert section.prefixes_struct is not None
            if len(section.prefixes_struct) == 0:
                return None
            query = np.asarray(context[-section.plen :], dtype=np.uint32).reshape(1, section.plen)
            query_view = query.view(section.prefixes_struct.dtype).reshape(-1)
            row = int(np.searchsorted(section.prefixes_struct, query_view[0]))
            if row == len(section.prefixes_struct) or section.prefixes_struct[row] != query_view[0]:
                return None
        return section.topk_tokens[row], section.logprobs_at(row)

    def close(self) -> None:
        self.sections_by_order.clear()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


class CountLookup:
    """Scalar keyed lookup for COUNT's highest matching n-gram orders.

    Every position has ``num_orders`` slots. Missing slots have zero token IDs
    and ``-inf`` scores, so their contribution is zero while the model's
    averaging divisor stays fixed. Stored scores are converted from log10 to
    natural logs without renormalization or an additional backoff factor.
    ``K`` can be smaller than the stored row width; the highest-scoring
    candidates are retained. Optional table mixing combines scores for each
    exact history order before the matching orders are selected.
    """

    def __init__(
        self,
        table_dir: Union[str, os.PathLike],
        K: int = 16,
        N_max: int = 5,
        num_orders: int = 1,
        mix_table_dir: Optional[Union[str, os.PathLike]] = None,
        mix_lambda: float = 0.0,
    ):
        self.K = int(K)
        self.N_max = int(N_max)
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.N_max <= 0:
            raise ValueError("N_max must be positive")
        if (
            isinstance(num_orders, bool)
            or not isinstance(num_orders, (int, np.integer))
            or not 1 <= num_orders <= self.N_max
        ):
            raise ValueError("num_orders must be an integer in 1..N_max")
        self.num_orders = int(num_orders)
        if not 0.0 <= mix_lambda <= 1.0:
            raise ValueError("mix_lambda must be in [0, 1]")
        if mix_lambda > 0 and mix_table_dir is None:
            raise ValueError("mix_lambda > 0 requires mix_table_dir")
        self.mix_lambda = float(mix_lambda)
        self.table_path = self._table_path(table_dir)
        self._mix_path = self._table_path(mix_table_dir) if mix_table_dir is not None else None
        self._index: Optional[CountTable] = None
        self._mix_index: Optional[CountTable] = None

    @staticmethod
    def _table_path(table_dir: Union[str, os.PathLike]) -> str:
        path = Path(table_dir)
        if path.is_dir():
            table_root = path
            index_path = path / "forward_index_topk.bin"
        elif path.name == "forward_index_topk.bin":
            table_root = path.parent
            index_path = path
        else:
            raise ValueError(
                "table_dir must be a directory containing forward_index_topk.bin "
                "or that file directly"
            )
        if (table_root / "ngram_table_meta.json").exists():
            raise ValueError(
                "canonical-history metadata is not supported by the keyed table reader"
            )
        if not index_path.is_file():
            raise FileNotFoundError(f"missing COUNT table: {index_path}")
        return str(index_path)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_index"] = None
        state["_mix_index"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _open_index(self, path: str) -> CountTable:
        index = CountTable(path)
        index._ensure_open()
        if index.K < self.K:
            index.close()
            raise ValueError(f"COUNT table has K={index.K} but caller passed K={self.K}")
        if index.vocab_size <= index.K:
            index.close()
            raise ValueError(f"COUNT table has invalid vocabulary size {index.vocab_size}")
        max_order = max(index.sections_by_order)
        if max_order < self.N_max:
            index.close()
            raise ValueError(
                f"COUNT table has max order {max_order} but caller "
                f"passed N_max={self.N_max}"
            )
        return index

    def _ensure_open(self) -> None:
        if self._index is None:
            self._index = self._open_index(self.table_path)
        if self.mix_lambda > 0 and self._mix_index is None:
            assert self._mix_path is not None
            index = self._open_index(self._mix_path)
            if index.vocab_size != self._index.vocab_size:
                index.close()
                raise ValueError("mixed tables must have the same vocabulary size")
            self._mix_index = index

    def __del__(self):
        for name in ("_index", "_mix_index"):
            index = getattr(self, name, None)
            if index is not None:
                try:
                    index.close()
                except Exception:
                    pass

    def _lookup_order(self, context, order):
        assert self._index is not None
        ln10 = np.float32(np.log(10.0))
        if self.mix_lambda in (0.0, 1.0):
            index = self._index if self.mix_lambda == 0.0 else self._mix_index
            assert index is not None
            row = index.lookup_order(context, order)
            if row is None:
                return None
            tokens, scores = row
            if self.K < len(tokens):
                selected = np.argsort(-scores, kind="stable")[: self.K]
                tokens, scores = tokens[selected], scores[selected]
            return tokens, scores * ln10

        # Interpolate the two exact-order rows in score space before choosing
        # orders. A repeated token receives the sum of its scaled weights.
        assert self._mix_index is not None
        weights: dict[int, float] = {}
        for index, scale in (
            (self._index, 1 - self.mix_lambda),
            (self._mix_index, self.mix_lambda),
        ):
            row = index.lookup_order(context, order)
            if row is None:
                continue
            for token, score in zip(*row):
                if np.isfinite(score):
                    token = int(token)
                    weights[token] = weights.get(token, 0.0) + scale * float(
                        np.exp(float(score * ln10))
                    )
        weights = {token: weight for token, weight in weights.items() if weight > 0}
        selected = sorted(weights, key=lambda token: (-weights[token], token))[: self.K]
        tokens = np.zeros(self.K, dtype=np.int32)
        scores = np.full(self.K, -np.inf, dtype=np.float32)
        for slot, token in enumerate(selected):
            tokens[slot] = token
            scores[slot] = np.log(weights[token])
        return tokens, scores

    def _lookup_matrix(
        self, histories: np.ndarray, lengths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized lookup over right-aligned histories.

        ``histories`` has shape ``(P, N_max - 1)`` with each context's most
        recent token in the last column and zeros where the true history is
        shorter; ``lengths`` holds the true history length of each row. Every
        order is resolved with one batched binary search, longest order first,
        and the matching orders fill each row's slots in descending order.
        """
        assert self._index is not None
        assert self.mix_lambda in (0.0, 1.0)
        index = self._index if self.mix_lambda == 0.0 else self._mix_index
        assert index is not None
        count, width = histories.shape
        shape = (count, self.num_orders, self.K)
        token_ids = np.zeros(shape, dtype=np.int32)
        log_scores = np.full(shape, -np.inf, dtype=np.float32)
        slots = np.zeros(count, dtype=np.int64)
        ln10 = np.float32(np.log(10.0))
        for order in range(self.N_max, 0, -1):
            section = index.sections_by_order.get(order)
            if section is None:
                continue
            plen = section.plen
            candidates = np.flatnonzero((slots < self.num_orders) & (lengths >= plen))
            if not len(candidates):
                continue
            rows = section.find_rows(histories[candidates, width - plen :])
            found = rows >= 0
            if not found.any():
                continue
            candidates, rows = candidates[found], rows[found]
            tokens = section.topk_tokens[rows]
            scores = section.logprobs_at(rows)
            if self.K < tokens.shape[1]:
                selected = np.argsort(-scores, axis=1, kind="stable")[:, : self.K]
                tokens = np.take_along_axis(tokens, selected, axis=1)
                scores = np.take_along_axis(scores, selected, axis=1)
            matched = np.isfinite(scores).any(axis=1)
            if not matched.any():
                continue
            candidates = candidates[matched]
            slot = slots[candidates]
            token_ids[candidates, slot] = tokens[matched]
            log_scores[candidates, slot] = scores[matched] * ln10
            slots[candidates] += 1
        return token_ids, log_scores

    def _lookup_contexts_interpolated(
        self, contexts: Sequence[Sequence[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-context lookup used only when two tables are interpolated."""
        shape = (len(contexts), self.num_orders, self.K)
        token_ids = np.zeros(shape, dtype=np.int32)
        log_scores = np.full(shape, -np.inf, dtype=np.float32)
        max_prefix_length = self.N_max - 1
        for row, context in enumerate(contexts):
            limited_context = context[-max_prefix_length:] if max_prefix_length else ()
            slot = 0
            for order in range(min(self.N_max, len(limited_context) + 1), 0, -1):
                selected = self._lookup_order(limited_context, order)
                if selected is None or not np.isfinite(selected[1]).any():
                    continue
                token_ids[row, slot], log_scores[row, slot] = selected
                slot += 1
                if slot == self.num_orders:
                    break
        return token_ids, log_scores

    def lookup_batch(self, contexts: Sequence[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """Return IDs and natural-log scores with shape ``(P, J, K)``.

        ``P`` is the number of contexts and ``J`` is ``num_orders``.
        """
        self._ensure_open()
        assert self._index is not None
        if self.mix_lambda not in (0.0, 1.0):
            return self._lookup_contexts_interpolated(contexts)
        width = self.N_max - 1
        histories = np.zeros((len(contexts), width), dtype=np.uint32)
        lengths = np.zeros(len(contexts), dtype=np.int64)
        for row, context in enumerate(contexts):
            recent = np.asarray(context[-width:] if width else (), dtype=np.uint32)
            lengths[row] = len(recent)
            if len(recent):
                histories[row, width - len(recent) :] = recent
        return self._lookup_matrix(histories, lengths)

    def lookup_tokens(self, input_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Look up causal contexts for one sequence or a batch of sequences.

        Input shape ``(S,)`` or ``(B, S)`` produces ``(..., J, K)`` fields.
        Context at position ``i`` ends at the current input token, excludes
        future tokens, and never crosses a batch-row boundary. EOS does not
        reset the fixed-table context.
        """
        tokens = np.asarray(input_ids, dtype=np.uint32)
        if tokens.ndim not in (1, 2):
            raise ValueError("input_ids must have shape (S,) or (B, S)")
        sequences = tokens[None, :] if tokens.ndim == 1 else tokens
        batch, length = sequences.shape
        width = self.N_max - 1
        shape = (*tokens.shape, self.num_orders, self.K)
        if self.mix_lambda not in (0.0, 1.0):
            contexts = [
                sequence[max(0, position + 1 - width) : position + 1]
                for sequence in sequences
                for position in range(length)
            ]
            ids, scores = self.lookup_batch(contexts)
            return ids.reshape(shape), scores.reshape(shape)
        self._ensure_open()
        # Sliding windows over a left-padded copy give every position its
        # right-aligned history without a Python loop over positions.
        if batch * length == 0:
            histories = np.zeros((0, width), dtype=np.uint32)
            lengths = np.zeros(0, dtype=np.int64)
        elif width:
            padded = np.zeros((batch, length + width - 1), dtype=np.uint32)
            padded[:, width - 1 :] = sequences
            windows = np.lib.stride_tricks.sliding_window_view(padded, width, axis=1)
            histories = np.ascontiguousarray(windows).reshape(batch * length, width)
            lengths = np.tile(np.minimum(np.arange(length) + 1, width), batch)
        else:
            histories = np.zeros((batch * length, 0), dtype=np.uint32)
            lengths = np.zeros(batch * length, dtype=np.int64)
        ids, scores = self._lookup_matrix(histories, lengths)
        return ids.reshape(shape), scores.reshape(shape)
