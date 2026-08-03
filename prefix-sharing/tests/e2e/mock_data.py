"""Mock data generator for E2E precision tests."""

import torch
import torch.nn.functional as F


def make_mock_batch(prefix_len: int = 128, seq_len: int = 384):
    """Build 2 sequences with shared *prefix_len* prefix.

    Returns the tuple ``(tokens, labels, loss_mask, attention_mask,
    position_ids)`` expected by Megatron ``get_batch()``.

    Sequence 0 is the provider (full tokens 0..L-1).
    Sequence 1 is the reuser (first P tokens shared, then unique suffix).

    Args:
        prefix_len: Number of shared prefix tokens.  Must be aligned to
            ``compress_ratio`` (128) when that path is active.
        seq_len: Total sequence length.

    Returns:
        tuple[torch.Tensor, ...] — (tokens, labels, loss_mask,
        attention_mask, position_ids), all on CPU with shape ``[2, L]``.
    """
    P = prefix_len
    L = seq_len
    B = 2

    tokens = torch.tensor([
        list(range(L)),                            # seq0: provider
        list(range(P)) + list(range(1000, 1000 + L - P)),  # seq1: reuser
    ], dtype=torch.long)

    # Standard causal LM: labels = shift-left, last position padded
    labels = tokens[:, 1:]                         # [B, L-1]
    labels = F.pad(labels, (0, 1), value=0)        # pad to [B, L]

    loss_mask = torch.ones(B, L, dtype=torch.float32)
    loss_mask[:, -1] = 0  # last token has no label

    # Causal attention mask [B, 1, L, L]
    attention_mask = torch.tril(torch.ones(B, 1, L, L, dtype=torch.bool))

    position_ids = torch.arange(L, dtype=torch.long).unsqueeze(0).expand(B, -1)

    return tokens, labels, loss_mask, attention_mask, position_ids


class MockDataIterator:
    """Iterator that yields one batch and then stops."""

    def __init__(self, *batch):
        self._batch = batch
        self._exhausted = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._exhausted:
            raise StopIteration
        self._exhausted = True
        return self._batch

    # Megatron training loop checks `hasattr(data_iterator, '__next__')`
    # before calling `next(data_iterator)`.  This iterator already has it.
