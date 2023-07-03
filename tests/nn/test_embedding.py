import torch

from matten.nn.embedding import _AtomicNumberToIndex


def test_atomic_number_to_index():
    allowed_species = [6, 1, 8]
    n2i = _AtomicNumberToIndex(allowed_species)

    atomic_numbers = torch.tensor([6, 6, 8, 1, 8])
    index = n2i(atomic_numbers)
    assert index.dtype == torch.long
    assert torch.equal(index, torch.tensor([1, 1, 2, 0, 2]))
