"""CPU checks for conditional routing and trainable adapter behavior."""

from types import SimpleNamespace

import pytest
import torch

from cda_unet.model import BatchNormBank, CDAUNet
from cda_unet.schema import get_panel


@pytest.fixture
def model():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    panel = SimpleNamespace(
        channels=("a", "b", "c"),
        gates=(
            SimpleNamespace(name="root", parent=None, x="a", y="b"),
            SimpleNamespace(name="child1", parent="root", x="b", y="c"),
            SimpleNamespace(name="child2", parent="root", x="b", y="c"),
        ),
    )
    return CDAUNet(panel, features=(4, 8, 16, 32))


def test_odd_spatial_sizes_and_mixed_eval_preserve_order(model):
    model.eval()
    images = torch.randn(3, 1, 101, 103)
    ids = torch.tensor([2, 0, 1])
    with torch.no_grad():
        mixed = model(images, ids)
        separate = torch.cat([model(image[None], int(gate)) for image, gate in zip(images, ids)])
        scalar = model(images, 0)
        singleton = model(images, torch.tensor([0]))
    assert mixed.shape == images.shape
    torch.testing.assert_close(mixed, separate)
    torch.testing.assert_close(scalar, singleton, rtol=0, atol=0)


def test_bn_routes_encoder_by_input_group_and_decoder_by_gate(model):
    encoder = model.downs[0].conv[1]
    decoder = model.decoder_blocks[0].base.conv[1]
    assert isinstance(encoder, BatchNormBank)
    assert len(encoder.banks) == 2
    assert len(decoder.banks) == 3
    assert model.gate_input_groups.tolist() == [0, 1, 1]
    assert decoder.banks[1].weight.data_ptr() != decoder.banks[2].weight.data_ptr()
    model.train()
    model(torch.randn(2, 1, 33, 35), 1)
    assert [int(b.num_batches_tracked) for b in encoder.banks] == [0, 1]
    assert [int(b.num_batches_tracked) for b in decoder.banks] == [0, 1, 0]
    model(torch.randn(2, 1, 33, 35), 2)
    assert [int(b.num_batches_tracked) for b in encoder.banks] == [0, 2]
    assert [int(b.num_batches_tracked) for b in decoder.banks] == [0, 1, 1]


def test_adapter_starts_at_identity_but_output_projection_learns(model):
    for block, head in zip(model.decoder_blocks, model.stage_heads):
        assert torch.count_nonzero(block.up.weight) == 0
        assert torch.count_nonzero(head.weight[:block.kernel_size]) > 0
        assert torch.count_nonzero(head.weight[block.kernel_size:]) == 0
        assert torch.count_nonzero(head.bias) == 0
    model.train()
    images = torch.randn(2, 1, 33, 35)
    target = torch.randint(0, 2, images.shape).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(images, 1), target)
    loss.backward()
    for block in model.decoder_blocks:
        assert torch.isfinite(block.up.weight.grad).all()
        assert torch.count_nonzero(block.up.weight.grad) > 0
        # The zero output projection initially blocks gradients further into
        # the adapter; they become nonzero after that projection is updated.
        assert torch.count_nonzero(block.down.weight.grad) == 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.binary_cross_entropy_with_logits(model(images, 1), target).backward()
    assert any(torch.count_nonzero(block.down.weight.grad) > 0 for block in model.decoder_blocks)
    assert torch.count_nonzero(model.gate_embedding.weight.grad[1]) > 0
    assert torch.count_nonzero(model.gate_embedding.weight.grad[[0, 2]]) == 0


@pytest.mark.parametrize("gate_ids,exception", [
    (-1, ValueError),
    (3, ValueError),
    (True, TypeError),
    (torch.tensor([0.0]), TypeError),
    (torch.tensor([[0]]), ValueError),
    (torch.tensor([], dtype=torch.long), ValueError),
    (torch.tensor([0, 0, 0]), ValueError),
])
def test_invalid_gate_ids_fail_before_inference(model, gate_ids, exception):
    with pytest.raises(exception):
        model(torch.randn(2, 1, 33, 35), gate_ids)


def test_mixed_gate_training_is_rejected(model):
    with pytest.raises(ValueError, match="single gate"):
        model(torch.randn(2, 1, 33, 35), torch.tensor([0, 1]))


@pytest.mark.parametrize("panel_name,groups,gates,parameters", [
    ("mass", 5, 9, 35_978_369),
    ("flow", 6, 7, 35_978_593),
])
def test_published_architecture_size(panel_name, groups, gates, parameters):
    # Meta tensors verify the full architecture without allocating model
    # storage; forward and learning behavior are exercised above on CPU.
    with torch.device("meta"):
        model = CDAUNet(get_panel(panel_name))
    assert len(model.input_groups) == groups
    assert model.num_gates == gates
    assert sum(p.numel() for p in model.parameters()) == parameters
