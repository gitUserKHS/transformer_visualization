import torch

from backend.app.factory_engine import FactoryManager
from backend.app.production_model import ProductionMiniLM, ProductionModelConfig


def small_model() -> ProductionMiniLM:
    return ProductionMiniLM(
        ProductionModelConfig(
            vocab_size=256,
            context_length=16,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=128,
        )
    )


def test_forward_capture_reports_activation_flow():
    model = small_model()
    ids = torch.randint(0, 256, (2, 12))
    output = model(
        ids,
        labels=ids,
        capture_layer=0,
        capture_flow=True,
    )

    assert len(output["flow"]) == 5
    assert output["flow"][0]["name"] == "Token embedding"
    assert output["flow"][-1]["name"] == "Vocabulary logits"
    assert all(row["rms"] >= 0 for row in output["flow"])
    assert output["trace"]["q_shape"] == [2, 4, 12, 16]


def test_gradient_updates_and_loss_surface_are_real_and_finite():
    torch.manual_seed(7)
    manager = FactoryManager.__new__(FactoryManager)
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.randint(0, 256, (2, 12))

    loss = model(ids, labels=ids)["loss"]
    assert loss is not None
    loss.backward()
    gradients = manager._gradient_flow(model)
    samples = manager._capture_parameter_samples(model)
    optimizer.step()
    updates = manager._parameter_updates(samples)
    before_surface = [parameter.detach().clone() for parameter in model.parameters()]
    surface = manager._loss_surface(model, ids, ids, seed=9)

    assert gradients
    assert any(item["gradient_norm"] > 0 for item in gradients)
    assert any(item["update_rms"] > 0 for item in updates.values())
    assert len(surface["losses"]) == 5
    assert all(len(row) == 5 for row in surface["losses"])
    assert torch.isfinite(torch.tensor(surface["losses"])).all()
    assert all(
        torch.allclose(before, after, atol=2e-6)
        for before, after in zip(before_surface, model.parameters(), strict=True)
    )
