import pytest

torch = pytest.importorskip("torch")

from post_training.evaluate_sft_loss import batched, evaluate_loss


def test_batched_splits_in_order() -> None:
    assert batched([{"i": i} for i in range(5)], 2) == [
        [{"i": 0}, {"i": 1}],
        [{"i": 2}, {"i": 3}],
        [{"i": 4}],
    ]


def test_evaluate_loss_weights_by_target_tokens() -> None:
    class Output:
        def __init__(self, loss: float) -> None:
            self.loss = torch.tensor(loss)

    class FakeModel:
        def __init__(self) -> None:
            self.losses = iter([1.0, 3.0])

        def eval(self) -> None:
            return None

        def __call__(self, **_) -> Output:
            return Output(next(self.losses))

    encoded = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([-100, 2, 3]),
        },
        {
            "input_ids": torch.tensor([4, 5, 6, 7]),
            "labels": torch.tensor([-100, 5, 6, 7]),
        },
    ]

    report = evaluate_loss(
        FakeModel(),
        encoded,
        pad_token_id=0,
        batch_size=1,
        device="cpu",
    )

    assert report["target_tokens"] == 5
    assert report["batch_losses"] == [1.0, 3.0]
    assert report["loss"] == 2.2
