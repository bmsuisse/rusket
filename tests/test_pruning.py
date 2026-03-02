import pandas as pd

import rusket


class DummyPruningCallback:
    def __init__(self, prune_epoch: int = 2):
        self.prune_epoch = prune_epoch
        self.epochs_called = []

    def __call__(self, epoch: int, metric_score: float) -> bool:
        self.epochs_called.append(epoch)
        if epoch >= self.prune_epoch:
            return True
        return False


def test_bpr_pruning_callback() -> None:
    df = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 2, 2],
            "item_id": [0, 1, 1, 2, 0, 2],
        }
    )

    cb = DummyPruningCallback(prune_epoch=3)

    rusket.cross_validate(
        rusket.ItemKNN,
        df,
        user_col="user_id",
        item_col="item_id",
        param_grid={"k": [2], "iterations": [5]},
        n_folds=2,
        metrics=["precision"],
        callbacks=[cb],
        verbose=False,
    )

    # Now the callback is executed during the python loop fallback
    assert len(cb.epochs_called) > 0


def test_optuna_dummy_integration() -> None:
    import optuna

    from rusket.model_selection import OptunaPruningCallback

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=0))

    def objective(trial: optuna.Trial) -> float:
        cb = OptunaPruningCallback(trial, report_interval=1)
        # Should prune immediately as value is bad
        cb(0, 0.0)
        return 0.0

    study.optimize(objective, n_trials=1)
