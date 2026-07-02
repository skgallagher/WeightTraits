from weighttraits.training.monitor import LossMonitor, LossMonitorConfig, TrainingEvent


def test_loss_monitor_warns_when_loss_rises_from_best():
    monitor = LossMonitor(
        LossMonitorConfig(
            metric="eval_loss",
            min_delta=0.0,
            loss_increase_relative=0.05,
            loss_increase_patience=2,
        )
    )

    assert monitor.update(TrainingEvent(step=1, eval_loss=1.0)).warnings == []
    assert monitor.update(TrainingEvent(step=2, eval_loss=1.06)).warnings == []
    decision = monitor.update(TrainingEvent(step=3, eval_loss=1.08))

    assert decision.warnings
    assert "increased relative to best" in decision.warnings[0]
    assert not decision.should_stop


def test_loss_monitor_stops_after_patience_without_improvement():
    monitor = LossMonitor(
        LossMonitorConfig(metric="eval_loss", min_delta=0.01, patience=2)
    )

    monitor.update(TrainingEvent(step=1, eval_loss=1.0))
    assert not monitor.update(TrainingEvent(step=2, eval_loss=0.995)).should_stop
    decision = monitor.update(TrainingEvent(step=3, eval_loss=0.994))

    assert decision.should_stop
    assert "no eval_loss improvement" in decision.reasons[0]


def test_loss_monitor_stops_when_loss_diff_gets_small():
    monitor = LossMonitor(
        LossMonitorConfig(metric="eval_loss", plateau_window=3, plateau_min_delta=0.002)
    )

    assert not monitor.update(TrainingEvent(step=1, eval_loss=0.500)).should_stop
    assert not monitor.update(TrainingEvent(step=2, eval_loss=0.499)).should_stop
    decision = monitor.update(TrainingEvent(step=3, eval_loss=0.4985))

    assert decision.should_stop
    assert "absolute change" in decision.reasons[0]


def test_loss_monitor_improvement_resets_bad_count():
    monitor = LossMonitor(
        LossMonitorConfig(metric="eval_loss", min_delta=0.01, patience=2)
    )

    monitor.update(TrainingEvent(step=1, eval_loss=1.0))
    first_bad = monitor.update(TrainingEvent(step=2, eval_loss=0.995))
    improved = monitor.update(TrainingEvent(step=3, eval_loss=0.98))

    assert first_bad.state["bad_count"] == 1
    assert improved.state["bad_count"] == 0
    assert not improved.should_stop
