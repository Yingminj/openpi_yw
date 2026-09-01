from types import SimpleNamespace

import numpy as np

from openpi.serving.websocket_policy_server import _RTCConnectionState


class _FakePolicy:
    def __init__(self):
        self._model = SimpleNamespace(action_horizon=5)
        self.calls = []

    def infer(self, obs, **kwargs):
        self.calls.append((obs, kwargs))
        raw = np.arange(15, dtype=np.float32).reshape(5, 3) + 100 * (len(self.calls) - 1)
        return {"actions": raw[:, :2], "raw_actions": raw}


def _request(previous_chunk_id=-1, *, reset=False, prefix_start_step=0):
    return {
        "state": np.zeros(2, dtype=np.float32),
        "_rtc": {
            "enabled": True,
            "reset": reset,
            "previous_chunk_id": previous_chunk_id,
            "prefix_start_step": prefix_start_step,
            "inference_delay_steps": 2,
            "execution_horizon": 4,
            "max_guidance_weight": 10.0,
            "prefix_attention_schedule": "EXP",
        },
    }


def test_rtc_session_guides_from_requested_raw_chunk_suffix():
    policy = _FakePolicy()
    session = _RTCConnectionState(policy)

    first = session.infer(_request(reset=True))
    second = session.infer(_request(first["rtc"]["chunk_id"], prefix_start_step=2))

    assert first["rtc"]["prefix_applied"] is False
    assert second["rtc"]["prefix_applied"] is True
    assert "_rtc" not in policy.calls[1][0]
    kwargs = policy.calls[1][1]
    assert kwargs["prev_chunk_left_over_len"] == 3
    np.testing.assert_array_equal(
        kwargs["prev_chunk_left_over"][:3],
        np.arange(15, dtype=np.float32).reshape(5, 3)[2:],
    )
    assert np.all(kwargs["prev_chunk_left_over"][3:] == 0)


def test_rtc_reset_does_not_reuse_previous_chunk():
    policy = _FakePolicy()
    session = _RTCConnectionState(policy)
    first = session.infer(_request(reset=True))

    reset = session.infer(_request(first["rtc"]["chunk_id"], reset=True))

    assert reset["rtc"]["prefix_applied"] is False
    assert "prev_chunk_left_over" not in policy.calls[-1][1]
