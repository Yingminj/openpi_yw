import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class _RTCConnectionState:
    """Keep normalized action chunks isolated to one websocket client."""

    def __init__(self, policy: _base_policy.BasePolicy):
        self._policy = policy
        self._next_chunk_id = 0
        self._chunks: dict[int, np.ndarray] = {}

    def infer(self, request: dict) -> dict:
        obs = dict(request)
        rtc = obs.pop("_rtc", None)
        if not isinstance(rtc, dict) or not rtc.get("enabled", False):
            return self._policy.infer(obs)

        if rtc.get("reset", False):
            self._chunks.clear()

        previous_chunk_id = int(rtc.get("previous_chunk_id", -1))
        prefix_start = max(0, int(rtc.get("prefix_start_step", 0)))
        previous = self._chunks.get(previous_chunk_id)
        prefix_applied = previous is not None and prefix_start < len(previous)
        infer_kwargs = {"return_raw_actions": True}

        if prefix_applied:
            model = getattr(self._policy, "_model", None)
            action_horizon = int(getattr(model, "action_horizon", previous.shape[0]))
            leftover = previous[prefix_start : prefix_start + action_horizon]
            leftover_len = len(leftover)
            padded = np.zeros((action_horizon, previous.shape[1]), dtype=previous.dtype)
            padded[:leftover_len] = leftover

            inference_delay = max(0, min(int(rtc.get("inference_delay_steps", 0)), action_horizon))
            execution_horizon = max(
                inference_delay,
                min(int(rtc.get("execution_horizon", action_horizon)), action_horizon),
            )
            max_guidance_weight = float(rtc.get("max_guidance_weight", 10.0))
            if not np.isfinite(max_guidance_weight) or max_guidance_weight <= 0:
                raise ValueError("RTC max_guidance_weight must be finite and positive")

            infer_kwargs.update(
                {
                    "prev_chunk_left_over": padded,
                    "prev_chunk_left_over_len": leftover_len,
                    "inference_delay": inference_delay,
                    "prefix_horizon": execution_horizon,
                    "max_guidance_weight": max_guidance_weight,
                    "prefix_attention_schedule": str(rtc.get("prefix_attention_schedule", "EXP")),
                }
            )

        result = self._policy.infer(obs, **infer_kwargs)
        raw_actions = np.asarray(result.pop("raw_actions", None))
        if raw_actions.ndim != 2 or not np.isfinite(raw_actions).all():
            raise RuntimeError(f"RTC policy returned invalid normalized actions with shape {raw_actions.shape}")

        chunk_id = self._next_chunk_id
        self._next_chunk_id += 1
        self._chunks[chunk_id] = raw_actions
        while len(self._chunks) > 4:
            del self._chunks[min(self._chunks)]

        result["rtc"] = {
            "chunk_id": chunk_id,
            "previous_chunk_id": previous_chunk_id,
            "prefix_applied": prefix_applied,
            "prefix_start_step": prefix_start,
        }
        return result


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        rtc_state = _RTCConnectionState(self._policy)
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if not isinstance(obs, dict):
                    raise TypeError(f"Inference request must be a dictionary, got {type(obs).__name__}")

                infer_time = time.monotonic()
                action = rtc_state.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
