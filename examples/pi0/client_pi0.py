# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebSocket client for a FlagOS-Robo policy server.

``flagos-robo serve <model>`` starts :class:`WebsocketPolicyServer`
(``flagos_robo/serve/websocket_policy_server.py``), so this client speaks that
server's protocol:

1. connect; the server sends a msgpack **metadata** frame first.  It must be
   read before anything is sent, otherwise the first ``recv()`` would consume
   the metadata and it would be mistaken for the action.
2. the client sends one msgpack-encoded observation dict.
3. the server replies with a msgpack action dict carrying ``server_timing``.

Encoding uses the in-repo :mod:`flagos_robo.serve.msgpack_numpy` codec -- the
same one the server uses.  Plain msgpack cannot round-trip ``ndarray``; the
repo codec packs arrays as ``{__ndarray__, data, dtype, shape}``.

Images are sent as raw HWC ``uint8`` arrays (not base64): the server's
``validate_batch`` expects arrays, and the policy preprocessor does the real
resize/normalization.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from flagos_robo.serve import msgpack_numpy


def load_image(path: str) -> np.ndarray:
    """Read an image file into an HWC uint8 array (the server's expected form)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p.resolve()}")
    arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    if arr.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {arr.shape} from {p}")
    return np.ascontiguousarray(arr)


def load_state(state_path: str) -> np.ndarray:
    """Load a state tensor file (.pt) and return it with shape (1, state_dim)."""
    state = torch.load(state_path, map_location="cpu")
    if isinstance(state, torch.Tensor):
        state = state.numpy()
    state = np.asarray(state)
    if state.ndim == 1:
        state = state[np.newaxis, :]
    return state


def build_obs(args) -> dict:
    """Build the observation dict.

    Image keys must match ``engine_args.images_keys`` in the serve config.  The
    pi0 defaults are ``observation.images.{base_0_rgb,left_wrist_0_rgb,
    right_wrist_0_rgb}``; override with ``--image-keys`` for other layouts.
    """
    keys = [k.strip() for k in args.image_keys.split(",") if k.strip()]
    paths = [p for p in (args.img1, args.img2, args.img3) if p]
    if len(keys) != len(paths):
        raise ValueError(
            f"Got {len(paths)} image path(s) but {len(keys)} image key(s): {keys}. "
            "Pass --image-keys matching the images you provide."
        )
    obs = {
        "task": args.instruction,
        "observation.state": load_state(args.state_path),
    }
    for key, path in zip(keys, paths):
        obs[key] = load_image(path)
    return obs


async def infer_once(args) -> int:
    import websockets.asyncio.client as ws_client

    obs = build_obs(args)
    for key, value in sorted(obs.items()):
        if isinstance(value, np.ndarray):
            print(f"  obs {key:<44} shape={value.shape} dtype={value.dtype}")
        else:
            print(f"  obs {key:<44} {value!r}")

    uri = f"ws://{args.host}:{args.port}"
    print(f"-> Connecting to {uri}")
    async with ws_client.connect(
        uri, compression=None, max_size=None, open_timeout=args.timeout, ping_interval=None
    ) as websocket:
        metadata = msgpack_numpy.unpackb(await asyncio.wait_for(websocket.recv(), args.timeout))
        print(f"[v] Connected. Server metadata: {json.dumps(metadata, default=str)}")

        await websocket.send(msgpack_numpy.Packer().pack(obs))
        raw = await asyncio.wait_for(websocket.recv(), args.infer_timeout)

    if isinstance(raw, str):  # the server forwards its traceback as a text frame
        print("[Error] Server returned an error frame:")
        print(raw)
        return 1

    response = msgpack_numpy.unpackb(raw)
    action = np.asarray(response["action"])
    print(f"[v] Action shape={action.shape} dtype={action.dtype}")
    print(f"    first step: {np.round(action.reshape(-1, action.shape[-1])[0], 6).tolist()}")
    timing = response.get("server_timing")
    if timing:
        print(f"    server_timing: {json.dumps({k: float(v) for k, v in timing.items()})}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="WebSocket client for a FlagOS-Robo policy server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Server port (default: 5000)")
    parser.add_argument("--img1", required=True, help="Path to first camera RGB image")
    parser.add_argument("--img2", help="Path to second camera RGB image")
    parser.add_argument("--img3", help="Path to third camera RGB image")
    parser.add_argument(
        "--state-path",
        required=True,
        help="Path to state tensor file (.pt file) with shape (1, state_dim)",
    )
    parser.add_argument(
        "--instruction",
        default="Grab the orange and put it into the basket.",
        help="Task instruction for the robot",
    )
    parser.add_argument(
        "--image-keys",
        default=(
            "observation.images.base_0_rgb,"
            "observation.images.left_wrist_0_rgb,"
            "observation.images.right_wrist_0_rgb"
        ),
        help="Comma-separated observation image keys, matching engine_args.images_keys",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Connect/metadata timeout (s)")
    parser.add_argument("--infer-timeout", type=float, default=300.0, help="Inference timeout (s)")
    args = parser.parse_args()

    try:
        return asyncio.run(infer_once(args))
    except Exception as exc:
        print(f"[Error] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
