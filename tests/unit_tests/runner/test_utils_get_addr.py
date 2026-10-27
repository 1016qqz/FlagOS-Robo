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

"""get_addr() 的回环地址过滤回归测试。

迁移自上游时 `get_addr()` 带着一个继承缺陷：
`if not ip.startswith == "127.0.0.1"` 比较的是 **bound method 对象** 与 str，
恒为 False，`not False` 恒 True —— 两段「排除回环地址」的分支完全失效。
本文件把该语义钉死：无论哪条探测路径拿到 127.0.0.1，都不得作为对外地址返回。
"""

import socket

from flagos_robo.runner import utils


class _FakeSocket:
    """最小 socket 替身，connect 成功并把 getsockname 固定为给定地址。"""

    def __init__(self, addr):
        self._addr = addr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def connect(self, _target):
        return None

    def getsockname(self):
        return (self._addr, 0)


def _patch_socket(monkeypatch, udp_addr, fqdn_addr, hostname="fake-host"):
    """把 get_addr() 依赖的三条探测路径全部换成可控替身。"""

    def fake_socket(*_args, **_kwargs):
        if udp_addr is None:
            raise OSError("udp probe unavailable")
        return _FakeSocket(udp_addr)

    def fake_gethostbyname(_name):
        if fqdn_addr is None:
            raise socket.gaierror("fqdn probe unavailable")
        return fqdn_addr

    monkeypatch.setattr(utils.socket, "socket", fake_socket)
    monkeypatch.setattr(utils.socket, "gethostbyname", fake_gethostbyname)
    monkeypatch.setattr(utils.socket, "getfqdn", lambda: f"{hostname}.local")
    monkeypatch.setattr(utils.socket, "gethostname", lambda: hostname)


def test_udp_probe_returns_routable_ip(monkeypatch):
    """UDP 探测拿到可路由地址时直接返回它。"""
    _patch_socket(monkeypatch, udp_addr="10.1.2.3", fqdn_addr="127.0.0.1")
    assert utils.get_addr() == "10.1.2.3"


def test_loopback_from_udp_probe_is_rejected(monkeypatch):
    """UDP 探测只拿到回环地址时，必须继续往下找，不得返回 127.0.0.1。

    这是缺陷的直接回归点：修复前该分支恒真，会把 127.0.0.1 直接返回。
    """
    _patch_socket(monkeypatch, udp_addr="127.0.0.1", fqdn_addr="10.4.5.6")
    assert utils.get_addr() == "10.4.5.6"


def test_loopback_from_both_probes_falls_back_to_hostname(monkeypatch):
    """两条探测都只有回环地址时，回落到主机名而不是 127.0.0.1。"""
    _patch_socket(monkeypatch, udp_addr="127.0.0.1", fqdn_addr="127.0.0.1", hostname="only-host")
    assert utils.get_addr() == "only-host"


def test_loopback_variant_127_0_1_1_is_rejected(monkeypatch):
    """Debian 系常见的 127.0.1.1 同属回环段，同样不得对外返回。"""
    _patch_socket(monkeypatch, udp_addr="127.0.1.1", fqdn_addr="10.7.8.9")
    assert utils.get_addr() == "10.7.8.9"


def test_all_probes_failing_falls_back_to_hostname(monkeypatch):
    """探测全部抛错时走 except 分支，回落主机名（验证收窄后的 OSError 仍被捕获）。"""
    _patch_socket(monkeypatch, udp_addr=None, fqdn_addr=None, hostname="err-host")
    assert utils.get_addr() == "err-host"
