# palatini_pt/gw/tensor_mode.py
# -*- coding: utf-8 -*-
"""
Tensor-mode utilities:
- cT_of_k: 主 API（scripts 會優先找它）
- waveform_overlay: 產生 GR 與本模型的時間域波形，用於 Fig.7 疊圖

說明：
- cT_of_k 單純包 quadratic_action.cT2_of_k 再開根；
- waveform_overlay 提供簡潔可視化：同一輸入訊號，模型相位用 c_T(k)
  做群速近似的微小相移；locked=True 時兩者重合。
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

from . import quadratic_action as QA


def cT_of_k(*, k: np.ndarray, config: Dict | None, locked: bool) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    cT2 = QA.cT2_of_k(k=k, config=config, locked=locked)
    # 數值安全：避免 -0 的小負誤差
    return np.sqrt(np.maximum(0.0, cT2))


def waveform_overlay(*, config: Dict | None) -> Dict[str, np.ndarray]:
    """
    回傳簡單的時間域疊圖：
        t, h_GR(t), h_model(t)
    以一個帶寬窄的正弦包絡當模板；模型相位用 <c_T> 做微小偏移。
    """
    # 時域網格
    T = float(config.get("waveform", {}).get("T", 200.0)) if config else 200.0
    dt = float(config.get("waveform", {}).get("dt", 0.1)) if config else 0.1
    t = np.arange(0.0, T, dt)

    # 頻率與包絡
    f0 = float(config.get("waveform", {}).get("f0", 0.05)) if config else 0.05
    env = np.exp(-((t - 0.6 * T) ** 2) / (0.1 * T) ** 2)

    # 取一個代表性的 k~2π f0，算 c_T
    k0 = 2.0 * np.pi * f0
    cT_unlocked = float(cT_of_k(k=np.array([k0]), config=config, locked=False)[0])
    cT_locked = 1.0

    # GR 與模型的相位
    phi_GR = 2.0 * np.pi * f0 * t
    # 用 c_T 的倒數做微小相位差（群速近似），鎖定時 Δphi=0
    phi_model = (2.0 * np.pi * f0 / cT_unlocked) * t

    h_GR = env * np.sin(phi_GR)
    h_model = env * np.sin(phi_model)

    return {"t": t, "h_GR": h_GR, "h_model": h_model}
