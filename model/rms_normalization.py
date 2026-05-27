import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        rms = np.mean(np.square(np.array(x)))+eps
        rms = rms**0.5
        ans = np.array(x)*np.array(gamma)
        ans = ans/rms
        return np.round(list(ans),4)

