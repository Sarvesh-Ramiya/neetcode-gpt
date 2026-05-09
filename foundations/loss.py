import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        shifted = z - np.max(z)
        sum = np.sum(np.exp(shifted))
        return [np.round(np.exp(z_j)/sum,4) for z_j in shifted]
