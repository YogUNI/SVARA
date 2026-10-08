"""Model architectures: wav2vec 2.0 SLU, CRNN baseline, heads, and decoding."""

from svara.models.crnn_baseline import CRNNBaseline
from svara.models.wav2vec_slu import Wav2VecSLU

__all__ = ["CRNNBaseline", "Wav2VecSLU"]
