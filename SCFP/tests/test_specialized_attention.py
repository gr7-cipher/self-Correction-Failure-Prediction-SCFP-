import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scfp.models.deberta import SpecializedAttention  # noqa: E402


def _outputs():
    torch.manual_seed(0)
    att = SpecializedAttention(hidden_size=16, num_attention_heads=2).eval()
    h = torch.randn(1, 6, 16)
    seg = torch.tensor([[0, 0, 1, 1, 2, 2]])   # [CLS]+prompt, R0, critique
    h2 = h.clone()
    h2[0, 4:] += 5.0                            # change only the critique
    return att(h, segment_ids=seg), att(h2, segment_ids=seg)


def test_cls_sees_the_critique():
    o1, o2 = _outputs()
    assert not torch.allclose(o1[0, 0], o2[0, 0])


def test_earlier_segments_do_not_see_later_ones():
    o1, o2 = _outputs()
    assert torch.allclose(o1[0, 1], o2[0, 1])   # prompt token
    assert torch.allclose(o1[0, 2], o2[0, 2])   # initial-response token
