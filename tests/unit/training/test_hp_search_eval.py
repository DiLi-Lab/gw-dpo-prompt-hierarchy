"""Unit tests for the HP-search reward-accuracy metric."""

import pytest
import torch

from src.training.hp_search_eval import compute_reward_accuracy_metrics


def test_all_correct_yields_perfect_macro_avg():
    chosen = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
    rejected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    ref_chosen = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    ref_rejected = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
    gaps = [0, 1, 2, 3, 4]
    m = compute_reward_accuracy_metrics(
        chosen, rejected, ref_chosen, ref_rejected, gaps, beta=0.1,
    )
    assert m["macro_avg_accuracy"] == 1.0
    for g in range(5):
        assert m["per_gap_accuracy"][g] == 1.0
        assert m["per_gap_count"][g] == 1


def test_per_gap_accuracy_respects_construction():
    chosen = torch.tensor([0.0, 0.0, 5.0, 5.0, 5.0])
    rejected = torch.tensor([5.0, 5.0, 0.0, 0.0, 0.0])
    ref_chosen = torch.zeros(5)
    ref_rejected = torch.zeros(5)
    gaps = [0, 1, 2, 3, 4]
    m = compute_reward_accuracy_metrics(
        chosen, rejected, ref_chosen, ref_rejected, gaps, beta=0.1,
    )
    assert m["per_gap_accuracy"][0] == 0.0
    assert m["per_gap_accuracy"][1] == 0.0
    assert m["per_gap_accuracy"][2] == 1.0
    assert m["per_gap_accuracy"][3] == 1.0
    assert m["per_gap_accuracy"][4] == 1.0
    assert m["macro_avg_accuracy"] == pytest.approx(3.0 / 5.0)


def test_gap_weighted_accuracy():
    chosen = torch.tensor([0.0, 0.0, 5.0, 0.0, 5.0])
    rejected = torch.tensor([5.0, 5.0, 0.0, 5.0, 0.0])
    ref_chosen = torch.zeros(5)
    ref_rejected = torch.zeros(5)
    gaps = [0, 1, 2, 3, 4]
    m = compute_reward_accuracy_metrics(
        chosen, rejected, ref_chosen, ref_rejected, gaps, beta=0.1,
    )
    # Gap-weighted = sum(gap*correct) / sum(gap*count)
    #              = (0*0 + 1*0 + 2*1 + 3*0 + 4*1) / (0+1+2+3+4) = 6/10
    assert m["gap_weighted_accuracy"] == pytest.approx(0.6)


def test_tie_counts_as_incorrect():
    chosen = torch.tensor([0.0])
    rejected = torch.tensor([0.0])
    m = compute_reward_accuracy_metrics(
        chosen, rejected, torch.zeros(1), torch.zeros(1), [2], beta=0.1,
    )
    assert m["per_gap_accuracy"][2] == 0.0
    assert m["macro_avg_accuracy"] == 0.0


def test_empty_buckets_excluded_from_macro():
    chosen = torch.tensor([5.0, 5.0])
    rejected = torch.tensor([0.0, 0.0])
    ref_c = torch.zeros(2)
    ref_r = torch.zeros(2)
    gaps = [0, 4]
    m = compute_reward_accuracy_metrics(chosen, rejected, ref_c, ref_r, gaps, beta=0.1)
    assert m["macro_avg_accuracy"] == 1.0
    assert m["per_gap_count"][0] == 1
    assert m["per_gap_count"][1] == 0
    assert m["per_gap_count"][2] == 0
    assert m["per_gap_count"][3] == 0
    assert m["per_gap_count"][4] == 1


def test_mean_reward_margin():
    chosen = torch.tensor([5.0, 3.0])
    rejected = torch.tensor([0.0, 1.0])
    ref_c = torch.zeros(2)
    ref_r = torch.zeros(2)
    m = compute_reward_accuracy_metrics(chosen, rejected, ref_c, ref_r, [0, 1], beta=0.1)
    # Margins per pair: 0.1*(5-0) = 0.5, 0.1*(3-1) = 0.2. Mean = 0.35.
    assert m["mean_reward_margin"] == pytest.approx(0.35)


from types import SimpleNamespace

from datasets import Dataset


class _FakeCollator:
    """Minimal stub that packs raw records into tensors.

    Mimics the DPO collator's output shape: chosen and rejected are
    concatenated along dim 0, so a batch of ``bs`` pairs produces a
    ``(2 * bs, seq_len)`` tensor.
    """

    def __call__(self, examples):
        bs = len(examples)
        seq_len = 4
        input_ids = torch.arange(bs * 2 * seq_len).reshape(bs * 2, seq_len)
        attention_mask = torch.ones_like(input_ids)
        completion_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
        }


class _FakeModel:
    """Fake model returning fixed logits. The actual log-probs are
    produced by the monkeypatched ``selective_log_softmax``, so the
    logit values here don't matter."""

    training = False

    def eval(self):
        return self

    def parameters(self):
        yield torch.zeros(1, device="cpu")

    def __call__(self, input_ids, attention_mask, **kwargs):
        rows, seq_len = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(rows, seq_len, 2))


def test_evaluate_reward_accuracies_end_to_end(monkeypatch):
    from src.training import hp_search_eval
    from src.training.hp_search_eval import evaluate_reward_accuracies

    # 3 pairs with known level_gap.
    records = [
        {"prompt": "a", "chosen": "c", "rejected": "r",
          "level_gap": 0, "is_calibration": True, "margin": 0.0},
        {"prompt": "b", "chosen": "c", "rejected": "r",
          "level_gap": 2, "is_calibration": False, "margin": 2.0},
        {"prompt": "c", "chosen": "c", "rejected": "r",
          "level_gap": 4, "is_calibration": False, "margin": 4.0},
    ]
    ds = Dataset.from_list(records)

    # The wrapper calls the policy model first, then the reference, for
    # each batch. Row order within a model call is
    # [chosen_0, chosen_1, chosen_2, rejected_0, rejected_1, rejected_2].
    # Policy returns chosen > rejected so all pairs are "correct".
    policy_per_row = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    ref_per_row = [0.0] * 6
    call_counter = {"n": 0}

    def fake_selective_log_softmax(logits, labels):
        # Alternate policy vs. reference. The wrapper calls selective
        # _log_softmax once per model forward (policy, then ref).
        rows, seq_len_m1 = logits.shape[0], logits.shape[1]
        values = policy_per_row if call_counter["n"] % 2 == 0 else ref_per_row
        call_counter["n"] += 1
        # Spread the target per-row logp evenly across shifted positions.
        out = torch.zeros(rows, seq_len_m1)
        for r in range(rows):
            out[r, :] = values[r] / seq_len_m1
        return out

    monkeypatch.setattr(hp_search_eval, "selective_log_softmax",
                         fake_selective_log_softmax)

    result = evaluate_reward_accuracies(
        policy_model=_FakeModel(), ref_model=_FakeModel(),
        dataset=ds, beta=0.1, collator=_FakeCollator(),
        batch_size=3, device=torch.device("cpu"),
    )
    assert result["per_gap_accuracy"][0] == 1.0
    assert result["per_gap_accuracy"][2] == 1.0
    assert result["per_gap_accuracy"][4] == 1.0
    assert result["macro_avg_accuracy"] == 1.0
    assert result["per_gap_count"][0] == 1
    assert result["per_gap_count"][2] == 1
    assert result["per_gap_count"][4] == 1


def test_evaluate_reward_accuracies_tokenizes_raw_dataset(monkeypatch):
    """A raw dataset (no prompt_ids) paired with a tokenizer-bearing
    collator must be tokenized before collation, otherwise the TRL parent
    collator raises ``KeyError: 'prompt_ids'``.
    """
    from src.training import hp_search_eval
    from src.training.hp_search_eval import evaluate_reward_accuracies

    records = [
        {"prompt": "hi ", "chosen": "yes", "rejected": "no",
         "level_gap": 2, "is_calibration": False, "margin": 2.0},
        {"prompt": "yo ", "chosen": "ok", "rejected": "bad",
         "level_gap": 4, "is_calibration": False, "margin": 4.0},
    ]
    ds = Dataset.from_list(records)

    class _CharTokenizer:
        eos_token = "#"

        def __call__(self, text=None, **_kwargs):
            return {"input_ids": [ord(c) for c in text]}

    class _AssertingCollator:
        tokenizer = _CharTokenizer()

        def __call__(self, examples):
            for ex in examples:
                assert "prompt_ids" in ex, (
                    f"collator received un-tokenized example: {list(ex.keys())}"
                )
                assert "chosen_ids" in ex and "rejected_ids" in ex
            bs = len(examples)
            seq_len = 4
            input_ids = torch.arange(bs * 2 * seq_len).reshape(bs * 2, seq_len)
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "completion_mask": torch.ones_like(input_ids),
            }

    def fake_softmax(logits, _labels):
        return torch.zeros(logits.shape[0], logits.shape[1])

    monkeypatch.setattr(hp_search_eval, "selective_log_softmax", fake_softmax)

    result = evaluate_reward_accuracies(
        policy_model=_FakeModel(), ref_model=_FakeModel(),
        dataset=ds, beta=0.1, collator=_AssertingCollator(),
        batch_size=2, device=torch.device("cpu"),
    )
    assert "macro_avg_accuracy" in result
    assert result["per_gap_count"][2] == 1
    assert result["per_gap_count"][4] == 1


def test_evaluate_reward_accuracies_respects_ordering(monkeypatch):
    """Verify partial correctness: when chosen > rejected only at gap=4,
    macro_avg should reflect that."""
    from src.training import hp_search_eval
    from src.training.hp_search_eval import evaluate_reward_accuracies

    records = [
        {"prompt": "a", "chosen": "c", "rejected": "r",
          "level_gap": 0, "is_calibration": True, "margin": 0.0},
        {"prompt": "b", "chosen": "c", "rejected": "r",
          "level_gap": 4, "is_calibration": False, "margin": 4.0},
    ]
    ds = Dataset.from_list(records)

    # Row order in one forward: [chosen_0, chosen_1, rejected_0, rejected_1].
    # We want: gap=0 incorrect (chosen <= rejected), gap=4 correct.
    policy_per_row = [0.0, 1.0, 1.0, 0.0]  # chosen_0=0 < rejected_0=1; chosen_1=1 > rejected_1=0
    ref_per_row = [0.0] * 4
    call_counter = {"n": 0}

    def fake(logits, labels):
        rows, seq_len_m1 = logits.shape[0], logits.shape[1]
        values = policy_per_row if call_counter["n"] % 2 == 0 else ref_per_row
        call_counter["n"] += 1
        out = torch.zeros(rows, seq_len_m1)
        for r in range(rows):
            out[r, :] = values[r] / seq_len_m1
        return out

    monkeypatch.setattr(hp_search_eval, "selective_log_softmax", fake)

    result = evaluate_reward_accuracies(
        policy_model=_FakeModel(), ref_model=_FakeModel(),
        dataset=ds, beta=0.1, collator=_FakeCollator(),
        batch_size=2, device=torch.device("cpu"),
    )
    assert result["per_gap_accuracy"][0] == 0.0
    assert result["per_gap_accuracy"][4] == 1.0
    # Macro over populated buckets {0, 4} = (0 + 1) / 2 = 0.5
    assert result["macro_avg_accuracy"] == 0.5
