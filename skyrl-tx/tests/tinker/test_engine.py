from cloudpathlib import AnyPath

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tx.tinker.engine import TinkerEngine, DEFAULT_ADAM_PARAMS
from tx.tinker.config import EngineConfig
from tx.tinker import types


class FutureStub:
    """Minimal stub with request_id (engine only reads this attribute)."""

    def __init__(self, request_id: int):
        self.request_id = request_id


def make_fwd_bwd_input(token_lists: list[list[int]]):
    samples = []
    for tokens in token_lists:
        targets = tokens[1:] + [0]
        weights = [1] * len(tokens)
        samples.append(
            {
                "model_input": {"chunks": [{"tokens": tokens}]},
                "loss_fn_inputs": {
                    "target_tokens": {"data": targets},
                    "weights": {"data": weights},
                },
            }
        )
    payload = {"forward_backward_input": {"data": samples}}
    return types.ForwardBackwardInput.model_validate(payload)


def _assert_tree_allclose(t1, t2, rtol=1e-3, atol=1e-3, min_match_pct=99.0):
    """Assert that at least min_match_pct% of elements in two trees are close."""
    leaves1 = jax.tree.leaves(t1)
    leaves2 = jax.tree.leaves(t2)
    assert len(leaves1) == len(leaves2), "Gradient trees differ in structure/leaf count"
    for a, b in zip(leaves1, leaves2):
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        # Check how many elements are close
        matches = np.isclose(a_arr, b_arr, rtol=rtol, atol=atol)
        match_pct = 100.0 * np.sum(matches) / a_arr.size
        if match_pct < min_match_pct:
            # Show statistics about mismatches
            diff = np.abs(a_arr - b_arr)
            rel_diff = np.abs((a_arr - b_arr) / (np.abs(b_arr) + 1e-10))
            failing = ~matches
            raise AssertionError(
                f"Only {match_pct:.2f}% of elements match (required: {min_match_pct}%)\n"
                f"  Max absolute diff: {np.max(diff[failing])}\n"
                f"  Max relative diff: {np.max(rel_diff[failing])}\n"
                f"  Mean of mismatches: {np.mean(diff[failing])}"
            )


def test_adapter_gradient_calculation():
    config = EngineConfig(
        base_model="Qwen/Qwen3-0.6B",
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
    )
    engine = TinkerEngine(config)

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    # Create two LoRA adapters
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

    # Adapter1 samples (fixed across both rounds)
    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    # Adapter2 samples (round 1: 2 samples; round 2: 4 samples)
    a2_input1 = make_fwd_bwd_input(
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]
    )
    reqs_round1 = [
        (FutureStub(101), adapter1_id, a1_input),
        (FutureStub(102), adapter2_id, a2_input1),
    ]

    # Process round 1 batch
    engine.process_forward_backward_batch(reqs_round1)

    grads_A1_round1 = jax.tree.map(lambda x: x.copy(), engine.accumulated_grads[adapter1_id].grad_sum)

    # Clear stored grads so we can run another fwd/bwd without optimizer update.
    engine.accumulated_grads[adapter1_id].reset()
    engine.accumulated_grads[adapter2_id].reset()

    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])
    a2_input2 = make_fwd_bwd_input([[9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])
    reqs_round2 = [
        (FutureStub(201), adapter1_id, a1_input),
        (FutureStub(202), adapter2_id, a2_input2),
    ]

    # Process round 2 batch
    engine.process_forward_backward_batch(reqs_round2)

    grads_A1_round2 = jax.tree.map(lambda x: x.copy(), engine.accumulated_grads[adapter1_id].grad_sum)

    # Compare gradients using 99% match threshold
    _assert_tree_allclose(grads_A1_round1, grads_A1_round2, rtol=1e-3, atol=1e-2, min_match_pct=99.0)


def test_micro_batch_grad_accumulation():
    """
    Verifies that fwd-bwd with micro-batching produces the same
    per-adapter mean gradients as without micro-batching.
    """
    # Build engine and two adapters.
    config = EngineConfig(
        base_model="Qwen/Qwen3-0.6B",
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
        micro_batch_size=4,
    )
    engine = TinkerEngine(config)

    adapter1_id = "adapter1"
    adapter2_id = "adapter2"

    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

    # Fused batch with 6 total examples: 2 for adapter1, 4 for adapter2.
    a1_input = make_fwd_bwd_input([[1, 2, 3, 4], [5, 6, 7, 8]])  # 2 samples
    a2_input = make_fwd_bwd_input(
        [
            [9, 10, 11, 12],
            [13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24],
        ]
    )

    reqs = [
        (FutureStub(1001), adapter1_id, a1_input),
        (FutureStub(1002), adapter2_id, a2_input),
    ]

    # Run 1: micro-batching enabled
    engine.process_forward_backward_batch(reqs)
    acc_micro_a1 = engine.accumulated_grads[adapter1_id]
    acc_micro_a2 = engine.accumulated_grads[adapter2_id]
    mean_micro_a1 = acc_micro_a1.get_mean()
    mean_micro_a2 = acc_micro_a2.get_mean()

    # Sanity check gradient sum denominators with micro-batching
    assert acc_micro_a1.denominator == 2
    assert acc_micro_a2.denominator == 4

    # Build a second engine without micro-batching
    config = EngineConfig(
        base_model="Qwen/Qwen3-0.6B",
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
        micro_batch_size=0,
    )
    engine = TinkerEngine(config)

    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter1_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )
    engine.process_single_request(
        types.RequestType.CREATE_MODEL, adapter2_id, {"lora_config": {"rank": 32, "alpha": 32}}
    )

    # Run 2: micro-batching disabled
    engine.process_forward_backward_batch(reqs)
    acc_full_a1 = engine.accumulated_grads[adapter1_id]
    acc_full_a2 = engine.accumulated_grads[adapter2_id]
    mean_full_a1 = acc_full_a1.get_mean()
    mean_full_a2 = acc_full_a2.get_mean()

    # Sanity check gradient sum denominators without micro-batching
    assert acc_full_a1.denominator == 2
    assert acc_full_a2.denominator == 4

    # Compare MEAN gradients with and without micro-batching
    _assert_tree_allclose(mean_micro_a1, mean_full_a1, rtol=1e-3, atol=5e-3)
    _assert_tree_allclose(mean_micro_a2, mean_full_a2, rtol=1e-3, atol=5e-3)


def test_process_optim_step_hyperparams_behavior():
    """Request-scoped overrides apply for the step, base hyperparameters stay unchanged, and update size shifts."""
    config = EngineConfig(
        base_model="Qwen/Qwen3-0.6B",
        checkpoints_base=AnyPath(""),
        max_lora_adapters=8,
        max_lora_rank=32,
    )

    engine = TinkerEngine(config)

    low_adapter = "adapter_low"
    default_adapter = "adapter_default"

    for model_id in (low_adapter, default_adapter):
        engine.process_single_request(
            types.RequestType.CREATE_MODEL,
            model_id,
            {"lora_config": {"rank": 32, "alpha": 32}},
        )

    tokens = [[1, 2, 3, 4], [5, 6, 7, 8]]

    def apply_step(request_id: int, model_id: str, request: types.OptimStepInput) -> float:
        engine.process_forward_backward_batch([(FutureStub(request_id), model_id, make_fwd_bwd_input(tokens))])
        params_before = nnx.to_arrays(nnx.pure(engine.lora_params))
        result = engine.process_optim_step(model_id, request)
        assert isinstance(result, types.OptimStepOutput)
        params_after = nnx.to_arrays(nnx.pure(engine.lora_params))

        delta = jax.tree.map(lambda a, b: a - b, params_after, params_before)
        return jnp.sqrt(jax.tree.reduce(lambda a, x: a + (x.astype(jnp.float32) ** 2).sum(), delta, 0.0)).item()

    tiny_request = types.OptimStepInput(
        adam_params=types.AdamParams(learning_rate=1e-8, beta1=1e-8, beta2=1e-8, eps=1e-9)
    )
    default_request = types.OptimStepInput(adam_params=types.AdamParams(learning_rate=1e-4))

    # Apply override step on the first adapter.
    tiny_norm = apply_step(1, low_adapter, tiny_request)

    # Verify the request hyperparameter overrides do not persist after the step.
    hyperparams = engine.optimizers[low_adapter].opt_state.hyperparams
    assert hyperparams["learning_rate"].value == pytest.approx(DEFAULT_ADAM_PARAMS.learning_rate, rel=2e-3)
    assert hyperparams["b1"].value == pytest.approx(DEFAULT_ADAM_PARAMS.beta1, rel=2e-3)
    assert hyperparams["b2"].value == pytest.approx(DEFAULT_ADAM_PARAMS.beta2, rel=2e-3)
    assert hyperparams["eps"].value == pytest.approx(DEFAULT_ADAM_PARAMS.eps, rel=2e-3)

    # Apply fallback/default step on the second adapter (same engine).
    default_norm = apply_step(2, default_adapter, default_request)

    # Expect a large gap in update magnitude between the two adapters.
    assert tiny_norm > 0
    assert default_norm / tiny_norm == pytest.approx(1e4, rel=5e-3)
