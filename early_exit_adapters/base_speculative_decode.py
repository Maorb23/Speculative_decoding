import time

import numpy as np
import torch
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM


def load_truncated_qwen_draft_model(
    model_name,
    layer_index,
    device,
    dtype=None,
    token=None,
):
    """
    Load a Qwen/Llama-style causal LM and keep only layers 0..layer_index.

    This is the important speed path: the draft model runs only the early layers,
    while the target model remains the full model used for verification.
    """
    if dtype is None:
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

    draft_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        token=token,
        trust_remote_code=True,
    ).to(device)

    num_layers_to_keep = layer_index + 1
    draft_model.model.layers = torch.nn.ModuleList(
        list(draft_model.model.layers[:num_layers_to_keep])
    )

    draft_model.config.num_hidden_layers = num_layers_to_keep
    draft_model.model.config.num_hidden_layers = num_layers_to_keep

    draft_model.eval()
    for param in draft_model.parameters():
        param.requires_grad = False

    return draft_model


@torch.no_grad()
def draft_next_logits_from_truncated_model(
    target_model,
    draft_model,
    input_ids,
    adapter=None,
):
    """
    Draft next-token logits with a truncated model.

    draft_model computes the early hidden state cheaply:
        prefix -> layers 0..k -> early_h

    target_model supplies the shared final norm and lm_head:
        early_h -> optional adapter -> target final_norm -> target lm_head
    """
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    outputs = draft_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    next_h = outputs.hidden_states[-1][:, -1:, :]

    if adapter is not None:
        adapter_dtype = next(adapter.parameters()).dtype
        next_h = next_h.to(dtype=adapter_dtype)
        next_h = adapter(next_h)

    target_dtype = next(target_model.parameters()).dtype
    next_h = next_h.to(dtype=target_dtype)

    next_h = target_model.model.norm(next_h)
    logits = target_model.lm_head(next_h)[:, -1, :]

    return logits


@torch.no_grad()
def draft_next_logits_from_layer(
    model,
    input_ids,
    layer_index,
    adapter=None,
):
    """
    Debug/reference draft path using the full model.

    This is correct for quality checks, but it is not expected to be fast because
    it runs the full target model with output_hidden_states=True.
    """
    attention_mask = torch.ones_like(input_ids, device=input_ids.device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )

    next_h = outputs.hidden_states[layer_index + 1][:, -1:, :]

    if adapter is not None:
        adapter_dtype = next(adapter.parameters()).dtype
        next_h = next_h.to(dtype=adapter_dtype)
        next_h = adapter(next_h)

    model_dtype = next(model.parameters()).dtype
    next_h = next_h.to(dtype=model_dtype)

    next_h = model.model.norm(next_h)
    logits = model.lm_head(next_h)[:, -1, :]

    return logits


@torch.no_grad()
def generate_draft_tokens(
    target_model,
    input_ids,
    gamma=6,
    adapter=None,
    temperature=1.0,
    draft_model=None,
    layer_index=None,
):
    """
    Autoregressively generate gamma draft tokens.

    Fast path:
        draft_model is a truncated model and target_model is the full model.

    Fallback path:
        draft_model is None, so target_model is used with output_hidden_states.
    """
    generated = input_ids.clone()
    scores = []

    for _ in range(gamma):
        if draft_model is None:
            if layer_index is None:
                raise ValueError("layer_index is required when draft_model is None")
            logits = draft_next_logits_from_layer(
                model=target_model,
                input_ids=generated,
                layer_index=layer_index,
                adapter=adapter,
            )
        else:
            logits = draft_next_logits_from_truncated_model(
                target_model=target_model,
                draft_model=draft_model,
                input_ids=generated,
                adapter=adapter,
            )

        probs = torch.softmax(logits.float() / temperature, dim=-1)
        next_token = torch.multinomial(probs[0], num_samples=1)

        scores.append(logits)
        generated = torch.cat([generated, next_token.view(1, 1)], dim=-1)

    return {
        "sequences": generated,
        "scores": scores,
    }


@torch.no_grad()
def generate_draft_tokens_from_layer(
    model,
    input_ids,
    layer_index,
    gamma=6,
    adapter=None,
    temperature=1.0,
):
    """Backward-compatible wrapper around the full-model reference path."""
    return generate_draft_tokens(
        target_model=model,
        input_ids=input_ids,
        layer_index=layer_index,
        gamma=gamma,
        adapter=adapter,
        temperature=temperature,
        draft_model=None,
    )


@torch.no_grad()
def run_speculative_eval(
    model,
    tokenizer,
    ds,
    layer_index,
    gamma=6,
    seqlen=10,
    num_prefix_tokens=10,
    num_of_examples=100,
    adapter=None,
    draft_model=None,
    draft_temperature=1.0,
    device_type="cuda",
    print_every=1,
):
    """
    Run baseline or adapted speculative decoding.

    Args:
        model: Full target model used for verification and vanilla timing.
        draft_model: Optional truncated draft model. Pass this for speed.
        layer_index: Early layer index. Used by the fallback full-model draft
            path and included in the result metadata.
        adapter: Optional adapter for the same layer as draft_model/layer_index.
    """

    def sync():
        if str(device_type).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def now():
        sync()
        return time.time()

    n_accepted = 0
    n_generated = 0

    speculative_times = []
    vanilla_times = []
    draft_times = []
    verify_times = []
    sample_times = []

    example_count = 0
    used_truncated_draft = draft_model is not None

    transformers.set_seed(42)

    for _, example in tqdm(enumerate(ds), total=num_of_examples):
        if example_count >= num_of_examples:
            break

        input_text = example["text"]

        input_ids = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=max(num_prefix_tokens, 16),
        )["input_ids"].to(device_type)[0][:num_prefix_tokens]

        if input_ids.numel() < num_prefix_tokens:
            continue

        input_ids = input_ids[None]
        all_ids = input_ids.clone()

        spec_start = now()

        while all_ids.shape[1] - input_ids.shape[1] < seqlen:
            remaining = seqlen - (all_ids.shape[1] - input_ids.shape[1])
            step_gamma = min(gamma, remaining)

            draft_start = now()

            draft_output = generate_draft_tokens(
                target_model=model,
                draft_model=draft_model,
                input_ids=all_ids,
                layer_index=layer_index,
                gamma=step_gamma,
                adapter=adapter,
                temperature=draft_temperature,
            )

            draft_times.append(now() - draft_start)

            draft_sequences = draft_output["sequences"]
            draft_tokens = draft_sequences[0, -step_gamma:]

            q_logits = torch.stack(draft_output["scores"], dim=1)[0]
            q = torch.softmax(q_logits.float(), dim=-1)

            q_of_generated = q[
                torch.arange(step_gamma, device=device_type),
                draft_tokens,
            ]

            verify_start = now()

            attention_mask = torch.ones_like(
                draft_sequences,
                device=device_type,
            )

            p_logits = model(
                draft_sequences,
                attention_mask=attention_mask,
                return_dict=True,
            ).logits

            verify_times.append(now() - verify_start)

            sample_start = now()

            p = torch.softmax(
                p_logits[:, -step_gamma - 1:-1, :].float(),
                dim=-1,
            )[0]

            p_of_generated = p[
                torch.arange(step_gamma, device=device_type),
                draft_tokens,
            ]

            ratio = p_of_generated / torch.clamp(q_of_generated, min=1e-12)
            ratio = torch.clamp(ratio, max=1.0)

            is_accepted = torch.rand(step_gamma, device=device_type) < ratio

            index_to_reject = torch.argmin(
                torch.cat(
                    [
                        is_accepted,
                        torch.tensor([False], device=device_type),
                    ]
                ).int()
            ).item()

            accepted_tokens = draft_tokens[:index_to_reject]

            if index_to_reject == step_gamma:
                p_for_sample = p[-1]
            else:
                p_for_sample = p[index_to_reject] - q[index_to_reject]
                p_for_sample = torch.clamp(p_for_sample, min=0)
                p_for_sample = p_for_sample / torch.clamp(
                    p_for_sample.sum(),
                    min=1e-12,
                )

            n_accepted += index_to_reject
            n_generated += index_to_reject

            if index_to_reject < step_gamma:
                n_generated += 1

            big_token = torch.multinomial(p_for_sample, num_samples=1)

            new_tokens = torch.cat([accepted_tokens, big_token], dim=0).view(1, -1)
            all_ids = torch.cat([all_ids, new_tokens], dim=-1)

            sample_times.append(now() - sample_start)

        speculative_times.append(now() - spec_start)

        vanilla_start = now()

        attention_mask = torch.ones_like(input_ids, device=device_type)

        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=all_ids.shape[1] - input_ids.shape[1],
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )

        vanilla_times.append(now() - vanilla_start)

        example_count += 1

        alpha = n_accepted / max(n_generated, 1)
        mean_spec = np.mean(speculative_times)
        mean_vanilla = np.mean(vanilla_times)
        improvement = 1 - (mean_spec / mean_vanilla)

        if example_count % print_every == 0:
            print(
                f"idx={example_count - 1:03d} "
                f"alpha={alpha:.2%} "
                f"improvement={improvement:.4f} "
                f"spec={mean_spec:.4f}s "
                f"vanilla={mean_vanilla:.4f}s "
                f"draft={np.mean(draft_times):.4f}s "
                f"verify={np.mean(verify_times):.4f}s "
                f"sample={np.mean(sample_times):.4f}s"
            )

    alpha = n_accepted / max(n_generated, 1)
    speed_improvement = 1 - (np.mean(speculative_times) / np.mean(vanilla_times))

    return {
        "layer_index": layer_index,
        "gamma": gamma,
        "seqlen": seqlen,
        "num_prefix_tokens": num_prefix_tokens,
        "num_examples": example_count,
        "used_truncated_draft": used_truncated_draft,
        "alpha": float(alpha),
        "mean_speculative_time": float(np.mean(speculative_times)),
        "mean_vanilla_time": float(np.mean(vanilla_times)),
        "mean_draft_time": float(np.mean(draft_times)),
        "mean_verify_time": float(np.mean(verify_times)),
        "mean_sample_time": float(np.mean(sample_times)),
        "speed_improvement": float(speed_improvement),
        "n_accepted": int(n_accepted),
        "n_generated": int(n_generated),
    }
