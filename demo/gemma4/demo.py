"""Gemma 4 offline demo on MPK (Blackwell / SM100 only).

Targets the text path of google/gemma-4-12B (the "unified" encoder-free
variant). Architecture highlights handled here:
  - alternating sliding_attention (window 1024, head_dim 256, 8 KV heads) and
    full_attention layers (head_dim 512, 1 shared KV head with K=V projection)
  - dual RoPE: theta 1e4 full-rotary on sliding layers; theta 1e6
    "proportional" RoPE with partial_rotary_factor 0.25 on global layers
    (expressed via cos=1/sin=0 entries in the tables — no kernel support
    needed)
  - QK-norm plus an unweighted v_norm, softmax scale 1.0
  - sandwich norms: input / post_attention / pre_feedforward /
    post_feedforward RMSNorms with the residual added after the post-norms
  - GeGLU MLP (gelu_pytorch_tanh), via the gelu_mul task
  - embeddings scaled by bf16(sqrt(hidden_size)); lm_head tied to the
    *unscaled* embedding table

Known limitations of this first version:
  - single GPU, offline mode, greedy decoding only
  - max_num_batched_tokens <= 4 (shared-memory budget of the head_dim-512
    global-attention task)
  - num_kv_shared_layers != 0 (KV-cache sharing across trailing layers) is
    not supported; the 12B config does not use it
"""

from transformers import AutoTokenizer, AutoConfig
import torch
import argparse
import json
import math
import os

DEFAULT_SAVE_DIR = os.path.join("outputs", "gemma4")
MAX_SAVE_TOKENS = 100


def grid_for_rmsnorm_linear_layer(size: int, use_cutlass_kernel: bool = True):
    # same heuristic as demo/qwen3/demo.py
    if size % 64 == 0 and not use_cutlass_kernel:
        return size // 64
    if size / 96 > 400:
        assert size % 256 == 0, f"FATAL: Linear layer size not supported: {size}"
        return size // 256
    if size % 96 == 0:
        return 96
    elif size % 64 == 0:
        return 64
    raise ValueError(f"Unsupported linear size {size}")


def load_text_state_dict(model_name: str, device: str = "cuda"):
    """Load the checkpoint's text-model weights as a flat dict, keys relative
    to the text model root (e.g. 'layers.0.self_attn.q_proj.weight')."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob

    local_dir = (
        model_name
        if os.path.isdir(model_name)
        else snapshot_download(model_name, allow_patterns=["*.safetensors*", "*.json"])
    )
    state_dict = {}
    for shard in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        with safe_open(shard, framework="pt", device=device) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    # locate the text-model prefix (varies between ForCausalLM and unified
    # multimodal checkpoints)
    probe = "embed_tokens.weight"
    prefixes = sorted({k[: -len(probe)] for k in state_dict if k.endswith(probe)},
                      key=len)
    assert prefixes, "could not find embed_tokens.weight in checkpoint"
    prefix = prefixes[0]
    print(f"Using text-model weight prefix: '{prefix}'")
    text_sd = {
        k[len(prefix):]: v.to(torch.bfloat16)
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }
    return text_sd


def make_rope_tables(head_dim, rope_theta, partial_rotary_factor, max_pos, device):
    """HF-layout cos/sin tables [max_pos, head_dim] (halves duplicated).

    Gemma 4 "proportional" RoPE: only the first
    int(partial_rotary_factor * head_dim // 2) frequencies are non-zero, with
    exponents taken over the FULL head_dim; the rest are zero, i.e. cos=1 /
    sin=0, which the rotate_half kernel applies as identity.
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32, device=device)
            / head_dim)
    )
    inv_freq = torch.cat(
        [inv_freq,
         torch.zeros(head_dim // 2 - rope_angles, dtype=torch.float32, device=device)]
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [max_pos, head_dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--model", type=str, default="google/gemma-4-12B",
                        help="Model name on Hugging Face or a local path")
    parser.add_argument("--max-num-batched-tokens", default=4, type=int)
    parser.add_argument("--max-num-batched-requests", default=1, type=int)
    parser.add_argument("--page-size", default=4096, type=int)
    parser.add_argument("--max-num-pages", default=2, type=int)
    parser.add_argument("--max-seq-length", default=512, type=int)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-dir", help="Compiler output directory")
    parser.add_argument("--trace-name", default="")
    parser.add_argument("--profiling", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument(
        "--no-use-cutlass-kernel", action="store_false",
        dest="use_cutlass_kernel", default=True)
    parser.add_argument("--save-tokens", nargs="?", const="auto", default=None)
    parser.add_argument(
        "--prompt", type=str,
        default="Give me a short introduction to large language model.")
    args = parser.parse_args()

    # the head_dim-512 global-attention task's shared-memory layout caps the
    # per-task query rows (MAX_TOKENS); see attention_sm100.cuh
    assert args.max_num_batched_tokens <= 4, (
        "Gemma 4 global attention currently supports max_num_batched_tokens <= 4")
    assert args.page_size % 32 == 0, "page_size must be a multiple of kv tiles"

    if args.save_tokens:
        if args.save_tokens == "auto":
            filename = "mpk_output.json" if args.use_mirage else "torch_output.json"
            save_path = os.path.join(DEFAULT_SAVE_DIR, filename)
        else:
            save_path = args.save_tokens
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    else:
        save_path = None

    torch.set_default_dtype(torch.bfloat16)
    device = "cuda"
    torch.cuda.set_device(0)

    full_config = AutoConfig.from_pretrained(args.model)
    config = getattr(full_config, "text_config", full_config)

    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    num_layers = config.num_hidden_layers
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    global_head_dim = getattr(config, "global_head_dim", None) or head_dim
    num_global_kv_heads = (
        getattr(config, "num_global_key_value_heads", None) or num_kv_heads)
    sliding_window = config.sliding_window
    vocab_size = config.vocab_size
    layer_types = list(config.layer_types)
    attention_k_eq_v = bool(getattr(config, "attention_k_eq_v", False))

    assert getattr(config, "num_kv_shared_layers", 0) in (0, None), (
        "KV-shared trailing layers are not supported yet")
    assert getattr(config, "final_logit_softcapping", None) in (None, 0), (
        "final_logit_softcapping is not supported yet")
    assert config.hidden_activation == "gelu_pytorch_tanh"
    assert attention_k_eq_v and num_global_kv_heads == 1, (
        "this demo wires Gemma 4 global layers as K=V with one shared KV head")
    rms_eps = config.rms_norm_eps
    assert abs(rms_eps - 1e-6) < 1e-12, (
        "the attention task hardcodes eps=1e-6 for q/k/v norms")

    rope_params = getattr(config, "rope_parameters", None) or {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    state_dict = load_text_state_dict(args.model, device=device)

    # Gemma uses <end_of_turn> to terminate chat turns
    eos_token_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    total_num_requests = args.max_num_batched_requests if args.use_mirage else 1
    tokens = torch.full((total_num_requests, args.max_seq_length), 0,
                        dtype=torch.long, device=device)
    for r in range(total_num_requests):
        tokens[r, : input_ids.shape[-1]] = input_ids[0]
    prompt_lengths = torch.full((total_num_requests,), input_ids.shape[-1],
                                dtype=torch.int, device=device)

    if not args.use_mirage:
        # PyTorch reference path: requires a transformers version with Gemma 4
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16).to(device)
        model.eval()
        max_new = args.max_new_tokens or (
            args.max_seq_length - input_ids.shape[-1])
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=max_new, do_sample=False,
                eos_token_id=None if args.ignore_eos else eos_token_id)
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        print(response)
        if save_path:
            gen = out[0, input_ids.shape[-1]:][:MAX_SAVE_TOKENS].tolist()
            with open(save_path, "w") as f:
                json.dump({"token_ids": gen, "text": response, "mode": "torch"},
                          f, indent=2)
        raise SystemExit(0)

    import mirage as mi

    target_cc = (torch.cuda.get_device_properties(0).major * 10
                 + torch.cuda.get_device_properties(0).minor)
    assert target_cc == 100, "the Gemma 4 attention tasks require SM100 (B200)"

    # ---- weight preparation -------------------------------------------------
    # embeddings are scaled by bf16(sqrt(hidden_size)) (HF downcasts the scale
    # to bf16 first); the lm_head stays tied to the unscaled table
    embed_scale = torch.tensor(math.sqrt(hidden_size), dtype=torch.bfloat16)
    raw_embed = state_dict["embed_tokens.weight"]
    assert raw_embed.shape == (vocab_size, hidden_size)
    embed_weight = (raw_embed.float() * embed_scale.float()).to(torch.bfloat16)
    lm_head_weight = state_dict.get("lm_head.weight", raw_embed).contiguous()

    max_pos = args.max_seq_length
    rope_tables = {}
    for lt in ("sliding_attention", "full_attention"):
        p = rope_params[lt]
        hd = head_dim if lt == "sliding_attention" else global_head_dim
        rope_tables[lt] = make_rope_tables(
            hd, p["rope_theta"], p.get("partial_rotary_factor", 1.0),
            max_pos, device)

    # KV caches per layer; global layers have 1 KV head of width 512
    k_caches, v_caches = [], []
    for lt in layer_types:
        if lt == "sliding_attention":
            shape = (args.max_num_pages, args.page_size, num_kv_heads, head_dim)
        else:
            shape = (args.max_num_pages, args.page_size, 1, global_head_dim)
        k_caches.append(torch.zeros(shape, dtype=torch.bfloat16, device=device))
        v_caches.append(torch.zeros(shape, dtype=torch.bfloat16, device=device))

    # ---- MPK setup ----------------------------------------------------------
    input_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                              dtype=torch.long, device=device)
    output_tokens = torch.full((args.max_num_batched_tokens, 1), 0,
                               dtype=torch.long, device=device)
    step = torch.full((total_num_requests,), 0, dtype=torch.int32, device=device)
    num_new_tokens = torch.full((total_num_requests,), 1, dtype=torch.int32,
                                device=device)

    profiler_tensor = (
        torch.zeros(3000 * 128, dtype=torch.uint64, device=device).contiguous()
        if args.profiling else None)

    num_workers, num_schedulers = mi.get_configurations_from_gpu(0)
    qo_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device=device)
    paged_kv_indptr_buffer = torch.empty(
        args.max_num_batched_requests + 1, dtype=torch.int32, device=device)
    paged_kv_indices_buffer = torch.empty(
        args.max_num_pages, dtype=torch.int32, device=device)
    paged_kv_last_page_len_buffer = torch.empty(
        args.max_num_batched_requests, dtype=torch.int32, device=device)

    mpk = mi.PersistentKernel(
        mode="offline",
        world_size=1,
        mpi_rank=0,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        num_remote_schedulers=0,
        max_seq_length=args.max_seq_length,
        max_num_batched_requests=args.max_num_batched_requests,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_pages=args.max_num_pages,
        page_size=args.page_size,
        eos_token_id=eos_token_id if not args.ignore_eos else -1,
        meta_tensors={
            "step": step,
            "tokens": tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "num_new_tokens": num_new_tokens,
            "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr_buffer,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        },
        profiler_tensor=profiler_tensor,
        trace_name=args.trace_name,
        spec_decode_config=None,
        use_cutlass_kernel=args.use_cutlass_kernel,
    )
    mbt = args.max_num_batched_tokens

    x = mpk.attach_input(torch_tensor=input_tokens, name="input_token")
    rope_inputs = {}
    for lt, short in (("sliding_attention", "sliding"), ("full_attention", "global")):
        cos_t, sin_t = rope_tables[lt]
        rope_inputs[lt] = (
            mpk.attach_input(torch_tensor=cos_t, name=f"cos_pos_embed_{short}"),
            mpk.attach_input(torch_tensor=sin_t, name=f"sin_pos_embed_{short}"),
        )

    # fused widths
    fused_qkv_sliding = (num_q_heads + 2 * num_kv_heads) * head_dim
    q_split = 4  # global layers: 4 tasks x 4 query heads sharing the KV head
    qo_per_task = num_q_heads // q_split
    fused_qkv_global = q_split * (qo_per_task + 1) * global_head_dim

    # ---- intermediate buffers (reused by every layer) -----------------------
    def buf(name, cols):
        return mpk.new_tensor(dims=(mbt, cols), dtype=mi.bfloat16, name=name,
                              io_category="cuda_tensor")

    y = buf("embed_out", hidden_size)
    rmsnorm_out = buf("rmsnorm_out", hidden_size)
    attn_in_sliding = buf("attn_in_sliding", fused_qkv_sliding)
    attn_in_global = buf("attn_in_global", fused_qkv_global)
    attn_out_sliding = buf("attn_out_sliding", num_q_heads * head_dim)
    attn_out_global = buf("attn_out_global", num_q_heads * global_head_dim)
    attn_proj_out = buf("attn_proj_out", hidden_size)
    post_norm_out = buf("post_norm_out", hidden_size)
    attn_res_out = buf("attn_res_out", hidden_size)
    mlp_mid = buf("mlp_mid", 2 * intermediate_size)
    gelu_mul_out = buf("gelu_mul_out", intermediate_size)
    mlp_down_out = buf("mlp_down_out", hidden_size)
    post_ffn_norm_out = buf("post_ffn_norm_out", hidden_size)
    mlp_res_out = buf("mlp_res_out", hidden_size)
    argmax_in = buf("argmax_in", vocab_size)
    argmax_part_value = buf("argmax_part_value", mpk.num_workers)
    argmax_part_index = mpk.new_tensor(
        dims=(mbt, mpk.num_workers), dtype=mi.int64,
        name="argmax_part_index", io_category="cuda_tensor")
    argmax_out = mpk.attach_input(torch_tensor=output_tokens, name="output_token")

    # ---- embedding ----------------------------------------------------------
    w_embed = mpk.attach_input(torch_tensor=embed_weight, name="embed_tokens")
    mpk.embed_layer(input=x, weight=w_embed, output=y,
                    grid_dim=(1, 1, 1), block_dim=(128, 1, 1), input_source=1)
    x = y

    # ---- decoder layers ------------------------------------------------------
    for i, lt in enumerate(layer_types):
        is_sliding = lt == "sliding_attention"
        pfx = f"layers.{i}."
        cos_pos_embed, sin_pos_embed = rope_inputs[lt]
        hd = head_dim if is_sliding else global_head_dim

        w_attn_norm = mpk.attach_input(
            torch_tensor=state_dict[pfx + "input_layernorm.weight"],
            name=f"layer_{i}_input_layernorm")
        w_q = mpk.attach_input(
            torch_tensor=state_dict[pfx + "self_attn.q_proj.weight"],
            name=f"layer_{i}_q_proj")
        w_k = mpk.attach_input(
            torch_tensor=state_dict[pfx + "self_attn.k_proj.weight"],
            name=f"layer_{i}_k_proj")
        if is_sliding:
            w_v = mpk.attach_input(
                torch_tensor=state_dict[pfx + "self_attn.v_proj.weight"],
                name=f"layer_{i}_v_proj")
            # [q-group, k, v] interleaved per KV head
            w_qkv = mpk.shuffle_tensors(
                inputs=[w_q, w_k, w_v], shuffled_dim=0,
                num_groups=num_kv_heads, name=f"layer_{i}_qkv_proj")
        else:
            # K=V: no v_proj. Replicate the single 512-wide K head once per
            # Q-head group so each attention task's chunk is [4 q heads | K]
            assert pfx + "self_attn.v_proj.weight" not in state_dict
            k_replicated = torch.cat(
                [state_dict[pfx + "self_attn.k_proj.weight"]] * q_split, dim=0)
            w_k_rep = mpk.attach_input(
                torch_tensor=k_replicated, name=f"layer_{i}_k_proj_replicated")
            w_qkv = mpk.shuffle_tensors(
                inputs=[w_q, w_k_rep], shuffled_dim=0,
                num_groups=q_split, name=f"layer_{i}_qkv_proj")
        attn_in = attn_in_sliding if is_sliding else attn_in_global
        attn_out = attn_out_sliding if is_sliding else attn_out_global

        mpk.rmsnorm_layer(input=x, weight=w_attn_norm, output=rmsnorm_out,
                          grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        mpk.linear_layer(
            input=rmsnorm_out, weight=w_qkv, output=attn_in,
            grid_dim=(grid_for_rmsnorm_linear_layer(
                w_qkv.dim(0), args.use_cutlass_kernel), 1, 1),
            block_dim=(128, 1, 1))

        w_q_norm = mpk.attach_input(
            torch_tensor=state_dict[pfx + "self_attn.q_norm.weight"],
            name=f"layer_{i}_q_norm")
        w_k_norm = mpk.attach_input(
            torch_tensor=state_dict[pfx + "self_attn.k_norm.weight"],
            name=f"layer_{i}_k_norm")
        k_cache = mpk.attach_input(
            torch_tensor=k_caches[i], name=f"layer_{i}_k_cache")
        v_cache = mpk.attach_input(
            torch_tensor=v_caches[i], name=f"layer_{i}_v_cache")

        if is_sliding:
            mpk.gemma4_paged_attention_layer(
                input=attn_in, k_cache=k_cache, v_cache=v_cache,
                q_norm=w_q_norm, k_norm=w_k_norm,
                cos_pos_embed=cos_pos_embed, sin_pos_embed=sin_pos_embed,
                output=attn_out,
                grid_dim=(mpk.max_num_batched_requests, num_kv_heads, 1),
                block_dim=(128, 1, 1),
                sliding_window=sliding_window,
                k_eq_v=False, q_split=1, kv_tile_size=32)
        else:
            mpk.gemma4_paged_attention_layer(
                input=attn_in, k_cache=k_cache, v_cache=v_cache,
                q_norm=w_q_norm, k_norm=w_k_norm,
                cos_pos_embed=cos_pos_embed, sin_pos_embed=sin_pos_embed,
                output=attn_out,
                grid_dim=(mpk.max_num_batched_requests, q_split, 1),
                block_dim=(128, 1, 1),
                sliding_window=0,
                k_eq_v=True, q_split=q_split, kv_tile_size=16)

        # o_proj (no fused residual: Gemma adds the residual after the
        # post-attention norm)
        w_o = mpk.attach_input(
            torch_tensor=state_dict[pfx + "self_attn.o_proj.weight"],
            name=f"layer_{i}_o_proj")
        mpk.linear_layer(
            input=attn_out, weight=w_o, output=attn_proj_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(
                hidden_size, args.use_cutlass_kernel), 1, 1),
            block_dim=(128, 1, 1))
        w_post_norm = mpk.attach_input(
            torch_tensor=state_dict[pfx + "post_attention_layernorm.weight"],
            name=f"layer_{i}_post_attention_layernorm")
        mpk.rmsnorm_layer(input=attn_proj_out, weight=w_post_norm,
                          output=post_norm_out,
                          grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        mpk.elementwise_add_layer(
            input_a=post_norm_out, input_b=x, output=attn_res_out,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        x = attn_res_out

        # ---- MLP (GeGLU) with pre/post feedforward norms --------------------
        w_pre_ffn_norm = mpk.attach_input(
            torch_tensor=state_dict[pfx + "pre_feedforward_layernorm.weight"],
            name=f"layer_{i}_pre_feedforward_layernorm")
        w_gate = mpk.attach_input(
            torch_tensor=state_dict[pfx + "mlp.gate_proj.weight"],
            name=f"layer_{i}_gate_proj")
        w_up = mpk.attach_input(
            torch_tensor=state_dict[pfx + "mlp.up_proj.weight"],
            name=f"layer_{i}_up_proj")
        gateup_tasks = grid_for_rmsnorm_linear_layer(
            w_gate.dim(0) + w_up.dim(0), args.use_cutlass_kernel)
        w_gateup = mpk.shuffle_tensors(
            inputs=[w_gate, w_up], shuffled_dim=0,
            num_groups=gateup_tasks // 2, name=f"layer_{i}_gateup_proj")
        mpk.rmsnorm_layer(input=x, weight=w_pre_ffn_norm, output=rmsnorm_out,
                          grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        mpk.linear_layer(input=rmsnorm_out, weight=w_gateup, output=mlp_mid,
                         grid_dim=(gateup_tasks, 1, 1), block_dim=(128, 1, 1))
        mpk.gelu_mul_layer(input=mlp_mid, output=gelu_mul_out,
                           grid_dim=(gateup_tasks // 2, 1, 1),
                           block_dim=(128, 1, 1))
        w_down = mpk.attach_input(
            torch_tensor=state_dict[pfx + "mlp.down_proj.weight"],
            name=f"layer_{i}_down_proj")
        mpk.linear_layer(
            input=gelu_mul_out, weight=w_down, output=mlp_down_out,
            grid_dim=(grid_for_rmsnorm_linear_layer(
                hidden_size, args.use_cutlass_kernel), 1, 1),
            block_dim=(128, 1, 1))
        w_post_ffn_norm = mpk.attach_input(
            torch_tensor=state_dict[pfx + "post_feedforward_layernorm.weight"],
            name=f"layer_{i}_post_feedforward_layernorm")
        mpk.rmsnorm_layer(input=mlp_down_out, weight=w_post_ffn_norm,
                          output=post_ffn_norm_out,
                          grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        mpk.elementwise_add_layer(
            input_a=post_ffn_norm_out, input_b=x, output=mlp_res_out,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
        x = mlp_res_out

    # ---- final norm + lm head + argmax --------------------------------------
    w_norm = mpk.attach_input(
        torch_tensor=state_dict["norm.weight"], name="model_norm_weight")
    w_lm_head = mpk.attach_input(torch_tensor=lm_head_weight, name="lm_head")
    mpk.rmsnorm_layer(input=x, weight=w_norm, output=rmsnorm_out,
                      grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1))
    mpk.linear_layer(input=rmsnorm_out, weight=w_lm_head, output=argmax_in,
                     grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_partial_layer(
        input=argmax_in, output=(argmax_part_value, argmax_part_index),
        grid_dim=(mpk.num_workers, 1, 1), block_dim=(128, 1, 1))
    mpk.argmax_reduce_layer(
        input=(argmax_part_value, argmax_part_index), output=argmax_out,
        grid_dim=(1, 1, 1), block_dim=(128, 1, 1))

    results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
    with open("task_graph_0.json", "w") as f:
        f.write(results["json_file"])
    with open("kernel_0.cu", "w") as f:
        f.write(results["cuda_code"])

    mpk.compile(output_dir=args.output_dir)

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    mpk()
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)

    for r in range(total_num_requests):
        generated_ids = tokens[r, : step[r] + 1]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(response)

    tokens_generated = step.max().item() + 1 - prompt_lengths[0].item()
    per_tok_ms = run_time / max(prompt_lengths[0].item() + tokens_generated, 1)
    print("Prompt length {}, generate length {}, per-token latency: {:.3f} ms"
          .format(prompt_lengths[0].item(), tokens_generated, per_tok_ms))

    if save_path:
        end_idx = step[0].item() + 1
        prompt_len = prompt_lengths[0].item()
        slice_end = min(end_idx, prompt_len + MAX_SAVE_TOKENS)
        out = {
            "token_ids": tokens[0, prompt_len:slice_end].tolist(),
            "text": tokenizer.decode(tokens[0, :end_idx],
                                     skip_special_tokens=True),
            "latency_ms_per_token": per_tok_ms,
            "prompt_length": prompt_len,
            "generate_length": max(0, end_idx - prompt_len),
            "mode": "mpk",
        }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Saved tokens to {save_path}")
