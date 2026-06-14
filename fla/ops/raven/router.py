# Copyright (c) 2023-2026
#
# Fused Raven router: in-kernel Gumbel (Philox) + top-k (sort -> k-th-largest threshold -> mask,
# no torch.topk / no scatter) + normalize + fold g = f * s_multihot + s = 1 - exp(g), in ONE
# Triton kernel. Takes a PRE-COMPUTED per-(token,head) decay scalar `f`, which the layer builds
# with whatever decay mechanism it likes -- the kernel is decay-agnostic, so the decay can change
# (Mamba2, GLA, ...) without touching the kernel. r_proj and the decay stay eager.
#
# Differentiable (FusedRouterFn): backward returns grads to the router logits (r_proj) and to f.
# The gumbel noise is additive/detached (no grad); the keep set is recomputed deterministically
# in backward from the saved logits + seed. Top-k is selected on logits (monotonic in scores).
#
# `router_reference` is the original eager chain (pure PyTorch, autograd) kept here as a readable
# spec / correctness oracle / fallback; select it via `fused_router(..., backend='reference')`. It
# matches the kernel to fp precision with gumbel=False (gumbel draws from a different RNG stream).

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fla.utils import autotune_cache_kwargs


def _router_configs():
    # one row-block of BR rows per program; grid = N/BR (N = B*T*H) is always huge, so BR=8 is
    # plenty of parallelism and the kernel is memory-bound -> only num_warps is worth tuning.
    return [triton.Config({'BR': 8}, num_warps=nw) for nw in [2, 4, 8]]


@triton.autotune(configs=_router_configs(), key=['M', 'TOPK'], **autotune_cache_kwargs)
@triton.jit
def fused_router_fwd_kernel(
    router, f, g, s,                  # router/g/s: [N, M] ; f: [N] precomputed decay scalar
    N, seed,
    M: tl.constexpr, TOPK: tl.constexpr, SOFTMAX: tl.constexpr, GUMBEL: tl.constexpr, BR: tl.constexpr,
):
    i = tl.program_id(0)
    rows = i * BR + tl.arange(0, BR)
    rmask = rows < N
    o_m = tl.arange(0, M)
    off = rows[:, None] * M + o_m[None, :]
    logits = tl.load(router + off, mask=rmask[:, None], other=-float('inf')).to(tl.float32)
    if GUMBEL:
        # router - log(Exp(1)) == router + Gumbel(0,1);  Exp(1) = -log(U), U~Uniform(0,1)
        u = tl.maximum(tl.rand(seed, off), 1e-7)
        logits = logits - tl.log(-tl.log(u))
    f_row = tl.load(f + rows, mask=rmask, other=0.0).to(tl.float32)                   # [BR] decay scalar
    # top-k on logits -> (TOPK-1)-th largest threshold -> mask, exact-topk tie-break
    srt = tl.sort(logits, 1, descending=True)
    thresh = tl.sum(tl.where(o_m[None, :] == (TOPK - 1), srt, 0.0), axis=1)
    gt = logits > thresh[:, None]
    eq = logits == thresh[:, None]
    n_gt = tl.sum(gt.to(tl.int32), axis=1)
    eq_rank = tl.cumsum(eq.to(tl.int32), axis=1)
    keep = gt | (eq & (eq_rank <= (TOPK - n_gt)[:, None]))
    if SOFTMAX:
        lo = logits - tl.max(logits, axis=1)[:, None]
        e = tl.exp(lo)
        scores = e / tl.sum(e, axis=1)[:, None]
    else:
        scores = tl.sigmoid(logits)
    masked = tl.where(keep, scores, 0.0)
    if SOFTMAX:
        smh = masked
    else:
        smh = masked / (tl.sum(masked, axis=1)[:, None] + 1e-9)
    g_out = f_row[:, None] * smh
    s_out = 1.0 - tl.exp(g_out)
    tl.store(g + off, g_out.to(g.dtype.element_ty), mask=rmask[:, None])
    tl.store(s + off, s_out.to(s.dtype.element_ty), mask=rmask[:, None])


@triton.autotune(configs=_router_configs(), key=['M', 'TOPK'], **autotune_cache_kwargs)
@triton.jit
def fused_router_bwd_kernel(
    router, f, dg, ds, drouter, df,   # router/dg/ds/drouter: [N, M] ; f/df: [N]
    N, seed,
    M: tl.constexpr, TOPK: tl.constexpr, SOFTMAX: tl.constexpr, GUMBEL: tl.constexpr, BR: tl.constexpr,
):
    i = tl.program_id(0)
    rows = i * BR + tl.arange(0, BR)
    rmask = rows < N
    o_m = tl.arange(0, M)
    off = rows[:, None] * M + o_m[None, :]
    logits = tl.load(router + off, mask=rmask[:, None], other=-float('inf')).to(tl.float32)
    if GUMBEL:
        u = tl.maximum(tl.rand(seed, off), 1e-7)
        logits = logits - tl.log(-tl.log(u))
    f_row = tl.load(f + rows, mask=rmask, other=0.0).to(tl.float32)
    dg_v = tl.load(dg + off, mask=rmask[:, None], other=0.0).to(tl.float32)
    ds_v = tl.load(ds + off, mask=rmask[:, None], other=0.0).to(tl.float32)
    srt = tl.sort(logits, 1, descending=True)
    thresh = tl.sum(tl.where(o_m[None, :] == (TOPK - 1), srt, 0.0), axis=1)
    gt = logits > thresh[:, None]
    eq = logits == thresh[:, None]
    n_gt = tl.sum(gt.to(tl.int32), axis=1)
    eq_rank = tl.cumsum(eq.to(tl.int32), axis=1)
    keep = gt | (eq & (eq_rank <= (TOPK - n_gt)[:, None]))
    if SOFTMAX:
        lo = logits - tl.max(logits, axis=1)[:, None]
        e = tl.exp(lo)
        p = e / tl.sum(e, axis=1)[:, None]
        w = tl.where(keep, p, 0.0)
    else:
        p = tl.sigmoid(logits)
        masked = tl.where(keep, p, 0.0)
        S = tl.sum(masked, axis=1)[:, None] + 1e-9
        w = masked / S
    gout = f_row[:, None] * w
    gbar = tl.where(keep, dg_v - ds_v * tl.exp(gout), 0.0)
    df_row = tl.sum(gbar * w, axis=1)                                                # dL/df  [BR]
    dw = gbar * f_row[:, None]
    if SOFTMAX:
        dp = tl.where(keep, dw, 0.0)
        bb = tl.sum(dp * p, axis=1)[:, None]
        dr = p * (dp - bb)
    else:
        aa = tl.sum(dw * w, axis=1)[:, None]
        dp = tl.where(keep, (dw - aa) / S, 0.0)
        dr = dp * p * (1.0 - p)
    tl.store(drouter + off, dr.to(drouter.dtype.element_ty), mask=rmask[:, None])
    tl.store(df + rows, df_row.to(df.dtype.element_ty), mask=rmask)


class FusedRouterFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, router, f, topk, softmax, gumbel, seed):
        shape = tuple(router.shape)
        M = shape[-1]
        N = router.numel() // M
        rflat = router.reshape(N, M)
        fflat = f.reshape(N)
        g = torch.empty_like(rflat)
        s = torch.empty_like(rflat)

        def grid(meta):
            return (triton.cdiv(N, meta['BR']),)
        fused_router_fwd_kernel[grid](rflat, fflat, g, s, N, seed, M=M, TOPK=topk, SOFTMAX=softmax, GUMBEL=gumbel)
        ctx.save_for_backward(rflat, fflat)
        ctx.topk, ctx.softmax, ctx.gumbel, ctx.seed, ctx.shape = topk, softmax, gumbel, seed, shape
        return g.reshape(shape), s.reshape(shape)

    @staticmethod
    def backward(ctx, dg, ds):
        rflat, fflat = ctx.saved_tensors
        N, M = rflat.shape
        drouter = torch.empty_like(rflat)
        df = torch.empty_like(fflat)

        def grid(meta):
            return (triton.cdiv(N, meta['BR']),)
        fused_router_bwd_kernel[grid](
            rflat, fflat, dg.reshape(N, M), ds.reshape(N, M), drouter, df,
            N, ctx.seed, M=M, TOPK=ctx.topk, SOFTMAX=ctx.softmax, GUMBEL=ctx.gumbel)
        return drouter.reshape(ctx.shape), df.reshape(ctx.shape[:-1]), None, None, None, None


def router_reference(router, f, topk, router_score='sigmoid', gumbel=False, seed=None, bias=None):
    """Pure-PyTorch reference / eager fallback for the Raven router -- the routing chain Raven used
    before the fused kernel, kept here ONCE so the layer's fallback doesn't duplicate it. Computes
    in router's native dtype, so it (a) reproduces the eager layer path byte-for-byte (bf16/fp16
    included) and (b) matches the fused kernel to fp precision when called in fp32.

      router : [..., H, M] raw r_proj logits
      f      : decay, broadcast against s_multihot [..., H, M] -- pass [..., H, 1] for a scalar
               (Mamba2) decay or [..., H, M] for a per-slot (GLA) decay
      bias   : optional, broadcast against the scores ([..., M] / [H, M]); added to the scores for
               top-k SELECTION only (the bias_rmm path) -- the kept weights still come from the
               unbiased scores
    Returns g, s [..., H, M] (router.dtype).

    gumbel adds Gumbel(0,1) to the logits: seed=None draws from the GLOBAL torch RNG (matches the
    eager layer; activation checkpointing must preserve RNG state), an int seed uses a private
    generator for reproducibility. Top-k is taken on scores (== on logits, monotonic)."""
    if gumbel:
        # router + Gumbel(0,1) == router - log(Exp(1)) == router - log(-log(U)), U ~ Uniform(0,1).
        if seed is None:
            router = router - torch.empty_like(router).exponential_().log()
        else:
            gen = torch.Generator(device=router.device).manual_seed(int(seed))
            u = torch.rand(router.shape, generator=gen, device=router.device, dtype=router.dtype).clamp_min_(1e-7)
            router = router - torch.log(-torch.log(u))
    orig_scores = torch.softmax(router, dim=-1) if router_score == 'softmax' else torch.sigmoid(router)
    scores = orig_scores + bias if bias is not None else orig_scores
    route_idx = scores.topk(topk, dim=-1).indices
    topk_weights = torch.gather(orig_scores, dim=-1, index=route_idx)
    if router_score != 'softmax':
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
    s_multihot = torch.zeros_like(router).scatter_(-1, route_idx, topk_weights.to(router.dtype))
    g = (f * s_multihot).to(router.dtype)
    s = (1 - g.exp()).to(router.dtype)
    return g, s


def fused_router(router, f, topk, router_score='sigmoid', gumbel=False, seed=0, backend='triton'):
    """Fused, differentiable Raven router: in-kernel Gumbel + top-k + normalize + fold
    (g = f*s_multihot) + s = 1-exp(g), all in one kernel.
      router : [..., H, M] raw r_proj logits (NOT gumbel-noised)
      f      : [..., H]    PRE-COMPUTED per-(token,head) decay scalar (layer builds it via any
                           decay mechanism -- the kernel is decay-agnostic)
    Returns g, s [..., H, M] in router.dtype. With gumbel=True, adds Gumbel(0,1) to the logits
    in-kernel via Philox `seed` (pass a fresh per-step seed from torch RNG so activation-checkpoint
    recompute reproduces it). Top-k selected on logits (monotonic in scores).

    backend='triton' (default) runs the fused kernel; backend='reference' runs `router_reference`
    (the eager chain) on the same scalar f -- a numerical oracle / fallback that matches the kernel
    to fp precision with gumbel=False."""
    if backend == 'reference':
        # scalar f [..., H] -> [..., H, 1] so it broadcasts in router_reference's f * s_multihot.
        return router_reference(router, f.unsqueeze(-1), topk, router_score=router_score, gumbel=gumbel, seed=seed)
    if backend != 'triton':
        raise ValueError(f"Unsupported backend `{backend}`, expected 'triton' or 'reference'.")
    return FusedRouterFn.apply(router, f, topk, router_score == 'softmax', gumbel, seed)
