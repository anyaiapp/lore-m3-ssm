"""CUDA Triton selective scan: parallel over (batch × channel), same math as mamba3_ref."""

from __future__ import annotations

import torch

_fwd_kernel = None
_bwd_kernel = None


def triton_scan_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_kernels() -> None:
    global _fwd_kernel, _bwd_kernel
    if _fwd_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def selective_scan_fwd_kernel(
        u_ptr,
        delta_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        y_ptr,
        h_ptr,
        batch,
        seqlen,
        dim,
        stride_b,
        stride_t,
        stride_d,
    ):
        pid = tl.program_id(0)
        b = pid // dim
        d = pid % dim
        if b >= batch:
            return
        base = b * stride_b + d * stride_d
        h = 0.0
        for t in range(seqlen):
            off = base + t * stride_t
            u = tl.load(u_ptr + off).to(tl.float32)
            delta = tl.load(delta_ptr + off).to(tl.float32)
            av = tl.load(a_ptr + off).to(tl.float32)
            bv = tl.load(b_ptr + off).to(tl.float32)
            cv = tl.load(c_ptr + off).to(tl.float32)
            dt = 1.0 / (1.0 + tl.exp(-delta))
            sig_a = 1.0 / (1.0 + tl.exp(-av))
            decay = tl.exp(-dt * sig_a)
            inp = dt * bv * u
            h = decay * h + inp
            y = cv * h
            tl.store(y_ptr + off, y)
            tl.store(h_ptr + off, h)

    @triton.jit
    def selective_scan_bwd_kernel(
        u_ptr,
        delta_ptr,
        a_ptr,
        b_ptr,
        c_ptr,
        h_ptr,
        dy_ptr,
        du_ptr,
        ddelta_ptr,
        da_ptr,
        db_ptr,
        dc_ptr,
        batch,
        seqlen,
        dim,
        stride_b,
        stride_t,
        stride_d,
    ):
        pid = tl.program_id(0)
        b = pid // dim
        d = pid % dim
        if b >= batch:
            return
        base = b * stride_b + d * stride_d
        dh_carry = 0.0
        for t in range(seqlen - 1, -1, -1):
            off = base + t * stride_t
            u = tl.load(u_ptr + off).to(tl.float32)
            delta = tl.load(delta_ptr + off).to(tl.float32)
            av = tl.load(a_ptr + off).to(tl.float32)
            bv = tl.load(b_ptr + off).to(tl.float32)
            cv = tl.load(c_ptr + off).to(tl.float32)
            dy = tl.load(dy_ptr + off).to(tl.float32)
            h_t = tl.load(h_ptr + off).to(tl.float32)
            if t > 0:
                h_prev = tl.load(h_ptr + off - stride_t).to(tl.float32)
            else:
                h_prev = 0.0
            dt = 1.0 / (1.0 + tl.exp(-delta))
            sig_a = 1.0 / (1.0 + tl.exp(-av))
            decay = tl.exp(-dt * sig_a)
            dh_t = dy * cv + dh_carry
            dc = dy * h_t
            d_inp = dh_t
            d_decay = dh_t * h_prev
            du = d_inp * dt * bv
            db = d_inp * dt * u
            d_dt = d_inp * bv * u + d_decay * (-sig_a * decay)
            d_delta = d_dt * dt * (1.0 - dt)
            d_sig_a = d_decay * (-dt * decay)
            da = d_sig_a * sig_a * (1.0 - sig_a)
            tl.store(du_ptr + off, du)
            tl.store(ddelta_ptr + off, d_delta)
            tl.store(da_ptr + off, da)
            tl.store(db_ptr + off, db)
            tl.store(dc_ptr + off, dc)
            dh_carry = dh_t * decay

    _fwd_kernel = selective_scan_fwd_kernel
    _bwd_kernel = selective_scan_bwd_kernel


def _triton_forward(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_kernels()
    u = u.contiguous()
    delta = delta.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    y = torch.empty_like(u)
    h = torch.empty_like(u)
    batch, seqlen, dim = u.shape
    grid = (batch * dim,)
    _fwd_kernel[grid](
        u,
        delta,
        a,
        b,
        c,
        y,
        h,
        batch,
        seqlen,
        dim,
        u.stride(0),
        u.stride(1),
        u.stride(2),
    )
    return y, h


def _triton_backward(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    h: torch.Tensor,
    dy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _ensure_kernels()
    dy = dy.contiguous()
    du = torch.empty_like(u)
    ddelta = torch.empty_like(delta)
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    dc = torch.empty_like(c)
    batch, seqlen, dim = u.shape
    grid = (batch * dim,)
    _bwd_kernel[grid](
        u,
        delta,
        a,
        b,
        c,
        h,
        dy,
        du,
        ddelta,
        da,
        db,
        dc,
        batch,
        seqlen,
        dim,
        u.stride(0),
        u.stride(1),
        u.stride(2),
    )
    return du, ddelta, da, db, dc


class _SelectiveScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, a, b, c):
        y, h = _triton_forward(u, delta, a, b, c)
        ctx.save_for_backward(u, delta, a, b, c, h)
        return y

    @staticmethod
    def backward(ctx, dy):
        u, delta, a, b, c, h = ctx.saved_tensors
        return _triton_backward(u, delta, a, b, c, h, dy)


def selective_scan_triton(
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    if u.device.type != "cuda":
        raise RuntimeError("selective_scan_triton requires CUDA tensors")
    return _SelectiveScanTriton.apply(u, delta, a, b, c)
