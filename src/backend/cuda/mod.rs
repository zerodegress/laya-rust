mod kernels;

use crate::arch::*;
use crate::engine::{BackendId, DType, Engine, EngineInfo, ModelSpec, Out, Req};
use crate::systemone::Calibration;
use crate::weights::Tensors;
use kernels::SRC;

use cudarc::cublas::sys::cublasOperation_t as Op;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, StridedBatchedConfig};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::driver::sys;
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use std::collections::HashMap;
use std::sync::Arc;

struct Bufs {
    x: CudaSlice<f32>,
    t: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    ctx: CudaSlice<f32>,
    p: CudaSlice<f32>,
    mlp: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    cos_a: CudaSlice<f32>,
    sin_a: CudaSlice<f32>,
    cos_b: CudaSlice<f32>,
    sin_b: CudaSlice<f32>,
    mask_g: CudaSlice<f32>,
    mask_l: CudaSlice<f32>,
    mrow: CudaSlice<f32>,
    sc: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    feats: CudaSlice<f32>,
    aact: CudaSlice<f32>,
    act: CudaSlice<f32>,
    ids: CudaSlice<i64>,
    att: CudaSlice<i64>,
    mpos: CudaSlice<i64>,
    mmask: CudaSlice<i64>,
    qtype: CudaSlice<i64>,
}

pub struct CudaEngine {
    pub stream: Arc<CudaStream>,
    pub blas: CudaBlas,
    pub f: HashMap<String, CudaFunction>,
    pub w: HashMap<String, CudaSlice<f32>>,
    device: String,
    dtype: DType,
    b: Option<Bufs>,
    cn: usize,
    cl: usize,
    ck: usize,
    rl: usize,
    cal: Calibration,
}

fn cfg(n: usize, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads).max(1), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn rows(n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl CudaEngine {
    pub fn load(spec: &ModelSpec) -> CudaEngine {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let device = ctx.name().unwrap_or_else(|_| "cuda:0".to_string());
        let major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap();
        let minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap();
        let inc = std::env::var("CUDA_PATH")
            .or_else(|_| std::env::var("CUDA_HOME"))
            .unwrap_or_else(|_| "/usr/local/cuda".to_string());
        let opts = CompileOptions {
            include_paths: vec![format!("{}/include", inc)],
            options: vec![format!("--gpu-architecture=sm_{}{}", major, minor)],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(SRC, opts).unwrap();
        let module = ctx.load_module(ptx).unwrap();
        let mut f = HashMap::new();
        for name in [
            "k_gather_rows",
            "k_layernorm",
            "k_copy",
            "k_add",
            "k_add_bias",
            "k_gelu",
            "k_relu",
            "k_gelu_mul",
            "k_add_type",
            "k_rope_table",
            "k_split_qkv",
            "k_mask_build",
            "k_add_mask",
            "k_softmax",
            "k_heads_to_seq",
            "k_marker_gather",
            "k_act_input",
            "k_post",
        ] {
            f.insert(name.to_string(), module.load_function(name).unwrap());
        }
        let tensors = Tensors::open(&spec.base);
        let cal = tensors.meta.calibration();
        let dtype = tensors.meta.dtype();
        let mut w = HashMap::new();
        for n in weight_names() {
            let v = tensors.f32(&n);
            w.insert(n, stream.clone_htod(&v).unwrap());
        }
        let blas = CudaBlas::new(stream.clone()).unwrap();
        CudaEngine {
            stream,
            blas,
            f,
            w,
            device,
            dtype,
            b: None,
            cn: 0,
            cl: 0,
            ck: 0,
            rl: 0,
            cal,
        }
    }

    fn wr(&self, n: &str) -> &CudaSlice<f32> {
        &self.w[n]
    }

    fn alloc(&self, n: usize, l: usize, k: usize) -> Bufs {
        let stream = &self.stream;
        let a = |c: usize| unsafe { stream.alloc::<f32>(c).unwrap() };
        let ai = |c: usize| unsafe { stream.alloc::<i64>(c).unwrap() };
        Bufs {
            x: a(n * l * H),
            t: a(n * l.max(k) * H),
            qkv: a(n * l * 3 * H),
            q: a(n * NH * l * HD),
            k: a(n * NH * l * HD),
            v: a(n * NH * l * HD),
            attn: a(n * NH * l * l),
            ctx: a(n * l * H),
            p: a(n * l * H),
            mlp: a((n * l * 5248).max(n * k * H)),
            gate: a(n * l * INTER),
            cos_a: a(l * 32),
            sin_a: a(l * 32),
            cos_b: a(l * 32),
            sin_b: a(l * 32),
            mask_g: a(n * l * l),
            mask_l: a(n * l * l),
            mrow: a(n * k * H),
            sc: a(n * k),
            logits: a(n * k),
            feats: a(n * 4),
            aact: a(n * (H + 4)),
            act: a(n * 2),
            ids: ai(n * l),
            att: ai(n * l),
            mpos: ai(n * k),
            mmask: ai(n * k),
            qtype: ai(n),
        }
    }

    fn ensure(&mut self, n: usize, l: usize, k: usize) {
        if n <= self.cn && l <= self.cl && k <= self.ck {
            return;
        }
        let (nn, ll, kk) = (n.max(self.cn), l.max(self.cl), k.max(self.ck));
        self.cn = nn;
        self.cl = ll;
        self.ck = kk;
        let mut b = self.alloc(nn, ll, kk);
        {
            let func = self.f["k_rope_table"].clone();
            let (half, seq) = (32i32, ll as i32);
            let c = cfg(ll * 32, 256);
            for (freq, cs, sn) in [
                (ROPE_FREQ_FULL, &mut b.cos_a, &mut b.sin_a),
                (ROPE_FREQ_SLIDING, &mut b.cos_b, &mut b.sin_b),
            ] {
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(&self.w[freq])
                        .arg(cs)
                        .arg(sn)
                        .arg(&seq)
                        .arg(&half)
                        .launch(c)
                        .unwrap();
                }
            }
        }
        self.b = Some(b);
    }

    fn mm(&self, a: &CudaSlice<f32>, name: &str, c: &mut CudaSlice<f32>, m: i32, n: i32, k: i32) {
        let w = &self.w[name];
        let cfg = GemmConfig {
            transa: Op::CUBLAS_OP_N,
            transb: Op::CUBLAS_OP_N,
            m: n,
            n: m,
            k,
            alpha: 1.0f32,
            lda: n,
            ldb: k,
            beta: 0.0f32,
            ldc: n,
        };
        unsafe { self.blas.gemm(cfg, w, a, c).unwrap() };
    }

    fn linear(&self, a: &CudaSlice<f32>, name: &str, c: &mut CudaSlice<f32>, m: i32, n: i32, k: i32) {
        let w = &self.w[name];
        let cfg = GemmConfig {
            transa: Op::CUBLAS_OP_T,
            transb: Op::CUBLAS_OP_N,
            m: n,
            n: m,
            k,
            alpha: 1.0f32,
            lda: k,
            ldb: k,
            beta: 0.0f32,
            ldc: n,
        };
        unsafe { self.blas.gemm(cfg, w, a, c).unwrap() };
    }

    fn bmm_abt(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, batch: i32, m: i32, n: i32, k: i32) {
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: Op::CUBLAS_OP_T,
                transb: Op::CUBLAS_OP_N,
                m: n,
                n: m,
                k,
                alpha: 1.0f32,
                lda: k,
                ldb: k,
                beta: 0.0f32,
                ldc: n,
            },
            batch_size: batch,
            stride_a: (n * k) as i64,
            stride_b: (m * k) as i64,
            stride_c: (m * n) as i64,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, b, a, c).unwrap() };
    }

    fn bmm_ab(&self, a: &CudaSlice<f32>, b: &CudaSlice<f32>, c: &mut CudaSlice<f32>, batch: i32, m: i32, n: i32, k: i32) {
        let cfg = StridedBatchedConfig {
            gemm: GemmConfig {
                transa: Op::CUBLAS_OP_N,
                transb: Op::CUBLAS_OP_N,
                m: n,
                n: m,
                k,
                alpha: 1.0f32,
                lda: n,
                ldb: k,
                beta: 0.0f32,
                ldc: n,
            },
            batch_size: batch,
            stride_a: (k * n) as i64,
            stride_b: (m * k) as i64,
            stride_c: (m * n) as i64,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, b, a, c).unwrap() };
    }

    fn layernorm(
        &self,
        x: &CudaSlice<f32>,
        w: &CudaSlice<f32>,
        bias: Option<&CudaSlice<f32>>,
        y: &mut CudaSlice<f32>,
        nrows: usize,
    ) {
        let dim = H as i32;
        let eps = EPS;
        let null = 0u64;
        let func = self.f["k_layernorm"].clone();
        match bias {
            Some(v) => unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(x)
                    .arg(w)
                    .arg(v)
                    .arg(y)
                    .arg(&dim)
                    .arg(&eps)
                    .launch(rows(nrows))
                    .unwrap();
            },
            None => unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(x)
                    .arg(w)
                    .arg(&null)
                    .arg(y)
                    .arg(&dim)
                    .arg(&eps)
                    .launch(rows(nrows))
                    .unwrap();
            },
        }
    }

    fn add(&self, a: &mut CudaSlice<f32>, b: &CudaSlice<f32>, n: usize) {
        let func = self.f["k_add"].clone();
        let total = n as i64;
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(a)
                .arg(b)
                .arg(&total)
                .launch(cfg(n, 256))
                .unwrap();
        }
    }

    fn add_bias(&self, x: &mut CudaSlice<f32>, b: &CudaSlice<f32>, dim: usize, n: usize) {
        let func = self.f["k_add_bias"].clone();
        let (d, total) = (dim as i32, n as i64);
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(x)
                .arg(b)
                .arg(&d)
                .arg(&total)
                .launch(cfg(n, 256))
                .unwrap();
        }
    }

    fn split_qkv(
        &self,
        qkv: &CudaSlice<f32>,
        cos: &CudaSlice<f32>,
        sin: &CudaSlice<f32>,
        q: &mut CudaSlice<f32>,
        k: &mut CudaSlice<f32>,
        v: &mut CudaSlice<f32>,
        n: usize,
        rope: i32,
    ) {
        let (nh, hd, l) = (NH as i32, HD as i32, self.rl as i32);
        let total = (n * NH * self.rl * HD) as i64;
        let scale = SCALE;
        let func = self.f["k_split_qkv"].clone();
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(qkv)
                .arg(cos)
                .arg(sin)
                .arg(q)
                .arg(k)
                .arg(v)
                .arg(&total)
                .arg(&l)
                .arg(&nh)
                .arg(&hd)
                .arg(&scale)
                .arg(&rope)
                .launch(cfg(n * NH * self.rl * HD, 256))
                .unwrap();
        }
    }

    fn attention(
        &self,
        q: &mut CudaSlice<f32>,
        k: &CudaSlice<f32>,
        v: &CudaSlice<f32>,
        attn: &mut CudaSlice<f32>,
        ctx: &mut CudaSlice<f32>,
        mask: &CudaSlice<f32>,
        n: usize,
    ) {
        let l = self.rl;
        let batch = (n * NH) as i32;
        let li = l as i32;
        self.bmm_abt(q, k, attn, batch, li, li, HD as i32);
        {
            let func = self.f["k_add_mask"].clone();
            let total = (n * NH * l * l) as i64;
            let nh = NH as i32;
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&mut *attn)
                    .arg(mask)
                    .arg(&total)
                    .arg(&li)
                    .arg(&nh)
                    .launch(cfg(n * NH * l * l, 256))
                    .unwrap();
            }
        }
        {
            let func = self.f["k_softmax"].clone();
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&mut *attn)
                    .arg(&li)
                    .launch(rows(n * NH * l))
                    .unwrap();
            }
        }
        self.bmm_ab(attn, v, q, batch, li, HD as i32, li);
        {
            let func = self.f["k_heads_to_seq"].clone();
            let total = (n * NH * l * HD) as i64;
            let (nh, hd) = (NH as i32, HD as i32);
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&*q)
                    .arg(ctx)
                    .arg(&total)
                    .arg(&li)
                    .arg(&nh)
                    .arg(&hd)
                    .launch(cfg(n * NH * l * HD, 256))
                    .unwrap();
            }
        }
    }

    pub fn forward(&mut self, req: &Req) -> Out {
        let (n, l, k) = (req.n, req.l, req.k);
        assert!(n > 0 && l > 0 && k > 0);
        self.ensure(n, l, k);
        self.rl = l;
        let Bufs {
            mut x,
            mut t,
            mut qkv,
            mut q,
            k: kbuf,
            mut v,
            mut attn,
            mut ctx,
            mut p,
            mut mlp,
            mut gate,
            cos_a,
            sin_a,
            cos_b,
            sin_b,
            mut mask_g,
            mut mask_l,
            mut mrow,
            mut sc,
            mut logits,
            mut feats,
            mut aact,
            mut act,
            mut ids,
            mut att,
            mut mpos,
            mut mmask,
            mut qtype,
        } = self.b.take().unwrap();
        let mut kbuf = kbuf;
        let (hi, li, ki) = (H as i32, l as i32, k as i32);
        let nl = n * l;

        self.stream.memcpy_htod(req.ids, &mut ids.slice_mut(0..nl)).unwrap();
        self.stream.memcpy_htod(req.att, &mut att.slice_mut(0..nl)).unwrap();
        self.stream.memcpy_htod(req.mpos, &mut mpos.slice_mut(0..n * k)).unwrap();
        self.stream.memcpy_htod(req.mmask, &mut mmask.slice_mut(0..n * k)).unwrap();
        self.stream.memcpy_htod(req.qtype, &mut qtype.slice_mut(0..n)).unwrap();

        {
            let func = self.f["k_gather_rows"].clone();
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&ids)
                    .arg(self.wr(TOK_EMB))
                    .arg(&mut x)
                    .arg(&hi)
                    .launch(rows(nl))
                    .unwrap();
            }
        }
        {
            let c = LaunchConfig {
                grid_dim: ((l as u32).div_ceil(256).max(1), l as u32, n as u32),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            for (win, buf) in [(0i32, &mut mask_g), (WINDOW, &mut mask_l)] {
                let func = self.f["k_mask_build"].clone();
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(&att)
                        .arg(buf)
                        .arg(&li)
                        .arg(&win)
                        .launch(c)
                        .unwrap();
                }
            }
        }

        let rope_a = (&cos_a, &sin_a);
        let rope_b = (&cos_b, &sin_b);
        for layer in 0..NLAYER {
            let global = is_global_layer(layer);
            self.layernorm(&x, self.wr(ATTN_NORM[layer]), None, &mut t, nl);
            {
                self.mm(&t, WQKV[layer], &mut qkv, nl as i32, 3 * hi, hi);
            }
            let (ca, sa) = if global { rope_a } else { rope_b };
            self.split_qkv(&qkv, ca, sa, &mut q, &mut kbuf, &mut v, n, 1);
            let mask = if global { &mask_g } else { &mask_l };
            self.attention(&mut q, &kbuf, &v, &mut attn, &mut ctx, mask, n);
            {
                self.mm(&ctx, WO_ATTN[layer], &mut p, nl as i32, hi, hi);
            }
            if layer == 0 {
                let cpy = self.f["k_copy"].clone();
                let total = (nl * H) as i64;
                unsafe {
                    self.stream
                        .launch_builder(&cpy)
                        .arg(&mut x)
                        .arg(&t)
                        .arg(&total)
                        .launch(cfg(nl * H, 256))
                        .unwrap();
                }
            }
            self.add(&mut x, &p, nl * H);
            self.layernorm(&x, self.wr(MLP_NORM[layer]), None, &mut t, nl);
            {
                self.mm(&t, WI[layer], &mut mlp, nl as i32, 5248, hi);
            }
            {
                let func = self.f["k_gelu_mul"].clone();
                let (r, half) = (nl as i64, INTER as i64);
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(&mlp)
                        .arg(&mut gate)
                        .arg(&r)
                        .arg(&half)
                        .launch(cfg(nl * INTER, 256))
                        .unwrap();
                }
            }
            {
                self.mm(&gate, WO_MLP[layer], &mut p, nl as i32, hi, INTER as i32);
            }
            self.add(&mut x, &p, nl * H);
        }
        self.layernorm(&x, self.wr(FINAL_NORM), None, &mut t, nl);
        {
            let cpy = self.f["k_copy"].clone();
            let total = (nl * H) as i64;
            unsafe {
                self.stream
                    .launch_builder(&cpy)
                    .arg(&mut x)
                    .arg(&t)
                    .arg(&total)
                    .launch(cfg(nl * H, 256))
                    .unwrap();
            }
        }
        {
            let func = self.f["k_add_type"].clone();
            let total = (nl * H) as i64;
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&mut x)
                    .arg(self.wr(TYPE_EMB))
                    .arg(&qtype)
                    .arg(&total)
                    .arg(&li)
                    .arg(&hi)
                    .launch(cfg(nl * H, 256))
                    .unwrap();
            }
        }
        for layer in 0..2 {
            self.layernorm(
                &x,
                self.wr(H_NORM1_W[layer]),
                Some(self.wr(H_NORM1_B[layer])),
                &mut t,
                nl,
            );
            {
                self.mm(&t, H_WQKV[layer], &mut qkv, nl as i32, 3 * hi, hi);
            }
            self.add_bias(&mut qkv, self.wr(H_IN_BIAS[layer]), 3 * H, nl * 3 * H);
            self.split_qkv(&qkv, rope_a.0, rope_a.0, &mut q, &mut kbuf, &mut v, n, 0);
            self.attention(&mut q, &kbuf, &v, &mut attn, &mut ctx, &mask_g, n);
            {
                self.linear(&ctx, H_OUT_W[layer], &mut p, nl as i32, hi, hi);
            }
            self.add_bias(&mut p, self.wr(H_OUT_B[layer]), H, nl * H);
            self.add(&mut x, &p, nl * H);
            self.layernorm(
                &x,
                self.wr(H_NORM2_W[layer]),
                Some(self.wr(H_NORM2_B[layer])),
                &mut t,
                nl,
            );
            {
                self.mm(&t, H_LIN1_W[layer], &mut mlp, nl as i32, 4096, hi);
            }
            self.add_bias(&mut mlp, self.wr(H_LIN1_B[layer]), 4096, nl * 4096);
            {
                let func = self.f["k_relu"].clone();
                let total = (nl * 4096) as i64;
                unsafe {
                    self.stream
                        .launch_builder(&func)
                        .arg(&mut mlp)
                        .arg(&total)
                        .launch(cfg(nl * 4096, 256))
                        .unwrap();
                }
            }
            {
                self.mm(&mlp, H_LIN2_W[layer], &mut p, nl as i32, hi, 4096);
            }
            self.add_bias(&mut p, self.wr(H_LIN2_B[layer]), H, nl * H);
            self.add(&mut x, &p, nl * H);
        }
        {
            let func = self.f["k_marker_gather"].clone();
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&x)
                    .arg(&mpos)
                    .arg(&mut mrow)
                    .arg(&li)
                    .arg(&ki)
                    .arg(&hi)
                    .launch(rows(n * k))
                    .unwrap();
            }
        }
        self.layernorm(
            &mrow,
            self.wr(SCORER_W),
            Some(self.wr(SCORER_B)),
            &mut t,
            n * k,
        );
        {
            self.mm(&t, SCORER_HID, &mut mlp, (n * k) as i32, hi, hi);
        }
        self.add_bias(&mut mlp, self.wr(SCORER_HID_B), H, n * k * H);
        {
            let func = self.f["k_gelu"].clone();
            let total = (n * k * H) as i64;
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&mut mlp)
                    .arg(&total)
                    .launch(cfg(n * k * H, 256))
                    .unwrap();
            }
        }
        {
            self.mm(&mlp, SCORER_OUT, &mut sc, (n * k) as i32, 1, hi);
        }
        self.add_bias(&mut sc, self.wr(SCORER_OUT_B), 1, n * k);
        {
            let func = self.f["k_post"].clone();
            let c = LaunchConfig {
                grid_dim: (n as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: (k * 4) as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&sc)
                    .arg(&mmask)
                    .arg(&mut logits)
                    .arg(&mut feats)
                    .arg(&ki)
                    .launch(c)
                    .unwrap();
            }
        }
        {
            let func = self.f["k_act_input"].clone();
            let total = (n * (H + 4)) as i64;
            let nf = 4i32;
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&x)
                    .arg(&feats)
                    .arg(&mut aact)
                    .arg(&total)
                    .arg(&li)
                    .arg(&hi)
                    .arg(&nf)
                    .launch(cfg(n * (H + 4), 256))
                    .unwrap();
            }
        }
        {
            self.linear(&aact, ACT_W1, &mut p, n as i32, 256, (H + 4) as i32);
        }
        self.add_bias(&mut p, self.wr(ACT_B1), 256, n * 256);
        {
            let func = self.f["k_gelu"].clone();
            let total = (n * 256) as i64;
            unsafe {
                self.stream
                    .launch_builder(&func)
                    .arg(&mut p)
                    .arg(&total)
                    .launch(cfg(n * 256, 256))
                    .unwrap();
            }
        }
        {
            self.linear(&p, ACT_W2, &mut act, n as i32, 2, 256);
        }
        self.add_bias(&mut act, self.wr(ACT_B2), 2, n * 2);

        let out = Out {
            n,
            k,
            logits: self.stream.clone_dtoh(&logits.slice(0..n * k)).unwrap(),
            act: self.stream.clone_dtoh(&act.slice(0..n * 2)).unwrap(),
        };
        self.b = Some(Bufs {
            x,
            t,
            qkv,
            q,
            k: kbuf,
            v,
            attn,
            ctx,
            p,
            mlp,
            gate,
            cos_a,
            sin_a,
            cos_b,
            sin_b,
            mask_g,
            mask_l,
            mrow,
            sc,
            logits,
            feats,
            aact,
            act,
            ids,
            att,
            mpos,
            mmask,
            qtype,
        });
        out
    }
}

impl Engine for CudaEngine {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: BackendId::Cuda,
            device: self.device.clone(),
            dtype: self.dtype,
            layers: NLAYER,
            calibration: self.cal.clone(),
        }
    }

    fn forward(&mut self, req: &Req) -> Out {
        CudaEngine::forward(self, req)
    }
}
