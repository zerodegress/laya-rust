use crate::arch::*;
use crate::engine::{Out, Req};
use crate::ops::{Backend, TapSink, WeightFormat};
use crate::weights::Tensors;
use std::collections::HashMap;

pub struct Weights<B: Backend> {
    t: HashMap<String, B::Tensor>,
}

impl<B: Backend> Weights<B> {
    pub fn get(&self, name: &str) -> &B::Tensor {
        self.t
            .get(name)
            .unwrap_or_else(|| panic!("weight {} was not loaded", name))
    }
}

pub fn load<B: Backend>(bk: &B, t: &Tensors) -> Weights<B> {
    let names = weight_names();
    let mut map = HashMap::with_capacity(names.len());
    for n in &names {
        let dims = t.dims(n);
        let tensor = match bk.weight_format(n, &dims) {
            WeightFormat::F32 => {
                let (v, _) = t.f32_shaped(n);
                bk.upload_f32(&v, &dims)
            }
            WeightFormat::F16 => {
                let (v, _) = t.f32_shaped(n);
                let h: Vec<u16> = v.iter().map(|x| crate::weights::f2h(*x)).collect();
                bk.upload_f16(n, &h, &dims)
            }
            WeightFormat::Quant { bits, group } => {
                if bk.resident(n) {
                    bk.upload_quant(n, &[], &dims, bits, group)
                } else {
                    let (v, _) = t.f32_shaped(n);
                    bk.upload_quant(n, &v, &dims, bits, group)
                }
            }
        };
        map.insert(n.clone(), tensor);
    }
    Weights { t: map }
}

struct Ctx<'a, 't, B: Backend> {
    bk: &'a B,
    taps: Option<&'t mut dyn TapSink>,
}

impl<B: Backend> Ctx<'_, '_, B> {
    fn tap(&mut self, stage: &str, t: &B::Tensor) {
        if let Some(sink) = self.taps.as_mut() {
            let data = self.bk.download_f32(t);
            sink.tap(stage, &data);
        }
    }
}

fn rope_tables<B: Backend>(bk: &B, freq: &B::Tensor, len: usize) -> (B::Tensor, B::Tensor) {
    let half = HD / 2;
    let f = bk.download_f32(freq);
    assert!(
        f.len() >= half,
        "rope frequency tensor has {} entries, need at least {}",
        f.len(),
        half
    );
    let mut cs = vec![0f32; len * half];
    let mut sn = vec![0f32; len * half];
    for p in 0..len {
        for j in 0..half {
            let a = p as f32 * f[j];
            cs[p * half + j] = a.cos();
            sn[p * half + j] = a.sin();
        }
    }
    (
        bk.upload_f32(&cs, &[len, half]),
        bk.upload_f32(&sn, &[len, half]),
    )
}

pub fn forward<B: Backend>(
    bk: &B,
    w: &Weights<B>,
    req: &Req,
    taps: Option<&mut dyn TapSink>,
) -> Out {
    let (n, l, k) = (req.n, req.l, req.k);
    assert!(n > 0 && l > 0 && k > 0);
    let mut c = Ctx { bk, taps };

    let rope_a = rope_tables(bk, w.get(ROPE_FREQ_FULL), l);
    let rope_b = rope_tables(bk, w.get(ROPE_FREQ_SLIDING), l);
    let mask_g = bk.window_mask(req.att, n, l, None);
    let mask_l = bk.window_mask(req.att, n, l, Some(WINDOW as usize));

    let mut x = bk.embedding(w.get(TOK_EMB), req.ids);
    c.tap("embed", &x);

    for layer in 0..NLAYER {
        let global = is_global_layer(layer);
        let t = bk.layernorm(&x, w.get(ATTN_NORM[layer]), None, EPS);
        let qkv = bk.matmul(&t, w.get(WQKV[layer]));
        let (cos, sin) = if global { (&rope_a.0, &rope_a.1) } else { (&rope_b.0, &rope_b.1) };
        let (q, kk, v) = bk.split_qkv_rope(&qkv, Some((cos, sin)), SCALE, NH, HD, l);
        let mask = if global { &mask_g } else { &mask_l };
        let ctx = bk.attention(&q, &kk, &v, mask, NH, l);
        let p = bk.matmul(&ctx, w.get(WO_ATTN[layer]));
        x = if layer == 0 {
            bk.add(&t, &p)
        } else {
            bk.add(&x, &p)
        };

        let t = bk.layernorm(&x, w.get(MLP_NORM[layer]), None, EPS);
        let mlp = bk.matmul(&t, w.get(WI[layer]));
        let gate = bk.gelu_mul(&mlp, INTER);
        let p = bk.matmul(&gate, w.get(WO_MLP[layer]));
        x = bk.add(&x, &p);
        c.tap(&format!("enc.{}", layer), &x);
    }

    let t = bk.layernorm(&x, w.get(FINAL_NORM), None, EPS);
    let mut x = bk.add_type(&t, w.get(TYPE_EMB), req.qtype, l);
    c.tap("final", &x);

    for layer in 0..2 {
        let t = bk.layernorm(
            &x,
            w.get(H_NORM1_W[layer]),
            Some(w.get(H_NORM1_B[layer])),
            EPS,
        );
        let qkv = bk.matmul(&t, w.get(H_WQKV[layer]));
        let qkv = bk.add_bias(&qkv, w.get(H_IN_BIAS[layer]));
        let (q, kk, v) = bk.split_qkv_rope(&qkv, None, SCALE, NH, HD, l);
        let ctx = bk.attention(&q, &kk, &v, &mask_g, NH, l);
        let p = bk.matmul_t(&ctx, w.get(H_OUT_W[layer]));
        let p = bk.add_bias(&p, w.get(H_OUT_B[layer]));
        x = bk.add(&x, &p);

        let t = bk.layernorm(
            &x,
            w.get(H_NORM2_W[layer]),
            Some(w.get(H_NORM2_B[layer])),
            EPS,
        );
        let mlp = bk.matmul(&t, w.get(H_LIN1_W[layer]));
        let mlp = bk.add_bias(&mlp, w.get(H_LIN1_B[layer]));
        let mlp = bk.relu(&mlp);
        let p = bk.matmul(&mlp, w.get(H_LIN2_W[layer]));
        let p = bk.add_bias(&p, w.get(H_LIN2_B[layer]));
        x = bk.add(&x, &p);
        c.tap(&format!("head.{}", layer), &x);
    }

    let mrow = bk.gather_markers(&x, req.mpos, k);
    let t = bk.layernorm(&mrow, w.get(SCORER_W), Some(w.get(SCORER_B)), EPS);
    let hid = bk.matmul(&t, w.get(SCORER_HID));
    let hid = bk.add_bias(&hid, w.get(SCORER_HID_B));
    let hid = bk.gelu(&hid);
    let sc = bk.matmul(&hid, w.get(SCORER_OUT));
    let sc = bk.add_bias(&sc, w.get(SCORER_OUT_B));
    let (logits, feats) = bk.post(&sc, req.mmask, k);
    c.tap("logits", &logits);

    let aact = bk.act_input(&x, &feats, l);
    let p = bk.matmul_t(&aact, w.get(ACT_W1));
    let p = bk.add_bias(&p, w.get(ACT_B1));
    let p = bk.gelu(&p);
    let act = bk.matmul_t(&p, w.get(ACT_W2));
    let act = bk.add_bias(&act, w.get(ACT_B2));
    c.tap("act", &act);

    bk.sync();
    Out {
        n,
        k,
        logits: bk.download_f32(&logits),
        act: bk.download_f32(&act),
    }
}
