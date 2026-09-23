pub const H: usize = 1024;
pub const NH: usize = 16;
pub const HD: usize = 64;
pub const NLAYER: usize = 28;
pub const INTER: usize = 2624;
pub const NVOCAB: usize = 50368;
pub const SCALE: f32 = 0.3535533845424652;
pub const EPS: f32 = 1e-5;
pub const WINDOW: i32 = 64;
pub const HEAD_LAYERS: usize = 2;
pub const HEAD_INTER: usize = 4096;
pub const ACT_HIDDEN: usize = 256;
pub const ACT_INPUT: usize = H + 4;
pub const QTYPES: usize = 3;
pub const GLOBAL_EVERY: usize = 3;

pub const TOK_EMB: &str = "encoder.embeddings.tok_embeddings.weight";
pub const TYPE_EMB: &str = "type_emb.weight";
pub const FINAL_NORM: &str = "encoder.final_norm.weight";
pub const ROPE_FREQ_FULL: &str = "rope.freq_full";
pub const ROPE_FREQ_SLIDING: &str = "rope.freq_sliding";
pub const SCORER_W: &str = "scorer.0.weight";
pub const SCORER_B: &str = "scorer.0.bias";
pub const SCORER_HID: &str = "scorer.1.weight";
pub const SCORER_HID_B: &str = "scorer.1.bias";
pub const SCORER_OUT: &str = "scorer.3.weight";
pub const SCORER_OUT_B: &str = "scorer.3.bias";
pub const ACT_W1: &str = "act_head.0.weight";
pub const ACT_B1: &str = "act_head.0.bias";
pub const ACT_W2: &str = "act_head.2.weight";
pub const ACT_B2: &str = "act_head.2.bias";

pub const ATTN_NORM: [&str; 28] = [
    "encoder.embeddings.norm.weight",
    "encoder.layers.1.attn_norm.weight",
    "encoder.layers.2.attn_norm.weight",
    "encoder.layers.3.attn_norm.weight",
    "encoder.layers.4.attn_norm.weight",
    "encoder.layers.5.attn_norm.weight",
    "encoder.layers.6.attn_norm.weight",
    "encoder.layers.7.attn_norm.weight",
    "encoder.layers.8.attn_norm.weight",
    "encoder.layers.9.attn_norm.weight",
    "encoder.layers.10.attn_norm.weight",
    "encoder.layers.11.attn_norm.weight",
    "encoder.layers.12.attn_norm.weight",
    "encoder.layers.13.attn_norm.weight",
    "encoder.layers.14.attn_norm.weight",
    "encoder.layers.15.attn_norm.weight",
    "encoder.layers.16.attn_norm.weight",
    "encoder.layers.17.attn_norm.weight",
    "encoder.layers.18.attn_norm.weight",
    "encoder.layers.19.attn_norm.weight",
    "encoder.layers.20.attn_norm.weight",
    "encoder.layers.21.attn_norm.weight",
    "encoder.layers.22.attn_norm.weight",
    "encoder.layers.23.attn_norm.weight",
    "encoder.layers.24.attn_norm.weight",
    "encoder.layers.25.attn_norm.weight",
    "encoder.layers.26.attn_norm.weight",
    "encoder.layers.27.attn_norm.weight",
];
pub const MLP_NORM: [&str; 28] = [
    "encoder.layers.0.mlp_norm.weight",
    "encoder.layers.1.mlp_norm.weight",
    "encoder.layers.2.mlp_norm.weight",
    "encoder.layers.3.mlp_norm.weight",
    "encoder.layers.4.mlp_norm.weight",
    "encoder.layers.5.mlp_norm.weight",
    "encoder.layers.6.mlp_norm.weight",
    "encoder.layers.7.mlp_norm.weight",
    "encoder.layers.8.mlp_norm.weight",
    "encoder.layers.9.mlp_norm.weight",
    "encoder.layers.10.mlp_norm.weight",
    "encoder.layers.11.mlp_norm.weight",
    "encoder.layers.12.mlp_norm.weight",
    "encoder.layers.13.mlp_norm.weight",
    "encoder.layers.14.mlp_norm.weight",
    "encoder.layers.15.mlp_norm.weight",
    "encoder.layers.16.mlp_norm.weight",
    "encoder.layers.17.mlp_norm.weight",
    "encoder.layers.18.mlp_norm.weight",
    "encoder.layers.19.mlp_norm.weight",
    "encoder.layers.20.mlp_norm.weight",
    "encoder.layers.21.mlp_norm.weight",
    "encoder.layers.22.mlp_norm.weight",
    "encoder.layers.23.mlp_norm.weight",
    "encoder.layers.24.mlp_norm.weight",
    "encoder.layers.25.mlp_norm.weight",
    "encoder.layers.26.mlp_norm.weight",
    "encoder.layers.27.mlp_norm.weight",
];
pub const WQKV: [&str; 28] = [
    "encoder.layers.0.attn.Wqkv.weight",
    "encoder.layers.1.attn.Wqkv.weight",
    "encoder.layers.2.attn.Wqkv.weight",
    "encoder.layers.3.attn.Wqkv.weight",
    "encoder.layers.4.attn.Wqkv.weight",
    "encoder.layers.5.attn.Wqkv.weight",
    "encoder.layers.6.attn.Wqkv.weight",
    "encoder.layers.7.attn.Wqkv.weight",
    "encoder.layers.8.attn.Wqkv.weight",
    "encoder.layers.9.attn.Wqkv.weight",
    "encoder.layers.10.attn.Wqkv.weight",
    "encoder.layers.11.attn.Wqkv.weight",
    "encoder.layers.12.attn.Wqkv.weight",
    "encoder.layers.13.attn.Wqkv.weight",
    "encoder.layers.14.attn.Wqkv.weight",
    "encoder.layers.15.attn.Wqkv.weight",
    "encoder.layers.16.attn.Wqkv.weight",
    "encoder.layers.17.attn.Wqkv.weight",
    "encoder.layers.18.attn.Wqkv.weight",
    "encoder.layers.19.attn.Wqkv.weight",
    "encoder.layers.20.attn.Wqkv.weight",
    "encoder.layers.21.attn.Wqkv.weight",
    "encoder.layers.22.attn.Wqkv.weight",
    "encoder.layers.23.attn.Wqkv.weight",
    "encoder.layers.24.attn.Wqkv.weight",
    "encoder.layers.25.attn.Wqkv.weight",
    "encoder.layers.26.attn.Wqkv.weight",
    "encoder.layers.27.attn.Wqkv.weight",
];
pub const WO_ATTN: [&str; 28] = [
    "encoder.layers.0.attn.Wo.weight",
    "encoder.layers.1.attn.Wo.weight",
    "encoder.layers.2.attn.Wo.weight",
    "encoder.layers.3.attn.Wo.weight",
    "encoder.layers.4.attn.Wo.weight",
    "encoder.layers.5.attn.Wo.weight",
    "encoder.layers.6.attn.Wo.weight",
    "encoder.layers.7.attn.Wo.weight",
    "encoder.layers.8.attn.Wo.weight",
    "encoder.layers.9.attn.Wo.weight",
    "encoder.layers.10.attn.Wo.weight",
    "encoder.layers.11.attn.Wo.weight",
    "encoder.layers.12.attn.Wo.weight",
    "encoder.layers.13.attn.Wo.weight",
    "encoder.layers.14.attn.Wo.weight",
    "encoder.layers.15.attn.Wo.weight",
    "encoder.layers.16.attn.Wo.weight",
    "encoder.layers.17.attn.Wo.weight",
    "encoder.layers.18.attn.Wo.weight",
    "encoder.layers.19.attn.Wo.weight",
    "encoder.layers.20.attn.Wo.weight",
    "encoder.layers.21.attn.Wo.weight",
    "encoder.layers.22.attn.Wo.weight",
    "encoder.layers.23.attn.Wo.weight",
    "encoder.layers.24.attn.Wo.weight",
    "encoder.layers.25.attn.Wo.weight",
    "encoder.layers.26.attn.Wo.weight",
    "encoder.layers.27.attn.Wo.weight",
];
pub const WI: [&str; 28] = [
    "encoder.layers.0.mlp.Wi.weight",
    "encoder.layers.1.mlp.Wi.weight",
    "encoder.layers.2.mlp.Wi.weight",
    "encoder.layers.3.mlp.Wi.weight",
    "encoder.layers.4.mlp.Wi.weight",
    "encoder.layers.5.mlp.Wi.weight",
    "encoder.layers.6.mlp.Wi.weight",
    "encoder.layers.7.mlp.Wi.weight",
    "encoder.layers.8.mlp.Wi.weight",
    "encoder.layers.9.mlp.Wi.weight",
    "encoder.layers.10.mlp.Wi.weight",
    "encoder.layers.11.mlp.Wi.weight",
    "encoder.layers.12.mlp.Wi.weight",
    "encoder.layers.13.mlp.Wi.weight",
    "encoder.layers.14.mlp.Wi.weight",
    "encoder.layers.15.mlp.Wi.weight",
    "encoder.layers.16.mlp.Wi.weight",
    "encoder.layers.17.mlp.Wi.weight",
    "encoder.layers.18.mlp.Wi.weight",
    "encoder.layers.19.mlp.Wi.weight",
    "encoder.layers.20.mlp.Wi.weight",
    "encoder.layers.21.mlp.Wi.weight",
    "encoder.layers.22.mlp.Wi.weight",
    "encoder.layers.23.mlp.Wi.weight",
    "encoder.layers.24.mlp.Wi.weight",
    "encoder.layers.25.mlp.Wi.weight",
    "encoder.layers.26.mlp.Wi.weight",
    "encoder.layers.27.mlp.Wi.weight",
];
pub const WO_MLP: [&str; 28] = [
    "encoder.layers.0.mlp.Wo.weight",
    "encoder.layers.1.mlp.Wo.weight",
    "encoder.layers.2.mlp.Wo.weight",
    "encoder.layers.3.mlp.Wo.weight",
    "encoder.layers.4.mlp.Wo.weight",
    "encoder.layers.5.mlp.Wo.weight",
    "encoder.layers.6.mlp.Wo.weight",
    "encoder.layers.7.mlp.Wo.weight",
    "encoder.layers.8.mlp.Wo.weight",
    "encoder.layers.9.mlp.Wo.weight",
    "encoder.layers.10.mlp.Wo.weight",
    "encoder.layers.11.mlp.Wo.weight",
    "encoder.layers.12.mlp.Wo.weight",
    "encoder.layers.13.mlp.Wo.weight",
    "encoder.layers.14.mlp.Wo.weight",
    "encoder.layers.15.mlp.Wo.weight",
    "encoder.layers.16.mlp.Wo.weight",
    "encoder.layers.17.mlp.Wo.weight",
    "encoder.layers.18.mlp.Wo.weight",
    "encoder.layers.19.mlp.Wo.weight",
    "encoder.layers.20.mlp.Wo.weight",
    "encoder.layers.21.mlp.Wo.weight",
    "encoder.layers.22.mlp.Wo.weight",
    "encoder.layers.23.mlp.Wo.weight",
    "encoder.layers.24.mlp.Wo.weight",
    "encoder.layers.25.mlp.Wo.weight",
    "encoder.layers.26.mlp.Wo.weight",
    "encoder.layers.27.mlp.Wo.weight",
];

pub const H_NORM1_W: [&str; 2] = [
    "head.layers.0.norm1.weight",
    "head.layers.1.norm1.weight",
];
pub const H_NORM1_B: [&str; 2] = [
    "head.layers.0.norm1.bias",
    "head.layers.1.norm1.bias",
];
pub const H_NORM2_W: [&str; 2] = [
    "head.layers.0.norm2.weight",
    "head.layers.1.norm2.weight",
];
pub const H_NORM2_B: [&str; 2] = [
    "head.layers.0.norm2.bias",
    "head.layers.1.norm2.bias",
];
pub const H_WQKV: [&str; 2] = [
    "head.layers.0.self_attn.in_proj_weight",
    "head.layers.1.self_attn.in_proj_weight",
];
pub const H_IN_BIAS: [&str; 2] = [
    "head.layers.0.self_attn.in_proj_bias",
    "head.layers.1.self_attn.in_proj_bias",
];
pub const H_OUT_W: [&str; 2] = [
    "head.layers.0.self_attn.out_proj.weight",
    "head.layers.1.self_attn.out_proj.weight",
];
pub const H_OUT_B: [&str; 2] = [
    "head.layers.0.self_attn.out_proj.bias",
    "head.layers.1.self_attn.out_proj.bias",
];
pub const H_LIN1_W: [&str; 2] = [
    "head.layers.0.linear1.weight",
    "head.layers.1.linear1.weight",
];
pub const H_LIN1_B: [&str; 2] = [
    "head.layers.0.linear1.bias",
    "head.layers.1.linear1.bias",
];
pub const H_LIN2_W: [&str; 2] = [
    "head.layers.0.linear2.weight",
    "head.layers.1.linear2.weight",
];
pub const H_LIN2_B: [&str; 2] = [
    "head.layers.0.linear2.bias",
    "head.layers.1.linear2.bias",
];

pub fn weight_names() -> Vec<String> {
    let mut names: Vec<String> = vec![
        TOK_EMB.into(),
        TYPE_EMB.into(),
        FINAL_NORM.into(),
        ROPE_FREQ_FULL.into(),
        ROPE_FREQ_SLIDING.into(),
        SCORER_W.into(),
        SCORER_B.into(),
        SCORER_HID.into(),
        SCORER_HID_B.into(),
        SCORER_OUT.into(),
        SCORER_OUT_B.into(),
        ACT_W1.into(),
        ACT_B1.into(),
        ACT_W2.into(),
        ACT_B2.into(),
    ];
    for i in 0..NLAYER {
        names.push(ATTN_NORM[i].into());
        names.push(MLP_NORM[i].into());
        names.push(WQKV[i].into());
        names.push(WO_ATTN[i].into());
        names.push(WI[i].into());
        names.push(WO_MLP[i].into());
    }
    for i in 0..HEAD_LAYERS {
        names.push(H_NORM1_W[i].into());
        names.push(H_NORM1_B[i].into());
        names.push(H_NORM2_W[i].into());
        names.push(H_NORM2_B[i].into());
        names.push(H_WQKV[i].into());
        names.push(H_IN_BIAS[i].into());
        names.push(H_OUT_W[i].into());
        names.push(H_OUT_B[i].into());
        names.push(H_LIN1_W[i].into());
        names.push(H_LIN1_B[i].into());
        names.push(H_LIN2_W[i].into());
        names.push(H_LIN2_B[i].into());
    }
    names
}

pub fn transpose_names() -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for i in 0..NLAYER {
        names.push(WQKV[i].into());
        names.push(WO_ATTN[i].into());
        names.push(WI[i].into());
        names.push(WO_MLP[i].into());
    }
    for i in 0..HEAD_LAYERS {
        names.push(H_WQKV[i].into());
        names.push(H_LIN1_W[i].into());
        names.push(H_LIN2_W[i].into());
    }
    names.push(SCORER_HID.into());
    names.push(SCORER_OUT.into());
    names
}

pub fn is_global_layer(layer: usize) -> bool {
    layer % GLOBAL_EVERY == 0
}
