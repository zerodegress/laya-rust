#![cfg(any(feature = "cpu", feature = "mlx"))]

use laya_rust::ops::Backend;

const NEG_BIG: f32 = -3.4e38;

fn close(got: f32, want: f32, tol: f32) -> bool {
    (got - want).abs() <= tol * (1.0 + want.abs())
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{}: length", what);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            close(*g, *w, tol),
            "{}[{}]: got {} want {}",
            what,
            i,
            g,
            w
        );
    }
}

macro_rules! suite {
    ($name:ident, $backend:ty, $ctor:expr) => {
        mod $name {
            use super::*;

            fn bk() -> $backend {
                $ctor
            }

            fn t(data: &[f32], shape: &[usize]) -> <$backend as Backend>::Tensor {
                bk().upload_f32(data, shape)
            }

            fn dl(t: &<$backend as Backend>::Tensor) -> Vec<f32> {
                bk().download_f32(t)
            }

            #[test]
            fn layernorm_z_scores_rows() {
                let x = t(&[1.0, -1.0, 0.5, -0.5], &[2, 2]);
                let w = t(&[1.0, 1.0], &[2]);
                let y = bk().layernorm(&x, &w, None, 1e-5);
                assert_eq!(y.shape(), &[2, 2]);
                assert_close(&dl(&y), &[1.0, -1.0, 1.0, -1.0], 1e-4, "layernorm");
            }

            #[test]
            fn layernorm_applies_weight_and_bias() {
                let x = t(&[2.0, 2.0], &[1, 2]);
                let w = t(&[3.0, 5.0], &[2]);
                let b = t(&[1.0, -1.0], &[2]);
                let y = bk().layernorm(&x, &w, Some(&b), 1e-5);
                assert_close(&dl(&y), &[1.0, -1.0], 1e-4, "layernorm+bias");
            }

            #[test]
            fn matmul_and_matmul_t_agree_on_the_same_matrix() {
                let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
                let wdata = [1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 3.0, 4.0];
                let w = t(&wdata, &[2, 4]);
                let a = bk().matmul(&x, &w);
                assert_eq!(a.shape(), &[2, 4]);
                assert_close(&dl(&a)[0..2], &[1.0, 2.0], 1e-5, "matmul identity");
                assert_close(&dl(&a)[4..6], &[3.0, 4.0], 1e-5, "matmul identity");

                let mut wt = vec![0f32; 8];
                for i in 0..2 {
                    for o in 0..4 {
                        wt[o * 2 + i] = wdata[i * 4 + o];
                    }
                }
                let b = bk().matmul_t(&x, &t(&wt, &[4, 2]));
                assert_close(&dl(&a), &dl(&b), 1e-5, "mm vs mm_t");
            }

            #[test]
            fn gelu_matches_reference_points() {
                let x = t(&[-1.0, 0.0, 1.0, 2.0], &[1, 4]);
                let y = bk().gelu(&x);
                assert_close(
                    &dl(&y),
                    &[-0.158655, 0.0, 0.841345, 1.954500],
                    1e-4,
                    "gelu",
                );
            }

            #[test]
            fn relu_clamps_negatives() {
                let x = t(&[-2.0, -0.0, 0.5, 3.0], &[1, 4]);
                let y = bk().relu(&x);
                assert_close(&dl(&y), &[0.0, 0.0, 0.5, 3.0], 1e-6, "relu");
            }

            #[test]
            fn gelu_mul_gates_the_first_half_with_the_second() {
                let x = t(&[1.0, 0.0, 2.0, -3.0], &[1, 4]);
                let y = bk().gelu_mul(&x, 2);
                assert_eq!(y.shape(), &[1, 2]);
                let g = dl(&bk().gelu(&t(&[1.0, 0.0], &[1, 2])));
                assert_close(&dl(&y), &[g[0] * 2.0, g[1] * -3.0], 1e-5, "gelu_mul");
            }

            #[test]
            fn split_qkv_without_rope_scales_qk_not_v() {
                let qkv = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 6]);
                let (q, k, v) = bk().split_qkv_rope(&qkv, None, 1.0, 1, 2, 1);
                assert_eq!(q.shape(), &[1, 1, 1, 2]);
                assert_close(&dl(&q), &[1.0, 2.0], 1e-6, "q");
                assert_close(&dl(&k), &[3.0, 4.0], 1e-6, "k");
                assert_close(&dl(&v), &[5.0, 6.0], 1e-6, "v");

                let (q2, k2, _) = bk().split_qkv_rope(&qkv, None, 0.5, 1, 2, 1);
                assert_close(&dl(&q2), &[0.5, 1.0], 1e-6, "scaled q");
                assert_close(&dl(&k2), &[1.5, 2.0], 1e-6, "scaled k");
            }

            #[test]
            fn split_qkv_rope_uses_the_rotate_half_pairing() {
                let qkv = t(
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                    &[1, 12],
                );
                let cos = t(&[0.0, 1.0], &[1, 2]);
                let sin = t(&[1.0, 0.0], &[1, 2]);
                let (q, k, v) = bk().split_qkv_rope(&qkv, Some((&cos, &sin)), 1.0, 1, 4, 1);
                assert_close(&dl(&q), &[-3.0, 2.0, 1.0, 4.0], 1e-6, "rope q");
                assert_close(&dl(&k), &[-7.0, 6.0, 5.0, 8.0], 1e-6, "rope k");
                assert_close(&dl(&v), &[9.0, 10.0, 11.0, 12.0], 1e-6, "rope v");
            }

            #[test]
            fn attention_is_softmax_qk_over_v() {
                let q = t(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
                let k = t(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
                let v = t(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
                let mask = t(&[0.0; 4], &[1, 2, 2]);
                let out = bk().attention(&q, &k, &v, &mask, 1, 2);
                assert_eq!(out.shape(), &[2, 2], "heads_to_seq layout");
                let e = std::f32::consts::E;
                let p = e / (e + 1.0);
                assert_close(&dl(&out), &[p, 1.0 - p, 1.0 - p, p], 1e-5, "attention");
            }

            #[test]
            fn attention_mask_drops_keys() {
                let q = t(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
                let v = t(&[1.0, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
                let mask = t(&[0.0, NEG_BIG, 0.0, 0.0], &[1, 2, 2]);
                let out = bk().attention(&q, &q, &v, &mask, 1, 2);
                let e = std::f32::consts::E;
                let p = e / (e + 1.0);
                assert_close(&dl(&out), &[1.0, 0.0, 1.0 - p, p], 1e-5, "masked attention");
            }

            #[test]
            fn window_mask_marks_padding_and_distant_keys() {
                let m = bk().window_mask(&[1, 1, 0, 1], 1, 4, None);
                assert_eq!(m.shape(), &[1, 4, 4]);
                let d = dl(&m);
                for s in 0..4 {
                    assert_eq!(d[s * 4 + 2], NEG_BIG, "row {} padding column", s);
                    for t in [0, 3] {
                        assert_eq!(d[s * 4 + t], 0.0, "row {} column {}", s, t);
                    }
                }

                let m = bk().window_mask(&[1, 1, 1, 1], 1, 4, Some(1));
                let d = dl(&m);
                for s in 0..4usize {
                    for tt in 0..4usize {
                        let want = if s.abs_diff(tt) > 1 { NEG_BIG } else { 0.0 };
                        assert_eq!(d[s * 4 + tt], want, "window row {} col {}", s, tt);
                    }
                }
            }

            #[test]
            fn embedding_copies_rows() {
                let table = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
                let out = bk().embedding(&table, &[2, 0]);
                assert_eq!(out.shape(), &[2, 2]);
                assert_close(&dl(&out), &[5.0, 6.0, 1.0, 2.0], 1e-6, "embedding");
            }

            #[test]
            fn add_bias_wraps_over_the_last_axis() {
                let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
                let b = t(&[10.0, 20.0], &[2]);
                let y = bk().add_bias(&x, &b);
                assert_close(
                    &dl(&y),
                    &[11.0, 22.0, 13.0, 24.0, 15.0, 26.0],
                    1e-6,
                    "add_bias",
                );
            }

            #[test]
            fn add_type_uses_the_sequence_index() {
                let x = t(&[0.0; 8], &[4, 2]);
                let te = t(&[1.0, 1.0, 100.0, 100.0], &[2, 2]);
                let y = bk().add_type(&x, &te, &[1, 0], 2);
                assert_close(
                    &dl(&y),
                    &[100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0, 1.0],
                    1e-6,
                    "add_type",
                );
            }

            #[test]
            fn gather_markers_picks_the_right_rows() {
                let mut data = Vec::new();
                for b in 0..2 {
                    for s in 0..3 {
                        data.push((b * 100 + s) as f32);
                        data.push((b * 100 + s) as f32 + 0.5);
                    }
                }
                let x = t(&data, &[6, 2]);
                let y = bk().gather_markers(&x, &[2, 0, 1, 1], 2);
                assert_eq!(y.shape(), &[4, 2]);
                assert_close(
                    &dl(&y),
                    &[2.0, 2.5, 0.0, 0.5, 101.0, 101.5, 101.0, 101.5],
                    1e-6,
                    "gather_markers",
                );
            }

            #[test]
            fn act_input_concatenates_first_token_with_features() {
                let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]);
                let feats = t(&[0.25, 0.75], &[2, 1]);
                let y = bk().act_input(&x, &feats, 2);
                assert_eq!(y.shape(), &[2, 3]);
                assert_close(&dl(&y), &[1.0, 2.0, 0.25, 5.0, 6.0, 0.75], 1e-6, "act_input");
            }

            #[test]
            fn post_masks_scores_and_derives_features() {
                let sc = t(&[1.0, 2.0, 3.0], &[3]);
                let (logits, feats) = bk().post(&sc, &[1, 0, 1], 3);
                assert_eq!(logits.shape(), &[3]);
                assert_eq!(feats.shape(), &[1, 4]);

                assert_close(&dl(&logits), &[1.0, -10000.0, 3.0], 1e-6, "masked logits");
                let f = dl(&feats);
                assert!(close(f[0], 0.8808, 1e-3), "top1 = {}", f[0]);
                assert!(close(f[1], 0.8808 - 0.1192, 1e-3), "top1-top2 = {}", f[1]);
                assert!(f[2] > 0.0 && f[2] < 1.0, "normalised entropy = {}", f[2]);
                assert!(close(f[3], 2.0 / 255.0, 1e-4), "n/255 = {}", f[3]);
            }

            #[test]
            fn post_survives_a_single_visible_marker() {
                let sc = t(&[5.0, 1.0], &[2]);
                let (logits, feats) = bk().post(&sc, &[1, 0], 2);
                let f = dl(&feats);
                assert!(f[0].is_finite() && f[2].is_finite(), "feats = {:?}", f);
                assert_close(&dl(&logits)[1..2], &[-10000.0], 1e-6, "sentinel");
            }
        }
    };
}

#[cfg(feature = "cpu")]
suite!(cpu, laya_rust::backend::cpu::CpuBackend, laya_rust::backend::cpu::CpuBackend);

#[cfg(feature = "mlx")]
suite!(mlx, laya_rust::backend::mlx::MlxBackend, laya_rust::backend::mlx::MlxBackend::dense());
