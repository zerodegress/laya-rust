use laya_rust::gguf::{self, Value, ALIGNMENT, T_F16, T_F32};

fn tmp(name: &str) -> String {
    std::env::temp_dir()
        .join(format!("laya-gguf-test-{}-{}", std::process::id(), name))
        .to_string_lossy()
        .into_owned()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    gguf::as_f32_bytes(v)
}

#[test]
fn metadata_value_types_round_trip() {
    let path = tmp("kv.gguf");
    let mut w = gguf::Writer::new();
    w.kv("general.architecture", Value::Str("laya".into()));
    w.kv("laya.u8", Value::U8(200));
    w.kv("laya.i8", Value::I8(-100));
    w.kv("laya.u16", Value::U16(60000));
    w.kv("laya.i16", Value::I16(-30000));
    w.kv("laya.u32", Value::U32(4_000_000_000));
    w.kv("laya.i32", Value::I32(-2_000_000_000));
    w.kv("laya.f32", Value::F32(0.353_553_38));
    w.kv("laya.bool_t", Value::Bool(true));
    w.kv("laya.bool_f", Value::Bool(false));
    w.kv("laya.u64", Value::U64(u64::MAX));
    w.kv("laya.i64", Value::I64(i64::MIN));
    w.kv("laya.f64", Value::F64(1.0e-5));
    w.kv(
        "laya.str",
        Value::Str("现代BERT · [MASK] 50284".into()),
    );
    w.kv(
        "laya.arr_i32",
        Value::Array(vec![Value::I32(1), Value::I32(2), Value::I32(3)]),
    );
    w.kv(
        "laya.arr_f32",
        Value::Array(vec![Value::F32(1.5), Value::F32(-2.25)]),
    );
    w.kv(
        "laya.arr_str",
        Value::Array(vec![Value::Str("a b".into()), Value::Str("c d".into())]),
    );
    w.kv(
        "laya.arr_nested",
        Value::Array(vec![
            Value::Array(vec![Value::U8(1), Value::U8(2)]),
            Value::Array(vec![Value::U8(3)]),
        ]),
    );
    w.tensor("t.f32", &[3, 4], T_F32, f32_bytes(&(0..12).map(|i| i as f32 * 0.5).collect::<Vec<f32>>()));
    w.tensor("t.f16", &[8], T_F16, gguf::as_f16_bytes(&[0.0, 1.0, -1.0, 0.5, 0.25, 1e-5, 1234.0, -0.0]));
    w.tensor("t.oneelem", &[1], T_F32, f32_bytes(&[42.0]));
    w.write(&path);

    let h = gguf::read(&path);
    assert_eq!(h.data_off % ALIGNMENT, 0, "tensor data must be aligned");
    for t in &h.tensors {
        assert_eq!(t.offset % ALIGNMENT, 0, "tensor {} offset not aligned", t.name);
    }
    assert_eq!(h.str_or("general.architecture", "?"), "laya");
    assert_eq!(h.u64_or("laya.u8", 0), 200);
    assert_eq!(h.u64_or("laya.u32", 0), 4_000_000_000);
    assert_eq!(h.u64_or("laya.u64", 0), u64::MAX);
    assert_eq!(h.f32_or("laya.f32", 0.0), 0.353_553_38);
    assert_eq!(h.f32_or("laya.f64", 0.0), 1.0e-5);
    assert!(matches!(h.get("laya.bool_t"), Some(Value::Bool(true))));
    assert!(matches!(h.get("laya.bool_f"), Some(Value::Bool(false))));
    assert_eq!(
        h.get("laya.str").and_then(|v| v.as_str()),
        Some("现代BERT · [MASK] 50284")
    );
    assert!(
        h.get("laya.arr_i32")
            .map(|v| v.as_f32s())
            .unwrap_or_default()
            .is_empty(),
        "int arrays are not float arrays"
    );
    assert_eq!(
        h.get("laya.arr_str").map(|v| v.as_strs()),
        Some(vec!["a b".to_string(), "c d".to_string()])
    );
    assert_eq!(
        h.get("laya.arr_f32").map(|v| v.as_f32s()),
        Some(vec![1.5, -2.25])
    );
    match h.get("laya.arr_nested") {
        Some(Value::Array(outer)) => {
            assert_eq!(outer.len(), 2);
            match &outer[0] {
                Value::Array(inner) => assert_eq!(inner.len(), 2),
                other => panic!("expected nested array, got {:?}", other),
            }
        }
        other => panic!("expected nested array, got {:?}", other),
    }
    assert_eq!(h.get("laya.i64").and_then(|v| v.as_u64()), Some(i64::MIN as u64));

    let t32 = h.tensor("t.f32").expect("t.f32");
    assert_eq!(t32.dims, vec![3, 4]);
    assert_eq!(t32.ty, T_F32);
    assert_eq!(t32.nbytes(), 48);
    let raw = gguf::read_tensor_bytes(&path, &h, "t.f32");
    let back = gguf::decode_f32(&raw, T_F32, "t.f32");
    assert_eq!(back, (0..12).map(|i| i as f32 * 0.5).collect::<Vec<f32>>());

    let t16 = h.tensor("t.f16").expect("t.f16");
    assert_eq!(t16.nbytes(), 16);
    let raw = gguf::read_tensor_bytes(&path, &h, "t.f16");
    let back = gguf::decode_f32(&raw, T_F16, "t.f16");
    assert_eq!(back[1], 1.0);
    assert_eq!(back[6], 1234.0);
    assert_eq!(back[7], 0.0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn f16_round_trip_is_exact_for_representable_values() {
    for v in [
        0.0f32,
        1.0,
        -1.0,
        0.5,
        0.25,
        1024.0,
        -0.0,
        65504.0,
        5.960_464_5e-8,
        9.536_743_2e-7,
        -2.0,
        0.333_251_95,
    ] {
        let h = laya_rust::weights::f2h(v);
        assert_eq!(laya_rust::weights::h2f(h), v, "{} did not round-trip", v);
    }
    assert_eq!(laya_rust::weights::f2h(1e-5), 0x00a8);
    assert_eq!(laya_rust::weights::h2f(0x00a8), 168.0 * 2f32.powi(-24));
    assert_eq!(laya_rust::weights::f2h(f32::INFINITY) & 0x7fff, 0x7c00);
    assert_eq!(laya_rust::weights::h2f(laya_rust::weights::f2h(1234.5)), 1234.0);
}

#[test]
fn f16_rounds_within_half_an_ulp() {
    let vals = [
        1e-5f32,
        0.353_553_38,
        3.14159,
        -2.718_28,
        1234.5,
        0.100_582_8,
        1.636_903,
    ];
    for v in vals {
        let back = laya_rust::weights::h2f(laya_rust::weights::f2h(v));
        let tol = (v.abs() * 2f32.powi(-11)).max(2f32.powi(-25));
        assert!(
            (back - v).abs() <= tol,
            "{} -> {} error {} exceeds {}",
            v,
            back,
            (back - v).abs(),
            tol
        );
    }
}

#[test]
fn key_rules() {
    assert!(gguf::key_ok("laya.temperature.0"));
    assert!(gguf::key_ok("general.architecture"));
    assert!(gguf::key_ok("laya.rope_theta.full"));
    assert!(gguf::key_ok("tokenizer.ggml.mask_token_id"));
    assert!(!gguf::key_ok("Laya.temperature"));
    assert!(!gguf::key_ok("laya.temperature-by-options"));
    assert!(!gguf::key_ok("laya.temperature:2"));
    assert!(!gguf::key_ok("laya..temperature"));
    assert!(!gguf::key_ok(".laya"));
    assert!(!gguf::key_ok(""));
}

#[test]
fn tensor_count_and_types() {
    let path = tmp("count.gguf");
    let mut w = gguf::Writer::new();
    w.kv("general.architecture", Value::Str("laya".into()));
    w.tensor("a", &[1024], T_F32, f32_bytes(&vec![1.0; 1024]));
    w.tensor("b", &[3072, 1024], T_F16, gguf::as_f16_bytes(&vec![0.5; 3072 * 1024]));
    w.write(&path);
    let h = gguf::read(&path);
    assert_eq!(h.tensors.len(), 2);
    let b = h.tensor("b").unwrap();
    assert_eq!(b.dims, vec![3072, 1024]);
    assert_eq!(b.nbytes(), 3072 * 1024 * 2);
    assert_eq!(
        h.data_off + h.tensor("a").unwrap().offset,
        h.data_off,
        "first tensor starts at the data section"
    );
    assert_eq!(
        h.tensor("b").unwrap().offset,
        4096,
        "second tensor follows the 4096-byte first tensor, aligned"
    );
    std::fs::remove_file(&path).ok();
}
