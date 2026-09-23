pub trait TapSink {
    fn tap(&mut self, stage: &str, data: &[f32]);
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeightFormat {
    F32,
    F16,
    Quant { bits: i32, group: i32 },
}

pub trait Backend {
    type Tensor: Clone;

    fn name(&self) -> &'static str;

    fn sync(&self) {}

    fn upload_f32(&self, data: &[f32], shape: &[usize]) -> Self::Tensor;

    fn download_f32(&self, t: &Self::Tensor) -> Vec<f32>;

    fn weight_format(&self, _name: &str, _dims: &[usize]) -> WeightFormat {
        WeightFormat::F32
    }

    fn resident(&self, _name: &str) -> bool {
        false
    }

    fn upload_f16(&self, _name: &str, data: &[u16], shape: &[usize]) -> Self::Tensor {
        let v: Vec<f32> = data.iter().map(|h| crate::weights::h2f(*h)).collect();
        self.upload_f32(&v, shape)
    }

    fn upload_quant(
        &self,
        _name: &str,
        data: &[f32],
        shape: &[usize],
        _bits: i32,
        _group: i32,
    ) -> Self::Tensor {
        self.upload_f32(data, shape)
    }

    fn embedding(&self, table: &Self::Tensor, ids: &[i64]) -> Self::Tensor;

    fn layernorm(
        &self,
        x: &Self::Tensor,
        w: &Self::Tensor,
        b: Option<&Self::Tensor>,
        eps: f32,
    ) -> Self::Tensor;

    fn matmul(&self, x: &Self::Tensor, w: &Self::Tensor) -> Self::Tensor;

    fn matmul_t(&self, x: &Self::Tensor, w: &Self::Tensor) -> Self::Tensor;

    fn split_qkv_rope(
        &self,
        qkv: &Self::Tensor,
        rope: Option<(&Self::Tensor, &Self::Tensor)>,
        scale: f32,
        nh: usize,
        hd: usize,
        len: usize,
    ) -> (Self::Tensor, Self::Tensor, Self::Tensor);

    fn attention(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        v: &Self::Tensor,
        mask: &Self::Tensor,
        nh: usize,
        len: usize,
    ) -> Self::Tensor;

    fn window_mask(&self, att: &[i64], n: usize, len: usize, window: Option<usize>) -> Self::Tensor;

    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    fn add_bias(&self, x: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;

    fn gelu(&self, x: &Self::Tensor) -> Self::Tensor;

    fn gelu_mul(&self, x: &Self::Tensor, half: usize) -> Self::Tensor;

    fn relu(&self, x: &Self::Tensor) -> Self::Tensor;

    fn add_type(
        &self,
        x: &Self::Tensor,
        type_emb: &Self::Tensor,
        qtype: &[i64],
        len: usize,
    ) -> Self::Tensor;

    fn gather_markers(&self, x: &Self::Tensor, mpos: &[i64], k: usize) -> Self::Tensor;

    fn post(&self, sc: &Self::Tensor, mmask: &[i64], k: usize) -> (Self::Tensor, Self::Tensor);

    fn act_input(&self, x: &Self::Tensor, feats: &Self::Tensor, len: usize) -> Self::Tensor;
}
