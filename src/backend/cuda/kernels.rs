pub const SRC: &str = r#"
#include <cuda_fp16.h>

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float bs[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) bs[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? bs[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) bs[0] = v;
    }
    __syncthreads();
    float r = bs[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float bm[32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) bm[wid] = v;
    __syncthreads();
    int nw = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nw) ? bm[threadIdx.x] : -3.4e38f;
    if (wid == 0) {
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0) bm[0] = v;
    }
    __syncthreads();
    float r = bm[0];
    __syncthreads();
    return r;
}

extern "C" __global__ void k_gather_rows(const long long* __restrict__ ids, const float* __restrict__ tbl,
                                         float* __restrict__ out, int dim) {
    long long r = blockIdx.x;
    const float* src = tbl + ids[r] * (long long)dim;
    float* dst = out + r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void k_layernorm(const float* __restrict__ x, const float* __restrict__ w,
                                       const float* __restrict__ b, float* __restrict__ y,
                                       int dim, float eps) {
    long long r = blockIdx.x;
    const float* xr = x + r * dim;
    float* yr = y + r * dim;
    float s = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    float n = (float)dim;
    float mean = block_sum(s) / n;
    float s2 = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = xr[i] - mean;
        s2 += d * d;
    }
    float rstd = rsqrtf(block_sum(s2) / n + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = (xr[i] - mean) * rstd;
        if (w != 0) v *= w[i];
        if (b != 0) v += b[i];
        yr[i] = v;
    }
}

extern "C" __global__ void k_copy(float* __restrict__ dst, const float* __restrict__ src, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

extern "C" __global__ void k_add(float* __restrict__ a, const float* __restrict__ b, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] += b[i];
}

extern "C" __global__ void k_add_bias(float* __restrict__ x, const float* __restrict__ b, int dim,
                                      long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += b[i % dim];
}

extern "C" __global__ void k_gelu(float* __restrict__ x, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v / 1.4142135381698608f));
}

extern "C" __global__ void k_relu(float* __restrict__ x, long long n) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = fmaxf(x[i], 0.0f);
}

extern "C" __global__ void k_gelu_mul(const float* __restrict__ src, float* __restrict__ dst,
                                      long long rows, long long half) {
    long long stride = 2 * half;
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= rows * half) return;
    long long r = i / half, c = i % half;
    float x = src[r * stride + c];
    dst[i] = 0.5f * x * (1.0f + erff(x / 1.4142135381698608f)) * src[r * stride + half + c];
}

extern "C" __global__ void k_add_type(float* __restrict__ x, const float* __restrict__ te,
                                      const long long* __restrict__ qtype, long long n,
                                      int len, int dim) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long row = i / dim;
    int c = (int)(i % dim);
    x[i] += te[qtype[row / len] * (long long)dim + c];
}

extern "C" __global__ void k_rope_table(const float* __restrict__ freq, float* __restrict__ cs,
                                        float* __restrict__ sn, int seq, int half) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long n = (long long)seq * half;
    if (i >= n) return;
    int p = (int)(i / half), j = (int)(i % half);
    float a = (float)p * freq[j];
    cs[i] = cosf(a);
    sn[i] = sinf(a);
}

extern "C" __global__ void k_split_qkv(const float* __restrict__ qkv, const float* __restrict__ cs,
                                       const float* __restrict__ sn, float* __restrict__ q,
                                       float* __restrict__ k, float* __restrict__ v,
                                       long long n, int len, int nh, int hd, float scale, int rope) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int half = hd >> 1;
    int d = (int)(i % hd);
    int s = (int)((i / hd) % len);
    int h = (int)((i / ((long long)hd * len)) % nh);
    int b = (int)(i / ((long long)hd * len * nh));
    const float* base = qkv + ((long long)b * len + s) * (3 * nh * hd) + h * hd;
    float c = 0.0f, snv = 0.0f;
    if (rope) {
        c = cs[(long long)s * half + (d % half)];
        snv = sn[(long long)s * half + (d % half)];
    }
    float qx = base[d];
    float kx = base[nh * hd + d];
    if (rope) {
        float qr = (d < half) ? -base[d + half] : base[d - half];
        float kr = (d < half) ? -base[nh * hd + d + half] : base[nh * hd + d - half];
        qx = qx * c + qr * snv;
        kx = kx * c + kr * snv;
    }
    q[i] = qx * scale;
    k[i] = kx * scale;
    v[i] = base[2 * nh * hd + d];
}

extern "C" __global__ void k_mask_build(const long long* __restrict__ am, float* __restrict__ mask,
                                        int len, int window) {
    int s = blockIdx.y;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len) return;
    long long b = blockIdx.z;
    float m = 0.0f;
    if (am[b * len + t] == 0) m = -3.4e38f;
    else if (window > 0) {
        int df = s - t;
        if (df < 0) df = -df;
        if (df > window) m = -3.4e38f;
    }
    mask[(b * len + s) * len + t] = m;
}

extern "C" __global__ void k_add_mask(float* __restrict__ attn, const float* __restrict__ mask,
                                      long long n, int len, int nh) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long per = (long long)len * len;
    attn[i] += mask[(i / (per * nh)) * per + (i % per)];
}

extern "C" __global__ void k_softmax(float* __restrict__ x, int cols) {
    float* xr = x + (long long)blockIdx.x * cols;
    float mx = -3.4e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) mx = fmaxf(mx, xr[i]);
    mx = block_max(mx);
    float s = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(xr[i] - mx);
        xr[i] = e;
        s += e;
    }
    float inv = 1.0f / block_sum(s);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) xr[i] *= inv;
}

extern "C" __global__ void k_heads_to_seq(const float* __restrict__ src, float* __restrict__ dst,
                                          long long n, int len, int nh, int hd) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    int d = (int)(i % hd);
    int s = (int)((i / hd) % len);
    int h = (int)((i / ((long long)hd * len)) % nh);
    int b = (int)(i / ((long long)hd * len * nh));
    dst[((long long)b * len + s) * (nh * hd) + h * hd + d] = src[i];
}

extern "C" __global__ void k_marker_gather(const float* __restrict__ x, const long long* __restrict__ mpos,
                                           float* __restrict__ out, int len, int k, int dim) {
    long long r = blockIdx.x;
    long long b = r / k;
    const float* src = x + ((long long)b * len + mpos[r]) * dim;
    float* dst = out + r * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
}

extern "C" __global__ void k_act_input(const float* __restrict__ x, const float* __restrict__ feats,
                                       float* __restrict__ out, long long n, int len, int dim, int nf) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long b = i / (dim + nf);
    int c = (int)(i % (dim + nf));
    out[i] = (c < dim) ? x[((long long)b * len) * dim + c] : feats[b * nf + c - dim];
}

extern "C" __global__ void k_post(const float* __restrict__ sc, const long long* __restrict__ mm,
                                  float* __restrict__ logits, float* __restrict__ feats, int k) {
    extern __shared__ float pb[];
    long long b = blockIdx.x;
    const float* s = sc + b * k;
    const long long* m = mm + b * k;
    float* lg = logits + b * k;
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float v = m[j] ? s[j] : -10000.0f;
        lg[j] = v;
        pb[j] = v;
    }
    __syncthreads();
    float mx = -3.4e38f;
    for (int j = threadIdx.x; j < k; j += blockDim.x) mx = fmaxf(mx, pb[j]);
    mx = block_max(mx);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float e = expf(pb[j] - mx);
        pb[j] = e;
        sum += e;
    }
    float inv = 1.0f / block_sum(sum);
    for (int j = threadIdx.x; j < k; j += blockDim.x) pb[j] *= inv;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t1 = -1.0f, t2 = -1.0f, ent = 0.0f;
        long long nv = 0;
        for (int j = 0; j < k; j++) {
            float p = pb[j];
            if (p > t1) {
                t2 = t1;
                t1 = p;
            } else if (p > t2) {
                t2 = p;
            }
            ent -= p * logf(fmaxf(p, 1e-9f));
            if (m[j]) nv++;
        }
        if (t2 < 0.0f) t2 = 0.0f;
        float n = (float)(nv < 2 ? 2 : nv);
        feats[b * 4 + 0] = t1;
        feats[b * 4 + 1] = t1 - t2;
        feats[b * 4 + 2] = ent / logf(n);
        feats[b * 4 + 3] = n / 255.0f;
    }
}

"#;
