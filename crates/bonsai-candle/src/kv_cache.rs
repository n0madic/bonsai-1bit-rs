use candle::{Result, Tensor};

/// KV-Cache using concatenation for append operations.
///
/// Uses `Tensor::cat` instead of `slice_set` — better GPU performance
/// for autoregressive generation on Metal/CUDA.
#[derive(Debug, Clone)]
pub struct ConcatKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    dim: usize,
}

impl ConcatKvCache {
    pub fn new(dim: usize) -> Self {
        Self {
            k: None,
            v: None,
            dim,
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        self.k = Some(match &self.k {
            None => k.clone(),
            Some(k_cache) => Tensor::cat(&[k_cache, &k], self.dim)?,
        });
        self.v = Some(match &self.v {
            None => v.clone(),
            Some(v_cache) => Tensor::cat(&[v_cache, &v], self.dim)?,
        });
        Ok((
            self.k.as_ref().expect("k was just assigned").clone(),
            self.v.as_ref().expect("v was just assigned").clone(),
        ))
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, Tensor};

    #[test]
    fn append_concatenates_along_seq_dim() {
        let dev = &Device::Cpu;
        let mut cache = ConcatKvCache::new(2);
        let k1 = Tensor::zeros((1, 2, 3, 4), DType::F32, dev).unwrap();
        let v1 = Tensor::zeros((1, 2, 3, 4), DType::F32, dev).unwrap();
        let (k_out, v_out) = cache.append(&k1, &v1).unwrap();
        assert_eq!(k_out.dims(), [1, 2, 3, 4]);
        assert_eq!(v_out.dims(), [1, 2, 3, 4]);

        let k2 = Tensor::zeros((1, 2, 1, 4), DType::F32, dev).unwrap();
        let v2 = Tensor::zeros((1, 2, 1, 4), DType::F32, dev).unwrap();
        let (k_out, v_out) = cache.append(&k2, &v2).unwrap();
        assert_eq!(k_out.dims(), [1, 2, 4, 4]);
        assert_eq!(v_out.dims(), [1, 2, 4, 4]);
    }

    #[test]
    fn reset_restores_empty_state() {
        let dev = &Device::Cpu;
        let mut cache = ConcatKvCache::new(2);
        let k = Tensor::zeros((1, 2, 3, 4), DType::F32, dev).unwrap();
        let v = Tensor::zeros((1, 2, 3, 4), DType::F32, dev).unwrap();
        cache.append(&k, &v).unwrap();
        cache.reset();
        let (k_out, _) = cache.append(&k, &v).unwrap();
        assert_eq!(k_out.dims(), [1, 2, 3, 4]);
    }
}
