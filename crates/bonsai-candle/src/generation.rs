use candle::{DType, Error, Result, Tensor};
use rand::{distr::Distribution, SeedableRng};

#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self { rng, sampling }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        logits.argmax(candle::D::Minus1)?.to_scalar::<u32>()
    }

    fn sample_multinomial(&mut self, prs: &[f32]) -> Result<u32> {
        let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(&mut self, prs: &mut [f32], top_p: f32) -> Result<u32> {
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
        argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        self.sample_multinomial(prs)
    }

    fn sample_topk(&mut self, prs: &mut [f32], top_k: usize) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_multinomial(prs)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let index = self.sample_multinomial(&prs)?;
            Ok(indices[index as usize] as u32)
        }
    }

    fn sample_topk_topp(&mut self, prs: &mut [f32], top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= prs.len() {
            self.sample_topp(prs, top_p)
        } else {
            let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
            let (indices, _, _) =
                argsort_indices.select_nth_unstable_by(top_k, |&i, &j| prs[j].total_cmp(&prs[i]));
            let mut prs = indices.iter().map(|&i| prs[i]).collect::<Vec<_>>();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                self.sample_topp(&mut prs, top_p)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits, |_| {})
    }

    pub fn sample_f(&mut self, logits: &Tensor, f: impl FnOnce(&mut [f32])) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Vec<f32>> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            let mut prs = prs.to_vec1()?;
            f(&mut prs);
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let mut prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    self.sample_multinomial(&prs)?
                } else {
                    self.sample_topp(&mut prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk(&mut prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let mut prs = prs(*temperature)?;
                self.sample_topk_topp(&mut prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, Tensor};

    #[test]
    fn argmax_returns_max_logit_index() {
        let mut proc = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let logits = Tensor::from_vec(vec![0.1f32, 0.9, 0.3], 3, &Device::Cpu).unwrap();
        assert_eq!(proc.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn topk_1_is_deterministic() {
        let mut proc = LogitsProcessor::from_sampling(
            0,
            Sampling::TopK {
                k: 1,
                temperature: 1.0,
            },
        );
        let logits = Tensor::from_vec(vec![0.1f32, 5.0, 0.3], 3, &Device::Cpu).unwrap();
        assert_eq!(proc.sample(&logits).unwrap(), 1);
    }

    #[test]
    fn topp_0_falls_back_to_multinomial() {
        // top_p=0 bypasses top-p filtering; should still return a valid token
        let mut proc = LogitsProcessor::from_sampling(
            42,
            Sampling::TopP {
                p: 0.0,
                temperature: 1.0,
            },
        );
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], 3, &Device::Cpu).unwrap();
        let token = proc.sample(&logits).unwrap();
        assert!(token < 3);
    }

    #[test]
    fn all_neg_inf_logits_returns_error() {
        let mut proc = LogitsProcessor::from_sampling(0, Sampling::All { temperature: 1.0 });
        let logits = Tensor::from_vec(vec![f32::NEG_INFINITY; 4], 4, &Device::Cpu).unwrap();
        assert!(proc.sample(&logits).is_err());
    }
}
