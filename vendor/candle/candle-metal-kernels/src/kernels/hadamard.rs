use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Kernels, MetalKernelError, Source};
use crate::{Buffer, ComputeCommandEncoder, Device, MTLSize};
use objc2_metal::MTLResourceUsage;

/// Largest block whose f32 staging area fits the 32 KiB threadgroup memory
/// guaranteed on Apple GPUs.
pub const HADAMARD_MAX_BLOCK: usize = 8192;

/// Fused sign flip + blockwise normalized natural-order FWHT over contiguous
/// f32 data of `n_blocks * block` elements (see `hadamard.metal`). `signs`
/// holds `blocks_per_row * block` values; `signs_first` selects forward
/// (`(x * s) @ H`) vs inverse (`(x @ H) * s`). `src` and `dst` must not alias.
#[allow(clippy::too_many_arguments)]
pub fn call_hadamard_block_fwht(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n_blocks: usize,
    block: usize,
    blocks_per_row: usize,
    signs_first: bool,
    src: BufferOffset,
    signs: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    if block < 2 || !block.is_power_of_two() || block > HADAMARD_MAX_BLOCK {
        return Err(MetalKernelError::FailedToCreateResource(format!(
            "hadamard: block {block} must be a power of two in 2..={HADAMARD_MAX_BLOCK}"
        )));
    }
    if n_blocks == 0 {
        return Ok(());
    }
    let pipeline = kernels.load_pipeline(device, Source::Hadamard, "hadamard_block_fwht_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let scale = 1f32 / (block as f32).sqrt();
    set_params!(
        encoder,
        (
            &src,
            &signs,
            dst,
            block as u32,
            blocks_per_row as u32,
            signs_first,
            scale
        )
    );

    // One butterfly pair per thread per stage; clamp to the pipeline limit
    // (the kernel strides over pairs when it has fewer threads than pairs).
    let max_threads = pipeline.max_total_threads_per_threadgroup();
    let threads = (block / 2).min(1 << max_threads.ilog2());
    let thread_group_count = MTLSize {
        width: n_blocks,
        height: 1,
        depth: 1,
    };
    let thread_group_size = MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };

    encoder.use_resource(src.buffer, MTLResourceUsage::Read);
    encoder.use_resource(signs.buffer, MTLResourceUsage::Read);
    encoder.use_resource(dst, MTLResourceUsage::Write);
    // Metal requires the length to be a multiple of 16 bytes (block = 2 is 8).
    encoder.set_threadgroup_memory_length(0, (block * std::mem::size_of::<f32>()).max(16));
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
