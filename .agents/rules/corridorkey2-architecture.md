# CorridorKey 2.0 Project Standards

1. **Strict VRAM Lifecycle & Zero Leaks:**
   - Every GPU operation (`CorridorKey 2.0 Inference`, `BiRefNet Alpha`, `SAM 3.1 Tracking`, `MatAnyone2`) MUST offload all submodules (`foundations`, `stage1`, `stage2`, `birefnet`, `sam3`) to `.cpu()` or delete them inside a `try ... finally` block immediately upon completion or cancellation.
   - Always clear intermediate tensor caches (`stage2_frame_cache`, `carry_cache`, `live_acc`, `baton_gpu`, `logits`) and call `gc.collect()` + `torch.cuda.empty_cache()`.

2. **Model Pipeline (No GVM / No VideoMaMa):**
   - Never include `GVM` or `VideoMaMa` in UI widgets, batch pipelines, or backend services.
   - Always load `BiRefNet` models offline (`local_files_only=True`) directly from the local HuggingFace cache (`C:\Users\imtaha\.cache\huggingface\hub\models--ZhengPeng7--*`).
   - Use **SAM 3.1** (`E:\CorridorKey 2.0\backend\weights\sam3\sam3.1-INT8-ConvRot.safetensors` and SAM 3.1 tracker architecture) instead of SAM 2 for point (`1`/`2`) and box (`R`) mask generation and video tracking.

3. **Rust Native Accelerator (`ck2_native`) & Studio Dark UI:**
   - Heavy per-pixel CPU operations (Chroma Key, Despill, Despeckle, Checkerboard compositing, QImage buffer packing) are accelerated via the Rust PyO3/Rayon module `ck2_native` (`E:\CorridorKey 2.0\backend\rust_core`).
   - Parameter panels must expose all native CorridorKey 2.0 parameters (`hint_quality`, `carry_hint`, `carry_hint_quality`, `hann_chunk`, `hann_stride`, `auto_vram`, `low_vram`, `stage2_batch`, `model_scale`, `despill_strength`, `linear_brightness`, `linear_contrast`, `linear_contrast_pivot`, `frame_dir_linear`, `cutout_linear`, `output_format`, `exr_dtype`, `crf`, `preset`, `pix_fmt`, `despeckle`, `garbage_matte`).
   - All hotkeys (`Space`, `I`, `O`, `1`, `2`, `3`, `R`, `E`, `` ` ``, `Left`, `Right`, `Home`, `End`, `F`) must use application-wide shortcuts and child controls must use `Qt.ClickFocus` so keyboard navigation never lags or gets stolen.
