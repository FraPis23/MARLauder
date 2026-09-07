"""Toolchain gate: torch and Warp both see the GPU and share tensors zero-copy.

Checks:
  1. torch sees the GPU (name, sm_XX capability).
  2. Warp initialises on that same GPU.
  3. torch<->Warp interop is ZERO-COPY (same device, same data_ptr).
  4. a trivial Warp kernel runs and mutates a torch tensor in place.
"""
import torch
import warp as wp


def main() -> None:
    assert torch.cuda.is_available(), "torch cannot see CUDA"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[torch] {torch.__version__} | GPU: {name} | sm_{cap[0]}{cap[1]}")

    wp.init()
    print(f"[warp]  {wp.config.version} | device: {wp.get_device('cuda:0')}")

    # tensore torch su GPU
    t = torch.arange(8, dtype=torch.float32, device=dev)

    # Warp view of the SAME buffer (zero-copy)
    a = wp.from_torch(t, dtype=wp.float32)
    assert a.ptr == t.data_ptr(), "interop NON zero-copy (data_ptr diverso)"
    print(f"[interop] zero-copy OK | data_ptr={hex(t.data_ptr())}")

    # kernel warp che raddoppia in-place
    @wp.kernel
    def double(x: wp.array(dtype=wp.float32)):
        i = wp.tid()
        x[i] = x[i] * 2.0

    wp.launch(double, dim=t.numel(), inputs=[a], device="cuda:0")
    wp.synchronize()

    expected = torch.arange(8, dtype=torch.float32, device=dev) * 2.0
    assert torch.allclose(t, expected), f"kernel output errato: {t}"
    print(f"[kernel] double() OK -> {t.tolist()}")
    print("\nGATE Fase 0 PASS: torch+warp su GPU, interop zero-copy, kernel funziona.")


if __name__ == "__main__":
    main()
