"""GATE Fase 0 — smoke test toolchain.

Verifica:
  1. torch vede la GPU (nome, capability sm_XX).
  2. warp inizializza sulla stessa GPU.
  3. interop torch<->warp ZERO-COPY (stesso device, stesso data_ptr).
  4. un kernel warp banale gira e modifica un tensore torch in-place.
"""
import torch
import warp as wp


def main() -> None:
    assert torch.cuda.is_available(), "torch NON vede CUDA"
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"[torch] {torch.__version__} | GPU: {name} | sm_{cap[0]}{cap[1]}")

    wp.init()
    print(f"[warp]  {wp.config.version} | device: {wp.get_device('cuda:0')}")

    # tensore torch su GPU
    t = torch.arange(8, dtype=torch.float32, device=dev)

    # vista warp dello STESSO buffer (zero-copy)
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
