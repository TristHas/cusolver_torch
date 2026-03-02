import torch

from cusolver_torch import eigh


def main():
    assert torch.cuda.is_available(), "CUDA is required"
    a = torch.randn(8, 64, 64, device="cuda", dtype=torch.float32)
    a = (a + a.transpose(-1, -2)) * 0.5

    w, v, info = eigh(a, driver="xsyev_batched", compute_vectors=True)
    torch.cuda.synchronize()

    nz = int((info.detach().cpu() != 0).sum().item())
    rel_res = float(((a @ v - v * w.unsqueeze(-2)).norm() / a.norm()).item())
    print(f"ok: w={tuple(w.shape)} v={tuple(v.shape)} nonzero_info={nz} rel_res={rel_res:.3e}")


if __name__ == "__main__":
    main()
