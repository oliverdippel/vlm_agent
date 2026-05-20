import platform
import subprocess

import torch


def print_system_info():
    print(f"Python platform: {platform.platform()}")
    print(f"PyTorch version: {torch.__version__}")
    print()


def check_cuda():
    print("CUDA check:")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  CUDA GPU count: {gpu_count}")

        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  No CUDA GPU detected by PyTorch.")

    print()


def check_mps():
    print("MPS check:")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")

        x = torch.randn(1024, 1024, device=device)
        y = x @ x.T

        print("  MPS working.")
        print(f"  Matrix mul result shape: {y.shape}")
        print(f"  MPS device: {device}")
    else:
        print("  No MPS device available to PyTorch.")

    print()


def check_macos_gpus():
    print("macOS GPU info:")

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except Exception as e:
        print(f"  Could not run system_profiler: {e}")


if __name__ == "__main__":
    print_system_info()
    check_cuda()
    check_mps()
    check_macos_gpus()