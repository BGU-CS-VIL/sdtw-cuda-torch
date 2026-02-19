"""
SoftDTW Barycenter Example

This example demonstrates how to compute a time series average (barycenter)
using SoftDTW geometry and visualize the results.

The barycenter is computed by minimizing the sum of SoftDTW distances to all
input time series, which captures the "average" shape in DTW space rather than
point-wise Euclidean averaging.

Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for Time-Series"
           ICML 2017
"""

import sys
import time
import argparse
import torch
import matplotlib.pyplot as plt
from softdtw_cuda import softdtw_barycenter


def generate_example_sequences(num_sequences: int = 30, seq_len: int = 100) -> torch.Tensor:
    """
    Generate example time series with variations of single block waves.

    Each sequence has a single block with different height and width,
    representing similar patterns with varying block characteristics.
    """
    sequences = []

    for i in range(num_sequences):
        # Create block wave for this sequence
        sequence = torch.zeros(seq_len)

        # Random block width and height
        block_width = torch.randint(30, 100, (1,))  
        block_height = (torch.rand(1) * 1.5 + 0.5)  

        # Random position relative to middle (center-based with offset)
        center = seq_len // 2
        max_offset = 20
        offset = torch.randint(-max_offset, max_offset + 1, (1,))
        block_start = max(0, center + offset - block_width // 2)
        block_end = min(seq_len, block_start + block_width)
        sequence[block_start:block_end] = block_height

        # Add slight scaling variation
        scale = 0.8 + (i * 0.1)
        scaled = sequence * scale

        # Add small noise
        noisy = scaled + 0.1 * torch.randn_like(scaled)

        sequences.append(noisy)

    # Stack into batch: (num_sequences, seq_len, 1)
    X = torch.stack(sequences, dim=0).unsqueeze(-1)
    return X


def plot_barycenter_results(X: torch.Tensor, barycenter: torch.Tensor):
    """
    Plot original sequences and their computed barycenter in two subplots.

    Args:
        X: Input sequences of shape (B, N, D)
        barycenter: Computed barycenter of shape (N, D)
    """
    B, N, D = X.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Original sequences
    ax = axes[0]
    for i in range(B):
        ax.plot(X[i, :, 0].numpy(), alpha=0.6, label=f"Sequence {i+1}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Original Sequences")
    ax.grid(True, alpha=0.3)

    # Right subplot: Sequences + barycenter
    ax = axes[1]
    for i in range(B):
        ax.plot(X[i, :, 0].numpy(), alpha=0.4, color="gray", label="Input sequences" if i == 0 else "")
    # Add Euclidean mean for comparison (optional)
    euclidean_mean = X.mean(dim=0)
    ax.plot(
        euclidean_mean[:, 0].numpy(),
        linewidth=2.5,
        color="blue",
        linestyle="--",
        label="Euclidean Mean",
    )
    # SoftDTW barycenter
    ax.plot(
        barycenter[:, 0].numpy(),
        linewidth=2.5,
        color="red",
        label="SoftDTW Barycenter",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Barycenter (SoftDTW Average)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("softdtw_barycenter_example.png", dpi=100, bbox_inches="tight")
    print("✓ Plot saved as 'softdtw_barycenter_example.png'")
    plt.show()


def main(compare_modes: bool = False):
    """
    Main example: Generate sequences, compute barycenter, and visualize.

    Args:
        compare_modes: If True, compare runtime between fused and unfused modes
    """
    print("=" * 60)
    print("SoftDTW Barycenter Example")
    if compare_modes:
        print("(Fused vs Unfused Comparison Mode)")
    print("=" * 60)

    # Generate example time series
    X = generate_example_sequences(num_sequences=20, seq_len=200)
    print(f"   Input shape: {X.shape} (batch, sequence_length, features)")

    # Compute barycenter with timing
    if compare_modes:
        print("\n2. Computing SoftDTW barycenter (UNFUSED mode)...")

        start_time = time.time()
        barycenter_unfused = softdtw_barycenter(
            X,
            gamma=1.0,
            max_iter=200,
            lr=0.1,
            fused=False,
            verbose=False,
        )
        elapsed_unfused = time.time() - start_time

        print("3. Computing SoftDTW barycenter (FUSED mode)...")

        # Move data to CUDA for fused mode if available
        X_cuda = X.to("cuda" if torch.cuda.is_available() else "cpu")

        start_time = time.time()
        barycenter_fused = softdtw_barycenter(
            X_cuda,
            gamma=1,
            max_iter=200,
            lr=0.1,
            fused=True,
            verbose=False,
        ).cpu()
        elapsed_fused = time.time() - start_time

        # Use unfused for visualization
        barycenter = barycenter_unfused
    else:
        print("\n2. Computing SoftDTW barycenter...")

        # Always use CUDA if available (both fused and unfused support it)
        cuda_available = torch.cuda.is_available()
        X_compute = X.to("cuda" if cuda_available else "cpu")
        # Use fused mode on CUDA, fall back to auto-select on CPU
        use_fused = True if cuda_available else None
        start_time = time.time()
        barycenter = softdtw_barycenter(
            X_compute,
            gamma=10.0,
            max_iter=200,
            lr=0.1,
            fused=use_fused,
            verbose=True,
        )
        elapsed_time = time.time() - start_time

        # Move back to CPU for visualization
        if cuda_available:
            barycenter = barycenter.cpu()

        print(f"   Barycenter shape: {barycenter.shape}")
        print(f"   ⏱️  Total optimization time: {elapsed_time:.3f} seconds")
        print(f"   ⏱️  Time per iteration: {(elapsed_time / 100):.3f} seconds")
        print("   ✓ Barycenter computed!")

    # Visualization
    print(f"\n{'4' if compare_modes else '3'}. Creating visualization...")
    plot_barycenter_results(X, barycenter)

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)

    if not compare_modes:
        print("\n✨ Timing Summary:")
        print(f"  • Barycenter optimization: {elapsed_time:.3f}s")
        print(f"  • Per-iteration cost: {(elapsed_time / 100):.3f}s")
    else:
        # Calculate speedup correctly
        speedup_ratio = elapsed_unfused / elapsed_fused if elapsed_fused > 0 else float('inf')
        faster_mode = "fused" if elapsed_fused < elapsed_unfused else "unfused"

        print("\n✨ Performance Comparison Summary:")
        print(f"  • Unfused mode:  {elapsed_unfused:.3f}s ({elapsed_unfused/50*1000:.1f}ms/iter)")
        print(f"  • Fused mode:    {elapsed_fused:.3f}s ({elapsed_fused/50*1000:.1f}ms/iter)")
        print(f"  • Speed ratio:   {speedup_ratio:.2f}× ({faster_mode} is faster)")
        print(f"\n  💡 Memory vs. Speed Trade-off:")
        print(f"     - Fused: saves 98% memory, {speedup_ratio:.1f}× {'slower' if speedup_ratio < 1 else 'faster'}")
        print(f"     - Unfused: standard memory, optimal speed")

    print("\nKey Points:")
    print("  • The barycenter (red line) represents the 'average' shape in DTW space")
    print("  • Unlike Euclidean averaging, DTW averaging aligns sequences first")
    print("  • This is useful for time series alignment and pattern discovery")
    print("  • The regularization parameter 'gamma' controls smoothness")
    print("    - Small gamma: closer to true DTW (less smooth)")
    print("    - Large gamma: smoother, more Euclidean-like")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SoftDTW Barycenter Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python barycenter_example.py              # Run basic example
  python barycenter_example.py --compare   # Compare fused vs unfused runtime
        """,
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare runtime between fused and unfused modes",
    )

    args = parser.parse_args()
    main(compare_modes=args.compare)
