"""
Example: Extract a circuit from a simple transformer model

This demonstrates the full extraction pipeline on a toy model.
"""

import torch
import torch.nn as nn
from circuit_extractor import extract_transformer_circuit


class SimpleTransformer(nn.Module):
    """
    A minimal transformer for demonstration purposes
    """

    def __init__(self, d_model=64, num_heads=4, vocab_size=100, max_seq_len=32):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Single attention layer (simplified)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward layer
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Args:
            x: Token indices (batch_size, seq_len)

        Returns:
            Logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        hidden = token_emb + pos_emb

        # Attention block
        attn_output, _ = self.attention(hidden, hidden, hidden)
        hidden = self.norm1(hidden + attn_output)

        # MLP block
        mlp_output = self.mlp(hidden)
        hidden = self.norm2(hidden + mlp_output)

        # Output
        logits = self.output(hidden)

        return logits


def generate_synthetic_data(vocab_size=100, seq_len=16, num_samples=100):
    """Generate synthetic data for circuit extraction"""
    # Random token sequences
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))

    # Random targets (next token prediction)
    targets = torch.randint(0, vocab_size, (num_samples, seq_len, vocab_size)).float()

    return inputs, targets


def main():
    print("=" * 60)
    print("Certified Proof-Carrying Circuits - Extraction Demo")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create model
    print("\n1. Creating simple transformer model...")
    model = SimpleTransformer(d_model=64, num_heads=4, vocab_size=100, max_seq_len=32)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Generate synthetic data
    print("\n2. Generating synthetic calibration and test data...")
    calibration_inputs, calibration_targets = generate_synthetic_data(num_samples=100)
    test_inputs, test_targets = generate_synthetic_data(num_samples=50)
    print(f"   Calibration data: {calibration_inputs.shape}")
    print(f"   Test data: {test_inputs.shape}")

    # Extract circuit with different pruning thresholds
    thresholds = [0.1, 0.05, 0.01]

    for threshold in thresholds:
        print(f"\n3. Extracting circuit with pruning threshold {threshold}...")
        print("   (This may take a few moments...)")

        try:
            circuit_data = extract_transformer_circuit(
                model=model,
                calibration_data=calibration_inputs,
                calibration_targets=calibration_targets,
                test_data=test_inputs,
                test_targets=test_targets,
                output_path=f"extracted_circuit_thresh_{threshold}.json",
                pruning_threshold=threshold
            )

            print(f"\n   ✓ Circuit extraction complete!")
            print(f"   Error bound (ε): {circuit_data['error_bound']['epsilon']:.6f}")
            print(f"   MAE: {circuit_data['error_bound']['mae']:.6f}")
            print(f"   Max error: {circuit_data['error_bound']['max_error']:.6f}")
            print(f"   Coverage: {circuit_data['error_bound']['coverage']:.2%}")
            print(f"   Sparsity: {circuit_data['metadata']['sparsity']:.2%}")
            print(f"   Components: {circuit_data['metadata']['num_components']}")

        except Exception as e:
            print(f"   ✗ Error during extraction: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Extraction demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
