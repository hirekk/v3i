"""Token embedding model."""

import torch
import torch.nn as nn

from v3i.operations import cumulative_cross_product


class OctonionCharEmbedding(nn.Module):
    def __init__(self, vocab_size, device=None) -> None:
        """Args:
        vocab_size (int): number of unique characters.
        device (torch.device, optional): for placement.
        """
        super().__init__()
        self.vocab_size = vocab_size
        # Each character is represented as an 8-dimensional vector (an octonion).
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=8)
        self.device = device if device is not None else torch.device("cpu")

    def forward(self, input_ids):
        """Converts a word (represented as a sequence of character IDs) into a
        single 7-d word embedding computed as the cumulative cross product
        of the characters' imaginary parts.

        Args:
            input_ids (Tensor): shape (batch_size, max_word_length)
                                Each row is a word (sequence of char IDs).
                                (If using padding, you should apply an appropriate mask.)

        Returns:
            Tensor: shape (batch_size, 7) with the word embeddings.
        """
        # Get octonion embeddings: shape (batch_size, max_word_length, 8)
        oct_embeddings = self.embedding(input_ids)
        # Extract the imaginary parts (indices 1 to 7) to yield 7-d vectors.
        imag_parts = oct_embeddings[:, :, 1:]  # shape: (B, L, 7)

        batch_size = imag_parts.shape[0]
        word_embeddings = []
        # Here we iterate over the batch; in production you might vectorize this.
        for i in range(batch_size):
            # For each word, we assume all positions are valid.
            # (If using a padding index, filter out padded positions.)
            word_vector = cumulative_cross_product(imag_parts[i])
            word_embeddings.append(word_vector)

        return torch.stack(word_embeddings, dim=0)


if __name__ == "__main__":
    # Example test: suppose our vocabulary has 50 characters.
    vocab_size = 50
    model = OctonionCharEmbedding(vocab_size=vocab_size)

    # Dummy input: batch of 3 words, each word as a sequence of 4 character IDs.
    dummy_input = torch.randint(0, vocab_size, (3, 4))
    output = model(dummy_input)
