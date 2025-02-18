import torch
import torch.nn as nn
from custom_char_embedding import OctonionCharEmbedding

class CustomLanguageModel(nn.Module):
    def __init__(self, char_vocab_size, hidden_dim, num_layers=1):
        """
        Args:
            char_vocab_size (int): number of unique characters (for the embedding layer).
            hidden_dim (int): hidden size for the RNN.
            num_layers (int): number of RNN layers.
        """
        super(CustomLanguageModel, self).__init__()
        self.char_embedding = OctonionCharEmbedding(vocab_size=char_vocab_size)
        # Since our word embedding is 7-d, our RNN will accept 7-dimensional inputs.
        self.rnn = nn.LSTM(input_size=7, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        # For demonstration, we predict (say) the next word from the current word embedding.
        # (In practice, you might have a different head and a larger output space.)
        self.output_layer = nn.Linear(hidden_dim, char_vocab_size)  # Example output dimension.
    
    def forward(self, input_ids):
        """
        Forward pass.
        
        Args:
            input_ids (Tensor): shape (batch_size, max_word_length) where each row is a word.
        Returns:
            Tensor: logits of shape (batch_size, char_vocab_size)
        """
        # Get word embeddings from the custom character encoder.
        word_embed = self.char_embedding(input_ids)  # (B, 7)
        
        # For our RNN, we need a sequence dimension; here we treat each word as a sequence of one embedding.
        x = word_embed.unsqueeze(1)  # shape: (B, 1, 7)
        
        # Pass through the RNN.
        out, _ = self.rnn(x)  # out: (B, 1, hidden_dim)
        # Use the final time step's output.
        out = out[:, -1, :]  # Shape: (B, hidden_dim)
        logits = self.output_layer(out)  # (B, char_vocab_size)
        return logits

if __name__ == "__main__":
    # Dummy test for the language model.
    char_vocab_size = 50  # assume 50 different characters
    hidden_dim = 32
    model = CustomLanguageModel(char_vocab_size=char_vocab_size, hidden_dim=hidden_dim)
    
    # Dummy input: batch of 3 words, each word represented as a sequence of 4 character IDs.
    dummy_input = torch.randint(0, char_vocab_size, (3, 4))
    logits = model(dummy_input)
    print("Logits shape (should be [3, 50]):", logits.shape) 