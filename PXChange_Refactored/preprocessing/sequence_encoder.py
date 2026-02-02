"""
Sequence encoding and decoding utilities
"""
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SOURCEID_VOCAB, PAD_TOKEN_ID, START_TOKEN_ID, END_TOKEN_ID


# Create reverse vocabulary for decoding
ID_TO_SOURCEID = {v: k for k, v in SOURCEID_VOCAB.items()}


def encode_sequences(sequence_strings):
    """
    Encode string sequences to token IDs.

    Args:
        sequence_strings: List of sourceID strings or array of strings

    Returns:
        token_ids: Array of token IDs
    """
    if isinstance(sequence_strings, (list, np.ndarray)):
        return np.array([SOURCEID_VOCAB.get(s, SOURCEID_VOCAB['UNK']) for s in sequence_strings])
    else:
        return SOURCEID_VOCAB.get(sequence_strings, SOURCEID_VOCAB['UNK'])


def decode_sequences(token_ids, remove_special_tokens=True):
    """
    Decode token IDs back to sourceID strings.

    Args:
        token_ids: Array or tensor of token IDs
        remove_special_tokens: Whether to remove PAD, START, END tokens

    Returns:
        sequence_strings: List of sourceID strings
    """
    if torch.is_tensor(token_ids):
        token_ids = token_ids.cpu().numpy()

    token_ids = np.atleast_1d(token_ids)

    decoded = []
    for token_id in token_ids:
        token_str = ID_TO_SOURCEID.get(int(token_id), 'UNK')

        if remove_special_tokens:
            if token_str in ['PAD', 'START', 'END']:
                continue

        decoded.append(token_str)

    return decoded


def create_decoder_input(target_tokens, start_token_id=START_TOKEN_ID):
    """
    Create decoder input by prepending START token and removing last token.
    Used for teacher forcing during training.

    Args:
        target_tokens: Target sequence tokens [batch_size, seq_len]
        start_token_id: ID of the START token

    Returns:
        decoder_input: Input for decoder [batch_size, seq_len]
    """
    if torch.is_tensor(target_tokens):
        batch_size, seq_len = target_tokens.shape
        decoder_input = torch.zeros_like(target_tokens)
        decoder_input[:, 0] = start_token_id
        decoder_input[:, 1:] = target_tokens[:, :-1]
    else:
        batch_size, seq_len = target_tokens.shape
        decoder_input = np.zeros_like(target_tokens)
        decoder_input[:, 0] = start_token_id
        decoder_input[:, 1:] = target_tokens[:, :-1]

    return decoder_input


def create_causal_mask(seq_len, device='cpu'):
    """
    Create causal (look-ahead) mask for autoregressive decoding.

    Args:
        seq_len: Sequence length
        device: torch device

    Returns:
        mask: Causal mask [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask


def create_padding_mask(sequence_tokens, pad_token_id=PAD_TOKEN_ID):
    """
    Create padding mask indicating which positions are padding.

    Args:
        sequence_tokens: Token sequences [batch_size, seq_len]
        pad_token_id: ID of padding token

    Returns:
        padding_mask: Boolean mask [batch_size, seq_len] (True = padding)
    """
    if torch.is_tensor(sequence_tokens):
        return sequence_tokens == pad_token_id
    else:
        return sequence_tokens == pad_token_id


def sequence_to_text(token_ids, remove_special=True):
    """
    Convert a sequence of token IDs to a readable text representation.

    Args:
        token_ids: Sequence of token IDs (array, list, or tensor)
        remove_special: Whether to remove special tokens (PAD, START, END)

    Returns:
        text: String representation of the sequence
    """
    tokens = decode_sequences(token_ids, remove_special_tokens=remove_special)
    return " -> ".join(tokens) if tokens else "(empty)"


def calculate_sequence_statistics(dataframe):
    """
    Calculate statistics about sequences in the dataset.

    Args:
        dataframe: DataFrame with sequences

    Returns:
        stats: Dictionary of statistics
    """
    seq_lengths = dataframe.groupby('SeqOrder').size()
    token_counts = dataframe['sourceID'].value_counts()

    stats = {
        'num_sequences': len(seq_lengths),
        'mean_length': seq_lengths.mean(),
        'std_length': seq_lengths.std(),
        'min_length': seq_lengths.min(),
        'max_length': seq_lengths.max(),
        'median_length': seq_lengths.median(),
        'token_distribution': token_counts.to_dict()
    }

    return stats


def print_sequence_examples(dataframe, num_examples=5):
    """
    Print examples of sequences from the dataset.

    Args:
        dataframe: DataFrame with sequences
        num_examples: Number of examples to print
    """
    print(f"\n{'='*80}")
    print(f"SEQUENCE EXAMPLES")
    print(f"{'='*80}\n")

    seq_orders = dataframe['SeqOrder'].unique()[:num_examples]

    for i, seq_order in enumerate(seq_orders, 1):
        seq_data = dataframe[dataframe['SeqOrder'] == seq_order].sort_values('Step')

        tokens = seq_data['sourceID'].values
        durations = seq_data.get('step_duration', seq_data['timediff']).values

        print(f"Example {i} (SeqOrder={seq_order}):")
        print(f"  Length: {len(tokens)} steps")
        print(f"  Sequence: {sequence_to_text(tokens, remove_special=False)}")

        if 'true_total_time' in seq_data.columns:
            print(f"  Total time: {seq_data['true_total_time'].iloc[0]:.1f}s")

        print(f"  Step durations (first 10): {durations[:10]}")
        print()


if __name__ == "__main__":
    # Test encoding/decoding
    print("Testing sequence encoder...")

    # Test vocabulary
    print(f"\nVocabulary size: {len(SOURCEID_VOCAB)}")
    print(f"Special tokens: START={START_TOKEN_ID}, END={END_TOKEN_ID}, PAD={PAD_TOKEN_ID}")

    # Test encoding
    test_sequence = ['START', 'MRI_CCS_11', 'MRI_FRR_18', 'END']
    encoded = encode_sequences(test_sequence)
    print(f"\nOriginal: {test_sequence}")
    print(f"Encoded: {encoded}")

    # Test decoding
    decoded = decode_sequences(encoded, remove_special_tokens=False)
    print(f"Decoded: {decoded}")

    # Test decoder input creation
    test_tokens = torch.tensor([[11, 1, 3, 14, 0, 0]])  # START, token, token, END, PAD, PAD
    decoder_input = create_decoder_input(test_tokens)
    print(f"\nTarget tokens: {test_tokens}")
    print(f"Decoder input: {decoder_input}")

    # Test causal mask
    causal_mask = create_causal_mask(5)
    print(f"\nCausal mask (5x5):")
    print(causal_mask.int())
