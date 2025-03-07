import torch

def gc_content(batch_sequences):
    """
    Computes the average GC content for a batch of one-hot encoded DNA sequences.
    
    Args:
        batch_sequences (torch.Tensor): Tensor of shape [batch_size, 4, seq_length]
                                        where the 4 channels correspond to A, C, G, and T.

    Returns:
        float: The average GC content across all sequences.
    """
    # Extract C and G channels (indices 1 and 2)
    gc_counts = batch_sequences[:, 1, :].sum(dim=1) + batch_sequences[:, 2, :].sum(dim=1)
    
    # Compute total sequence length (sum over all bases per sequence)
    total_bases = batch_sequences.sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
    
    # Compute GC content per sequence
    gc_content = gc_counts / total_bases
    
    # Compute and return the average GC content as a float
    return gc_content.mean().detach().cpu().item()

