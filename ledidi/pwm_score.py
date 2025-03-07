import numpy as np
import torch

def read_meme_pwm(filename):
    """
    Reads a PWM from a .meme file and returns it as a NumPy array and a PyTorch tensor.
    
    Args:
        filename (str): Path to the .meme file containing the motif.
    
    Returns:
        tuple: (numpy array of PWM, torch tensor of PWM)
    """
    pwm_list = []
    with open(filename, 'r') as file:
        in_matrix_section = False
        
        for line in file:
            line = line.strip()
            if line.startswith("letter-probability matrix"):
                in_matrix_section = True
                continue  # Skip the header line
            
            if in_matrix_section and line:
                pwm_row = [float(value) for value in line.split()]
                pwm_list.append(pwm_row)
            
            if line.startswith("MOTIF") and in_matrix_section:
                break  # Stop reading when the next motif starts
    
    pwm_np = np.array(pwm_list)  # Convert to NumPy array
    pwm_tensor = torch.tensor(pwm_np, dtype=torch.float32)  # Convert to PyTorch tensor
    return pwm_tensor


def pwm_max_score(pwm, batch_sequences):
    """
    Computes the maximum PWM score for each sequence in a batch.
    
    Args:
        pwm (torch.Tensor): Tensor of shape [motif_length, 4] containing PWM.
        batch_sequences (torch.Tensor): One-hot encoded DNA tensor of shape [batch_size, 4, seq_length].
    
    Returns:
        float or list: If batch_size > 10, returns the average max PWM score.
                       Otherwise, returns a list of max PWM scores per sequence.
    """
    # Move PWM to the same device as batch_sequences
    pwm = pwm.to(batch_sequences.device)
    
    motif_length = pwm.shape[0]
    batch_size, _, seq_length = batch_sequences.shape
    max_scores = []

    for i in range(seq_length - motif_length + 1):
        window = batch_sequences[:, :, i:i+motif_length]  # Extract motif-sized windows
        score = torch.sum(pwm.T * window, dim=(1, 2))  # Compute PWM score for each sequence
        max_scores.append(score)

    max_scores = torch.stack(max_scores, dim=1)  # Shape [batch_size, num_positions]
    max_pwm_scores = torch.max(max_scores, dim=1).values  # Max score per sequence

    return max_pwm_scores.mean().detach().cpu().item()  # Average max PWM score

