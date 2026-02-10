# ledidi.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          adapted from code written by Yang Lu

import time
import torch

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from semifreddo_full_model import Semifreddo
from semifreddo_full_v2_model import Semifreddo
# from ledidi.gc_calculation import gc_content
# from ledidi.pwm_score import read_meme_pwm_as_numpy
# from tangermeme.tools import fimo


def batch_pearsonr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes Pearson R for batched input.
    x, y: shape (batch_size, N)
    Returns: shape (batch_size,)
    """
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    
    vx = x - x_mean
    vy = y - y_mean
    
    numerator = torch.sum(vx * vy, dim=1)
    denominator = torch.sqrt(torch.sum(vx ** 2, dim=1) * torch.sum(vy ** 2, dim=1)) + 1e-8
    return numerator / denominator


class DesignWrapper(torch.nn.Module):
    """A wrapper for using multiple models in design.

    This wrapper will accept multiple models and turn their predictions into a
    vector. A requirement is that the output from each individual model is a
    tensor whose last dimension is 1, and that all of the models have the same
    other dimensions. For instance, if three models are passed in that each make 
    predictions of shape (batch_size, 1), the return from this wrapper would have
    shape (batch_size, 3).

    This wrapper is used to design edits when you want to balance the predictions
    from several models, e.g., by increasing predictions from one model without
    changing predictions from a second model. In practice, one would now pass in
    a vector of desired targets instead of a single value and the loss would be
    calculated over each of them.


    Parameters
    ----------
    models: list, tuple
        A set of torch.nn.Module objects.
    """

    def __init__(self, models):
        super(DesignWrapper, self).__init__()
        self.models = models
    
    def forward(self, X):
        return torch.cat([model(X) for model in self.models], dim=-1)


class Ledidi(torch.nn.Module):
    """Ledidi is a method for editing categorical sequences.

    Ledidi is a method for editing categorical sequences, such as those
    comprised of nucleotides or amino acids, to exhibit desired properties in
    a small number of edits. It does so through the use of an oracle model,
    which is a differentiable model that accepts a categorical sequence as
    input and makes relevant predictions. For instance, the model might take
    in one-hot encoded nucleotide sequence and predict the strength of binding 
    for a particular transcription factor. 

    Given a sequence and a desired output, Ledidi uses gradient descent to 
    design edits that bring the predicted output from the model closer to the
    desired output. Because the sequences that predictions are being made for
    must be categorical this involves using the Gumbel-softmax 
    reparameterization trick.


    Parameters
    ----------
    model: torch.nn.Module
        A model to use as an oracle that will be frozen as a part of the
        Ledidi procedure.

    target: int or None
        When given a multi-task model, the target to slice out and feed into
        output_loss when calculating the gradient. If None, perform no slicing.
        Default is None.

    input_loss: torch.nn.Loss, optional
        A loss to apply to the input space. By default this is the L1 loss
        which corresponds to the number of positions that have been edited.
        This loss is also divided by 2 to account for each edit changing
        two values within that position. Default is torch.nn.L1Loss.

    output_loss: torch.nn.Loss, optional
        A loss to apply to the output space. By default this is the L2 loss
        which corresponds to the mean squared error between the predicted values
        and the desired values.

    tau: float, positive, optional
        The sharpness of the sampled values from the Gumbel distribution used
        to generate the one-hot encodings at each step. Higher values mean
        sharper, i.e., more closely match the argmax of each position.
        Default is 1.

    l: float, positive, optional
        The mixing weight parameter between the input loss and the output loss,
        applied to the input loss. The smaller this value is the more important
        it is that the output loss is minimized. Default is 0.01.

    batch_size: int, optional
        The number of sequences to generate at each step and average loss over. 
        Default is 64.

    max_iter: int, optional
        The maximum number of iterations to continue generating samples.
        Default is 1000.

    report_iter: int optional
        The number of iterations to perform before reporting results of the
        optimization. Default is 100.

    lr: float, optional
        The learning rate of the procedure. Default is 0.1.

    input_mask: torch.Tensor or None, shape=(shape[-1],)
        A mask where 1 indicates what positions cannot be edited. This will 
        set the initial weights mask to -inf at those positions. If None, no 
        positions are masked out. Default is None.

    initial_weights: torch.Tensor or None, shape=(1, shape[0, shape[1])
        Initial weights to use in the weight matrix to specify priors in the
        composition of edits that can be made. Positive values mean more likely
        that certain edits are proposed, negative values mean less likely that
        those edits are proposed.

    eps: float, optional
        The epsilon to add to the one-hot encoding. Because the first step
        of the procedure is to take log(X + eps) the smaller eps is the
        higher a value in the design weight needs to be achieved before
        an edit can be induced. Default is 1e-4.

    random_state: int or None, optional
        Whether to force determinism.

    verbose: bool, optional
        Whether to print the loss during design. Default is True.
    """

    def __init__(self, 
                 models_list, 
                 input_loss=torch.nn.L1Loss(reduction='sum'), 
                 output_loss=torch.nn.MSELoss(), 
                 tau=1, 
                 l=0.1, 
                 lr=1.0, 
                 eps=1e-4, 
                 batch_size=10, 
                 max_iter=1000, 
                 early_stopping_iter=100, 
                 report_iter=100,  
                 return_history=False, 
                 verbose=True, 
                 num_channels=4, 
                 bin_size=2048, 
                 input_mask_slices_0=[224], 
                 input_mask_slices_1=None,
                 cropping_applied=32,
                 output_mask_path=None,
                 use_semifreddo=False,
                 semifreddo_temp_output_path_list=None,
                 g=50.0,
                 punish_ctcf=False,
                 ctcf_meme_path=None,
                 suppressing_mask=None):
        super().__init__()
        
        self.models_list = models_list
        
        # for 4 models for now
        model0, model1, model2, model3 = models_list
        
        for param in model0.parameters():
            param.requires_grad = False
            
        for param in model1.parameters():
            param.requires_grad = False
            
        for param in model2.parameters():
            param.requires_grad = False
            
        for param in model3.parameters():
            param.requires_grad = False
        
        # model in eval mode
        model0 = model0.eval()
        model1 = model1.eval()
        model2 = model2.eval()
        model3 = model3.eval()
        
        self.input_loss = input_loss
        self.output_loss = output_loss
        self.tau = tau
        self.l = l
        self.lr = lr
        self.eps = eps
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping_iter = early_stopping_iter
        self.report_iter = report_iter
        self.return_history = return_history
        self.verbose = verbose
        self.num_channels = num_channels
        self.bin_size =  bin_size
        self.input_mask_slices_0 = input_mask_slices_0 
        self.input_mask_slices_1 = input_mask_slices_1
        self.cropping_applied = cropping_applied
        self.output_mask_path = output_mask_path
        self.use_semifreddo = use_semifreddo
        self.semifreddo_temp_output_path_list = semifreddo_temp_output_path_list
        self.g = g
        self.punish_ctcf = punish_ctcf
        self.ctcf_meme_path = ctcf_meme_path
        self.suppressing_mask = suppressing_mask
        
        if (self.punish_ctcf == True) and (self.ctcf_meme_path is None):
            print("Please, provide a path to the CTCF motif in the meme format.")
        
        # if self.ctcf_meme_path is not None:
        #     self.ctcf_pwm = read_meme_pwm_as_numpy(self.ctcf_meme_path)
        # else:
        #     self.ctcf_pwm = None
        
        self.slice_0_length = len(self.input_mask_slices_0) * self.bin_size
        
        self.weights_0 = torch.nn.Parameter(
            torch.zeros((1, self.num_channels, self.slice_0_length), dtype=torch.float32, requires_grad=True)
        )
        
        # optionally, the second slice is optimized in parallel
        if self.input_mask_slices_1 is not None:
            self.slice_1_length = len(self.input_mask_slices_1) * self.bin_size
            
            self.weights_1 = torch.nn.Parameter(
                torch.zeros((1, self.num_channels, self.slice_1_length), dtype=torch.float32, requires_grad=True)
            )
        
        print("Model 0 in train mode:", model0.training)
        print("Model 1 in train mode:", model1.training)
        print("Model 2 in train mode:", model2.training)
        print("Model 3 in train mode:", model3.training)
        
        print("Gradients enabled for weights - slice 0:", self.weights_0.requires_grad)
        print("Weights shape - slice 0:", self.weights_0.shape)
        
        if self.input_mask_slices_1 is not None:
            print("Gradients enabled for weights - slice 1:", self.weights_1.requires_grad)
            print("Weights shape - slice 1:", self.weights_1.shape)

        
    def apply_freeze_mask(self, weights, X, suppressing_mask):
        with torch.no_grad():
            for pos in torch.where(suppressing_mask)[0]:
                orig_nt = torch.argmax(X[0, :, pos]).item()
                weights[0, :, pos] = -1e4  # suppress all
                weights[0, orig_nt, pos] = 0.0  # keep original
    
        
    def forward(self, X, X1=None, padding_bins=2):
        """Generate a set of edits given a sequence.

        This method will take in the one-hot encoded sequence and the current
        learned weight filter and propose edits based on the Gumbel-softmax
        distribution.


        Parameters
        ----------
        X: torch.Tensor, shape=(1, n_channels, length)
            A tensor containing a single one-hot encoded sequence to propose 
            edits for. This sequence is then expanded out to the desired batch 
            size to generate a batch of edits.


        Returns
        -------
        y: torch.Tensor, shape=(batch_size, n_channels, length)
            A tensor containing a batch of one-hot encoded sequences which
            may contain one or more edits compared to the sequence that was
            passed in. 
        """        
        
        if self.use_semifreddo:
            padding_bp = padding_bins * self.bin_size
            # slice 0
            
            if self.suppressing_mask is not None:
                self.apply_freeze_mask(self.weights_0, X[:, :, padding_bp:-padding_bp], self.suppressing_mask)       
            
            logits = torch.log(X[:, :, padding_bp:-padding_bp] + self.eps) + self.weights_0
            
            # slice 1
            if X1 is not None:
                logits_1 = torch.log(X1[:, :, padding_bp:-padding_bp] + self.eps) + self.weights_1
            
        else:
            start_slice = (min(self.input_mask_slices_0) + self.cropping_applied) * self.bin_size
            end_slice = (max(self.input_mask_slices_0) + 1 + self.cropping_applied) * self.bin_size
            
            logits = torch.log(X[:, :, start_slice:end_slice] + self.eps) + self.weights_0
            
            if self.input_mask_slices_1 is not None:
                start_slice_1 = (min(self.input_mask_slices_1) + self.cropping_applied) * self.bin_size
                end_slice_1 = (max(self.input_mask_slices_1) + 1 + self.cropping_applied) * self.bin_size
                
                logits_1 = torch.log(X[:, :, start_slice_1:end_slice_1] + self.eps) + self.weights_1
        
        # Expand X to create batch_size copies
        X_expanded = X.expand(self.batch_size, *X.shape[1:])
        
        logits = logits.expand(self.batch_size, -1, -1)
        edited_slice = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1)
        
        # Create a batch of modified sequences
        X_hat = X_expanded.clone()
        
        if self.input_mask_slices_1 is not None:            
            logits_1 = logits_1.expand(self.batch_size, -1, -1)
            edited_slice_1 = torch.nn.functional.gumbel_softmax(logits_1, tau=self.tau, hard=True, dim=1)

            if X1 is not None:
                X1_expanded = X1.expand(self.batch_size, *X1.shape[1:])
                X1_hat = X1_expanded.clone()
        
        if self.use_semifreddo:
            X_hat_slice_0 = X_hat.clone()
            X_hat_slice_0[:, :, padding_bp:-padding_bp] = edited_slice
            
            if self.input_mask_slices_1 is  None:
                return X_hat_slice_0
            
            elif X1 is not None:
                X_hat_slice_1 = X1_hat.clone()
                X_hat_slice_1[:, :, padding_bp:-padding_bp] = edited_slice_1
            
                return X_hat_slice_0, X_hat_slice_1
            
        else:
            X_hat[:, :, start_slice:end_slice] = edited_slice
            
            if self.input_mask_slices_1 is not None:
                X_hat[:, :, start_slice_1:end_slice_1] = edited_slice_1
            
            return X_hat
        

    def fit_transform(self, X, y_bar_list, X1=None):
        """Apply the Ledidi procedure to design edits for a sequence.

        This procedure takes in a single sequence and a desired output from
        the model and designs edits that cause the model to predict the desired
        output. This is done primarily by learning a weight matrix of logits
        that can be added the log'd one-hot encoded sequence. These weights
        are the only weights learned during the procedure.


        Parameters
        ----------
        X: torch.Tensor, shape=(1, n_channels, length)
            A tensor containing a single one-hot encoded sequence to propose 
            edits for. This sequence is then expanded out to the desired batch 
            size to generate a batch of edits.

        y_bar: torch.Tensor, shape=(1, *)
            The desired output from the model. Any shape for this tensor is
            permissable so long as the `output_loss` function can handle
            comparing it to the output from the given model.


        Returns
        -------
        y: torch.Tensor, shape=(batch_size, n_channels, length)
            A tensor containing a batch of one-hot encoded sequences which
            may contain one or more edits compared to the sequence that was
            passed in.
        """
        
        model0, model1, model2, model3 = self.models_list
        
        y_bar0, y_bar1, y_bar2, y_bar3 = y_bar_list
        
        optimizer = torch.optim.AdamW((self.weights_0,), lr=self.lr)
        
        if self.input_mask_slices_1 is not None:
            optimizer_1 = torch.optim.AdamW((self.weights_1,), lr=self.lr)
        
        # history = {'edits': [], 'input_loss': [], 'output_loss': [], 
        #     'total_loss': [], 'batch_size': self.batch_size}

        if self.ctcf_meme_path is None:
            history = {'input_loss': [], 'output_loss': [], 
                'total_loss': [], 'gc_content': [], 'batch_size': self.batch_size,
                'edit_positions': []}
        else:
            history = {'input_loss': [], 'output_loss': [], 
                'total_loss': [], 'gc_content': [], 'ctcf_fimo_sum_score': [], 'batch_size': self.batch_size,
                'edit_positions': []}
        
        # inpainting_mask - ensures only the valid positions
        # are taken into account while input_loss is calculates
        inpainting_mask = X[0].sum(dim=0) == 1

        if (self.input_mask_slices_1 is not None) and (self.use_semifreddo == True):
            inpainting_mask_1 = X1[0].sum(dim=0) == 1
        
        # prediction for the input sequence
        if self.use_semifreddo:
            semifreddo_model0 = Semifreddo(model=model0,
                                          slice_0_padded_seq=X, 
                                          edited_indices_slice_0=self.input_mask_slices_0,
                                          saved_temp_output_path=self.semifreddo_temp_output_path_list[0],
                                          slice_1_padded_seq=X1,
                                          edited_indices_slice_1=self.input_mask_slices_1,
                                          batch_size=1,
                                          cropping_applied=self.cropping_applied)
            y_hat0 = semifreddo_model0.forward()
            
            semifreddo_model1 = Semifreddo(model=model1,
                                          slice_0_padded_seq=X, 
                                          edited_indices_slice_0=self.input_mask_slices_0,
                                          saved_temp_output_path=self.semifreddo_temp_output_path_list[1],
                                          slice_1_padded_seq=X1,
                                          edited_indices_slice_1=self.input_mask_slices_1,
                                          batch_size=1,
                                          cropping_applied=self.cropping_applied)
            y_hat1 = semifreddo_model1.forward()
            
            semifreddo_model2 = Semifreddo(model2,
                                          slice_0_padded_seq=X, 
                                          edited_indices_slice_0=self.input_mask_slices_0,
                                          saved_temp_output_path=self.semifreddo_temp_output_path_list[2],
                                          slice_1_padded_seq=X1,
                                          edited_indices_slice_1=self.input_mask_slices_1,
                                          batch_size=1,
                                          cropping_applied=self.cropping_applied)
            y_hat2 = semifreddo_model2.forward()
            
            semifreddo_model3 = Semifreddo(model3,
                                          slice_0_padded_seq=X, 
                                          edited_indices_slice_0=self.input_mask_slices_0,
                                          saved_temp_output_path=self.semifreddo_temp_output_path_list[3],
                                          slice_1_padded_seq=X1,
                                          edited_indices_slice_1=self.input_mask_slices_1,
                                          batch_size=1,
                                          cropping_applied=self.cropping_applied)
            y_hat3 = semifreddo_model3.forward()
            
            # y_hat = torch.stack([y_hat0, y_hat1, y_hat2, y_hat3], dim=0).mean(dim=0)
            
        else:
            y_hat = self.model(X)
        # y_hat = y_hat.squeeze(1)
        
        n_iter_wo_improvement = 0
        
        # forcing all variables to float32    
        y_hat0 = y_hat0.float()
        y_hat1 = y_hat1.float()
        y_hat2 = y_hat2.float()
        y_hat3 = y_hat3.float()
        
        y_bar0 = y_bar0.float()
        y_bar1 = y_bar1.float()
        y_bar2 = y_bar2.float()
        y_bar3 = y_bar3.float()
        
        # loss between the prediction of the original sequence
        # and the desired prediction (aka starting loss)  
        if self.output_mask_path is not None:
            # LOCAL LOSS
            print("Local loss applied.")
            loaded_unmask_indices = torch.load(self.output_mask_path, weights_only=True)
            loaded_unmask_indices = loaded_unmask_indices.to(dtype=torch.long, device=y_hat0.device)
            
            y_hat0_unmasked = y_hat0[..., loaded_unmask_indices]
            y_hat1_unmasked = y_hat1[..., loaded_unmask_indices]
            y_hat2_unmasked = y_hat2[..., loaded_unmask_indices]
            y_hat3_unmasked = y_hat3[..., loaded_unmask_indices]
            
            y_bar0_unmasked = y_bar0[..., loaded_unmask_indices]
            y_bar1_unmasked = y_bar1[..., loaded_unmask_indices]
            y_bar2_unmasked = y_bar2[..., loaded_unmask_indices]
            y_bar3_unmasked = y_bar3[..., loaded_unmask_indices]
            
            scaling_factor = y_hat0.shape[-1] // y_hat0_unmasked.shape[-1]
            
            y_hat0_unmasked = y_hat0_unmasked.to(device=y_bar0_unmasked.device)
            y_hat1_unmasked = y_hat1_unmasked.to(device=y_bar1_unmasked.device)
            y_hat2_unmasked = y_hat2_unmasked.to(device=y_bar2_unmasked.device)
            y_hat3_unmasked = y_hat3_unmasked.to(device=y_bar3_unmasked.device)
            
            loss0 = self.output_loss(y_hat0_unmasked, y_bar0_unmasked)
            loss1 = self.output_loss(y_hat1_unmasked, y_bar1_unmasked)
            loss2 = self.output_loss(y_hat2_unmasked, y_bar2_unmasked)
            loss3 = self.output_loss(y_hat3_unmasked, y_bar3_unmasked)

            output_loss = torch.stack([loss0, loss1, loss2, loss3]).max() * scaling_factor
            
        else:
            # GLOBAL LOSS
            print("Global loss applied.")
            output_loss = self.output_loss(y_hat, y_bar)
        
        best_input_loss = 0.0
        best_output_loss = output_loss
        best_total_loss = output_loss
        best_sequence = X
        if X1 is not None:
            best_sequence_1 = X1
        best_weights_0 = torch.clone(self.weights_0)
        last_iter_update = 0
        
        if self.input_mask_slices_1 is not None:
            best_weights_1 = torch.clone(self.weights_1)
        
        # X_ is the original sequence expanded to the batch size
        X_ = X.repeat(self.batch_size, 1, 1)
        
        if X1 is not None:
            X_1 = X1.repeat(self.batch_size, 1, 1)
        
        # Ensure y_bar has shape (batch_size, num_targets, vector_len)
        # if y_bar.dim() == 2:
        #     y_bar = y_bar.unsqueeze(1)
                
        # y_bar = y_bar.expand(self.batch_size, *y_bar.shape[1:])
        
        y_bar0 = y_bar0.expand(self.batch_size, *y_bar0.shape[1:])
        y_bar1 = y_bar1.expand(self.batch_size, *y_bar1.shape[1:])
        y_bar2 = y_bar2.expand(self.batch_size, *y_bar2.shape[1:])
        y_bar3 = y_bar3.expand(self.batch_size, *y_bar3.shape[1:])
        
        tic = time.time()
        initial_tic = time.time()
        if self.verbose:
            print(("iter=I\tinput_loss=0.0\toutput_loss={:4.4}\t" +
                "total_loss={:4.4}\ttime=0.0").format(output_loss, 
                    best_total_loss))
        
        for i in range(1, self.max_iter+1):  
            # generating new sequence -> FORWARD PASS
            
            # prediction for the new sequence
            if self.use_semifreddo:
                
                if X1 is None:
                
                    X_hat = self(X)
                                        
                    semifreddo_model0 = Semifreddo(model=model0,
                                                slice_0_padded_seq=X_hat, 
                                                edited_indices_slice_0=self.input_mask_slices_0,
                                                saved_temp_output_path=self.semifreddo_temp_output_path_list[0],
                                                slice_1_padded_seq=X1,
                                                edited_indices_slice_1=self.input_mask_slices_1,
                                                batch_size=1,
                                                cropping_applied=self.cropping_applied)
                    y_hat0 = semifreddo_model0.forward()
            
                    semifreddo_model1 = Semifreddo(model=model1,
                                                slice_0_padded_seq=X_hat, 
                                                edited_indices_slice_0=self.input_mask_slices_0,
                                                saved_temp_output_path=self.semifreddo_temp_output_path_list[1],
                                                slice_1_padded_seq=X1,
                                                edited_indices_slice_1=self.input_mask_slices_1,
                                                batch_size=1,
                                                cropping_applied=self.cropping_applied)
                    y_hat1 = semifreddo_model1.forward()
                    
                    semifreddo_model2 = Semifreddo(model2,
                                                slice_0_padded_seq=X_hat, 
                                                edited_indices_slice_0=self.input_mask_slices_0,
                                                saved_temp_output_path=self.semifreddo_temp_output_path_list[2],
                                                slice_1_padded_seq=X1,
                                                edited_indices_slice_1=self.input_mask_slices_1,
                                                batch_size=1,
                                                cropping_applied=self.cropping_applied)
                    y_hat2 = semifreddo_model2.forward()
            
                    semifreddo_model3 = Semifreddo(model3,
                                                slice_0_padded_seq=X_hat, 
                                                edited_indices_slice_0=self.input_mask_slices_0,
                                                saved_temp_output_path=self.semifreddo_temp_output_path_list[3],
                                                slice_1_padded_seq=X1,
                                                edited_indices_slice_1=self.input_mask_slices_1,
                                                batch_size=1,
                                                cropping_applied=self.cropping_applied)
                    y_hat3 = semifreddo_model3.forward()
            
                    # y_hat = torch.stack([y_hat0, y_hat1, y_hat2, y_hat3], dim=0).mean(dim=0)
                
                else:
                    X_hat, X1_hat = self(X, X1)
                    
                    semifreddo_model = Semifreddo(model=self.model,
                                    slice_0_padded_seq=X_hat, 
                                    edited_indices_slice_0=self.input_mask_slices_0,
                                    saved_temp_output_path=self.semifreddo_temp_output_path,
                                    slice_1_padded_seq=X1_hat,
                                    edited_indices_slice_1=self.input_mask_slices_1,
                                    batch_size=self.batch_size,
                                    cropping_applied=self.cropping_applied)
                    y_hat = semifreddo_model.forward()
                
            else:
                X_hat = self(X)
                y_hat = self.model(X_hat)
            
            # forcing all variables to float32    
            y_hat0 = y_hat0.float()
            y_hat1 = y_hat1.float()
            y_hat2 = y_hat2.float()
            y_hat3 = y_hat3.float()
            
            y_bar0 = y_bar0.float()
            y_bar1 = y_bar1.float()
            y_bar2 = y_bar2.float()
            y_bar3 = y_bar3.float()
            
            X = X.float()
            X_hat = X_hat.float()
            
            # loss between the new and original sequence
            if X1 is not None:
                input_loss_slice_1 = self.input_loss(X_hat[:, :, inpainting_mask], X_[:, :, inpainting_mask]) / (self.batch_size * 2)
                input_loss_slice_2 = self.input_loss(X1_hat[:, :, inpainting_mask_1], X_1[:, :, inpainting_mask_1]) / (self.batch_size * 2)
                input_loss = input_loss_slice_1 + input_loss_slice_2
            else:
                input_loss = self.input_loss(X_hat[:, :, inpainting_mask], X_[:, :, inpainting_mask]) / (self.batch_size * 2)
              
            if self.output_mask_path is not None:
                # LOCAL LOSS                
                
                y_hat0_unmasked = y_hat0[..., loaded_unmask_indices]
                y_hat1_unmasked = y_hat1[..., loaded_unmask_indices]
                y_hat2_unmasked = y_hat2[..., loaded_unmask_indices]
                y_hat3_unmasked = y_hat3[..., loaded_unmask_indices]
                
                y_bar0_unmasked = y_bar0[..., loaded_unmask_indices]
                y_bar1_unmasked = y_bar1[..., loaded_unmask_indices]
                y_bar2_unmasked = y_bar2[..., loaded_unmask_indices]
                y_bar3_unmasked = y_bar3[..., loaded_unmask_indices]
                
                scaling_factor = y_hat0.shape[-1] // y_hat0_unmasked.shape[-1]
                
                y_hat0_unmasked = y_hat0_unmasked.to(device=y_bar0_unmasked.device)
                y_hat1_unmasked = y_hat1_unmasked.to(device=y_bar1_unmasked.device)
                y_hat2_unmasked = y_hat2_unmasked.to(device=y_bar2_unmasked.device)
                y_hat3_unmasked = y_hat3_unmasked.to(device=y_bar3_unmasked.device)
                
                loss0 = self.output_loss(y_hat0_unmasked, y_bar0_unmasked)
                loss1 = self.output_loss(y_hat1_unmasked, y_bar1_unmasked)
                loss2 = self.output_loss(y_hat2_unmasked, y_bar2_unmasked)
                loss3 = self.output_loss(y_hat3_unmasked, y_bar3_unmasked)

                output_loss = torch.stack([loss0, loss1, loss2, loss3]).max() * scaling_factor
                
            else:
                # GLOBAL LOSS
                output_loss = self.output_loss(y_hat, y_bar)
            
            # output_loss averaged over batch size
            output_loss = output_loss / self.batch_size
                       
            if self.ctcf_meme_path is not None:
                # pwm_CTCF = read_meme_pwm_as_numpy(self.ctcf_meme_path)
                pwm_CTCF_tensor = torch.from_numpy(self.ctcf_pwm).float()
                motifs_dict = {"CTCF": pwm_CTCF_tensor}
                
                X_hat_slice_bin = X_hat[:,:,4076:-4076]
                X_hat_slice_bin_cpu = X_hat_slice_bin.cpu().detach().numpy()
                # fimo score hits
                # X_hat_hits = fimo.fimo(motifs=motifs_dict, sequences=X_hat_slice_bin_cpu, threshold=1e-4, reverse_complement=True)
                
                # if X1 is not None:
                #     X1_hat_slice_bin = X1_hat[:,:,4076:-4076]
                #     X1_hat_slice_bin_cpu = X1_hat_slice_bin.cpu().detach().numpy()
                    # X1_hat_hits = fimo.fimo(motifs=motifs_dict, sequences=X1_hat_slice_bin_cpu, threshold=1e-4, reverse_complement=True)
                    
                    # score = max(X_hat_hits[0]["score"].sum(), X1_hat_hits[0]["score"].sum())
                # else:
                    # score = X_hat_hits[0]["score"].sum()
                    
            if self.punish_ctcf:
                total_loss = output_loss + torch.tensor(self.l, dtype=torch.float32) * input_loss + score * self.g
            else:
                total_loss = output_loss + torch.tensor(self.l, dtype=torch.float32) * input_loss
            
            # BACKWARD PASS
            # gradient calculation and weights update
            optimizer.zero_grad()
            
            if self.input_mask_slices_1 is not None:
                optimizer_1.zero_grad()
            
            total_loss.backward(retain_graph=True)                                    
            
            optimizer.step()
            
            if self.input_mask_slices_1 is not None:
                optimizer_1.step()
            
            if self.suppressing_mask is not None:
                padding_bp = 4096
                self.apply_freeze_mask(self.weights_0, X[:, :, padding_bp:-padding_bp], self.suppressing_mask)
            
                with torch.no_grad():
                    for pos in torch.where(self.suppressing_mask)[0]:
                        logits = self.weights_0[0, :, pos]
                        orig_nt = torch.argmax(X[:, :, padding_bp:-padding_bp][0, :, pos]).item()
                        assert torch.isclose(logits[orig_nt], torch.tensor(0.0, device=logits.device)), f"Position {pos} not frozen correctly!"
                        assert torch.all(logits != logits.max()) or logits[orig_nt] == logits.max(), f"Other nucleotides at {pos} are not suppressed!"
            
            output_loss = output_loss.item()
            input_loss = input_loss.item()
            total_loss = total_loss.item()
            
            if self.verbose and i % self.report_iter == 0:
                print(("iter={}\tinput_loss={:4.4}\toutput_loss={:4.4}\t" +
                    "total_loss={:4.4}\ttime={:4.4}").format(i, input_loss, 
                        output_loss, total_loss, time.time() - tic))
            
                tic = time.time()               
            
            if self.return_history:
                # history['edits'].append(torch.where(X_hat != X_)) # plotting needs to be fixed for the full model
                history['input_loss'].append(input_loss)
                history['output_loss'].append(output_loss)
                history['total_loss'].append(total_loss)
                
                if X1 is not None:
                    gc_cont = (gc_content(X_hat) + gc_content(X1_hat)) / 2
                else:
                    gc_cont = gc_content(X_hat)
                history['gc_content'].append(gc_cont)

                if self.ctcf_meme_path is not None:
                    history['ctcf_fimo_sum_score'].append(score)
                
                # Track edit positions (just for batch[0] to simplify)
                orig_nt = torch.argmax(X[:,:,4096:-4096], dim=1)
                edited_nt = torch.argmax(X_hat[:,:,4096:-4096], dim=1)
                edit_mask = orig_nt[0] != edited_nt[0]
                edit_positions = torch.nonzero(edit_mask, as_tuple=False).squeeze().tolist()
                if isinstance(edit_positions, int):  # in case of single position
                    edit_positions = [edit_positions]
                edit_mask_np = edit_mask.cpu().numpy().astype(int)
                history['edit_positions'].append(edit_mask_np.tolist())
                
            if total_loss < best_total_loss:
                last_iter_update = i
                best_input_loss = input_loss
                best_output_loss = output_loss
                best_total_loss = total_loss

                best_sequence = torch.clone(X_hat)
                best_weights_0 = torch.clone(self.weights_0)
                
                # calculating number of edits
                orig_nt = torch.argmax(X, dim=1)          # shape: (batch_size, seq_len)
                best_nt = torch.argmax(X_hat, dim=1)      # shape: (batch_size, seq_len)

                edit_counts = torch.sum(orig_nt != best_nt)
                # If batch_size == 1, extract scalar
                num_edits = edit_counts.item() if edit_counts.numel() == 1 else edit_counts.tolist()
                
                # PearsonR between prediction and target 
                # Flatten from (1, 1, 130305) → (1, 130305)
                y_hat_ct0_flat = y_hat0.view(y_hat0.size(0), -1)
                y_bar_ct0_flat = y_bar0.view(y_bar0.size(0), -1)
                
                # Ensure both are on the same device
                y_bar_ct0_flat = y_bar_ct0_flat.to(y_hat_ct0_flat.device)       
                               
                pearson_r_batch = batch_pearsonr(y_hat_ct0_flat, y_bar_ct0_flat)  # shape: (batch_size,)
                best_pearson_r = pearson_r_batch.item()  # since batch size = 1      
                
                if self.input_mask_slices_1 is not None:
                    best_weights_1 = torch.clone(self.weights_1)
                
                if X1 is not None:
                    best_sequence_1 = torch.clone(X1_hat)
                
                n_iter_wo_improvement = 0
            else:
                n_iter_wo_improvement += 1
                if n_iter_wo_improvement == self.early_stopping_iter:
                    break

        optimizer.zero_grad()
        self.weights_0 = torch.nn.Parameter(best_weights_0)
        
        if self.suppressing_mask is not None:
            padding_bp = 4096
            self.apply_freeze_mask(self.weights_0, X[:, :, padding_bp:-padding_bp], self.suppressing_mask)
        
        if self.verbose:
            print(("iter=F\tinput_loss={:4.4}\toutput_loss={:4.4}\t" +
                "total_loss={:4.4}\ttime={:4.4}").format(best_input_loss, 
                    best_output_loss, best_total_loss, 
                    time.time() - initial_tic))
            print("Last iteration with update: ", last_iter_update)
        if self.return_history:
            if X1 is not None:
                return best_sequence, best_sequence_1, history
            else:
                return best_sequence, history
        else:
            if X1 is not None:
                return best_sequence, best_sequence_1, int(last_iter_update)
            else:
                # return best_sequence
                if last_iter_update == 0:
                    return best_sequence, int(last_iter_update), float(best_output_loss), int(0), float('nan')
                else:
                    return best_sequence, int(last_iter_update), float(best_output_loss), int(num_edits), float(best_pearson_r)
