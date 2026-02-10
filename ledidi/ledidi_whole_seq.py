# ledidi.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
#          adapted from code written by Yang Lu

import time
import torch

import sys
import numpy as np

# from semifreddo_full_model import Semifreddo
# from ledidi.pwm_score import read_meme_pwm_as_numpy
# from tangermeme.tools import fimo


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
                 model, 
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
                 seq_length=1310720,
                 num_channels=4, 
                 bin_size=2048, 
                 punish_ctcf=False,
                 ctcf_meme_path=None):
        super().__init__()
        
        for param in model.parameters():
            param.requires_grad = False
        
        # model in eval mode
        self.model = model.eval()
        
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
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.bin_size =  bin_size
        self.punish_ctcf = punish_ctcf
        self.ctcf_meme_path = ctcf_meme_path
        
        if (self.punish_ctcf == True) and (self.ctcf_meme_path is None):
            print("Please, provide a path to the CTCF motif in the meme format.")
        
        if self.ctcf_meme_path is not None:
            self.ctcf_pwm = read_meme_pwm_as_numpy(self.ctcf_meme_path)
        else:
            self.ctcf_pwm = None
        
        self.weights = torch.nn.Parameter(
            torch.zeros((1, self.num_channels, self.seq_length), dtype=torch.float32, requires_grad=True)
        )
        
        print("Model in train mode:", self.model.training)
        
        print("Gradients enabled for weights:", self.weights.requires_grad)
        print("Weights shape:", self.weights.shape)

        
    def forward(self, X):
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
        
        # if we want to edit the entire seq
        logits = torch.log(X + self.eps) + self.weights
        
        logits = logits.expand(self.batch_size, -1, -1)
        edited = torch.nn.functional.gumbel_softmax(logits, tau=self.tau, hard=True, dim=1)
        return edited 


    def fit_transform(self, X, y_bar):
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
        
        optimizer = torch.optim.AdamW((self.weights,), lr=self.lr)

        history = {'edits': [], 'input_loss': [], 'output_loss': [], 
            'total_loss': [], 'batch_size': self.batch_size}
        
        # if self.ctcf_meme_path is None:
        #     history = {'input_loss': [], 'output_loss': [], 
        #         'total_loss': [], 'gc_content': [], 'batch_size': self.batch_size}
        # else:
        #     history = {'input_loss': [], 'output_loss': [], 
        #         'total_loss': [], 'gc_content': [], 'ctcf_fimo_sum_score': [], 'batch_size': self.batch_size}
        
        # inpainting_mask - ensures only the valid positions
        # are taken into account while input_loss is calculates
        inpainting_mask = X[0].sum(dim=0) == 1
        
        y_hat = self.model(X)
        
        n_iter_wo_improvement = 0
        
        # loss between the prediction of the original sequence
        # and the desired prediction (aka starting loss)  
        # if self.output_mask_path is not None:
        #     # LOCAL LOSS
        #     print("Local loss applied.")
        #     loaded_unmask_indices = torch.load(self.output_mask_path, weights_only=True)
        #     loaded_unmask_indices = loaded_unmask_indices.to(dtype=torch.long, device=y_hat.device)
            
        #     y_hat_unmasked = y_hat[..., loaded_unmask_indices]
        #     y_bar_unmasked = y_bar[..., loaded_unmask_indices]
            
        #     scaling_factor = y_hat.shape[-1] // y_hat_unmasked.shape[-1]
            
        #     output_loss = self.output_loss(y_hat_unmasked, y_bar_unmasked) * scaling_factor
            
        #     # mixed loss = global loss + scaled local loss
        #     # output_loss_everywhere = self.output_loss(y_hat, y_bar)
        #     # local_loss_scaled = self.output_loss(y_hat_unmasked, y_bar_unmasked) * scaling_factor
        #     # output_loss = output_loss_everywhere + local_loss_scaled
        
        # else:
        
        print("y_hat shape", y_hat.shape)
        print("y_bar shape", y_bar.shape)
        
        # GLOBAL LOSS
        print("Global loss applied.")
        output_loss = self.output_loss(y_hat, y_bar)
        
        best_input_loss = 0.0
        best_output_loss = output_loss
        best_total_loss = output_loss
        best_sequence = X
        best_weights = torch.clone(self.weights)
        last_iter_update = 0
        
        # for the movie purposes
        torch.save(best_sequence, f"/scratch1/smaruj/genomic_map_transformation/movie/seq_0.pt")
        
        # X_ is the original sequence expanded to the batch size
        X_ = X.repeat(self.batch_size, 1, 1)
        
        # Ensure y_bar has shape (batch_size, num_targets, vector_len)
        if y_bar.dim() == 2:
            y_bar = y_bar.unsqueeze(1)
                
        y_bar = y_bar.expand(self.batch_size, *y_bar.shape[1:])
        
        print("X_ shape", X_.shape)
        print("y_bar shape", y_bar.shape)
        
        tic = time.time()
        initial_tic = time.time()
        if self.verbose:
            print(("iter=I\tinput_loss=0.0\toutput_loss={:4.4}\t" +
                "total_loss={:4.4}\ttime=0.0").format(output_loss, 
                    best_total_loss))
        
        for i in range(1, self.max_iter+1):
            # generating new sequence -> FORWARD PASS
            
            # prediction for the new sequence
            X_hat = self(X)
            y_hat = self.model(X_hat)
            
            # for the movie purposes
            # torch.save(best_sequence, f"/scratch1/smaruj/genomic_map_transformation/movie/seq_{i}.pt")
            
            # if self.output_mask_path is not None:
            #     # LOCAL LOSS                
            #     y_hat_unmasked = y_hat[..., loaded_unmask_indices]  
            #     y_bar_unmasked = y_bar[..., loaded_unmask_indices]              
            #     output_loss = self.output_loss(y_hat_unmasked, y_bar_unmasked) * scaling_factor
            # else:
            
            input_loss = self.input_loss(X_hat[:, :, inpainting_mask], X_[:, :, inpainting_mask]) / (X_hat.shape[0] * 2)
            
            # GLOBAL LOSS
            output_loss = self.output_loss(y_hat, y_bar)
            
            # output_loss averaged over batch size
            output_loss = output_loss / self.batch_size
                
            total_loss = output_loss + torch.tensor(self.l, dtype=torch.float32) * input_loss
                  
            # if self.ctcf_meme_path is not None:
            #     # pwm_CTCF = read_meme_pwm_as_numpy(self.ctcf_meme_path)
            #     pwm_CTCF_tensor = torch.from_numpy(self.ctcf_pwm).float()
            #     motifs_dict = {"CTCF": pwm_CTCF_tensor}
                
            #     X_hat_slice_bin = X_hat[:,:,4076:-4076]
            #     X_hat_slice_bin_cpu = X_hat_slice_bin.cpu().detach().numpy()
            #     # fimo score hits
            #     X_hat_hits = fimo.fimo(motifs=motifs_dict, sequences=X_hat_slice_bin_cpu, threshold=1e-4, reverse_complement=True)
                
            #     if X1 is not None:
            #         X1_hat_slice_bin = X1_hat[:,:,4076:-4076]
            #         X1_hat_slice_bin_cpu = X1_hat_slice_bin.cpu().detach().numpy()
            #         X1_hat_hits = fimo.fimo(motifs=motifs_dict, sequences=X1_hat_slice_bin_cpu, threshold=1e-4, reverse_complement=True)
                    
            #         score = max(X_hat_hits[0]["score"].sum(), X1_hat_hits[0]["score"].sum())
            #     else:
            #         # score = X_hat_hits[0]["score"].sum()
            #         score = X_hat_hits[0]["score"].max()
                    
            # if self.punish_ctcf:
            #     gamma = 650
            #     # gamma=50
            #     total_loss = output_loss + torch.tensor(self.l, dtype=torch.float32) * input_loss + score * gamma
            # else:
            #     total_loss = output_loss + torch.tensor(self.l, dtype=torch.float32) * input_loss
            
            # BACKWARD PASS
            # gradient calculation and weights update
            optimizer.zero_grad()
            
            total_loss.backward(retain_graph=True)                                    
            
            optimizer.step()
            
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
                
                # gc_cont = gc_content(X_hat)
                # history['gc_content'].append(gc_cont)

                # if self.ctcf_meme_path is not None:
                #     history['ctcf_fimo_sum_score'].append(score)
                
            if total_loss < best_total_loss:
                last_iter_update = i
                best_input_loss = input_loss
                best_output_loss = output_loss
                best_total_loss = total_loss

                best_sequence = torch.clone(X_hat)
                best_weights = torch.clone(self.weights)
                
                n_iter_wo_improvement = 0
                
                # for the movie purposes
                torch.save(best_sequence, f"/scratch1/smaruj/genomic_map_transformation/movie/seq_{i}.pt")
                
            else:
                n_iter_wo_improvement += 1
                if n_iter_wo_improvement == self.early_stopping_iter:
                    break

        optimizer.zero_grad()
        self.weights = torch.nn.Parameter(best_weights)
        
        if self.verbose:
            print(("iter=F\tinput_loss={:4.4}\toutput_loss={:4.4}\t" +
                "total_loss={:4.4}\ttime={:4.4}").format(best_input_loss, 
                    best_output_loss, best_total_loss, 
                    time.time() - initial_tic))
            print("Last iteration with update: ", last_iter_update)
        if self.return_history:
            return best_sequence, history
        else:
            # return best_sequence
            return best_sequence, last_iter_update
