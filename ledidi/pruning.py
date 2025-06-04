# pruning.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import torch
from semifreddo_full_v2_model import Semifreddo


@torch.no_grad()
def greedy_pruning(model, X, X_hat, input_mask_slices_0=[224], 
                   output_loss=torch.nn.L1Loss(reduction='sum'),
                   X1=None, X1_hat=None, input_mask_slices_1=None,
                   threshold=1e3, fast_threshold=10, cropping_applied=32,
                   semifreddo_temp_output_path=None, verbose=False):
	"""A method for pruning edits to remove those that are irrelevant.

	This method will greedily go through all of the proposed edits and evaluate
	the effect of removing them, one at a time. As a greedy method, this will
	iteratively scan over all edits and remove the one with the smallest change
	in model output assuming that change is below the predefined threshold.
	Once the change in output from the edit with the smallest change is above
	the threshold, the procedure will stop and return the remaining edits.

	Note: Only one sequence is pruned at a time.


	Parameters
	----------
	model: torch.nn.Module
		A PyTorch model used to evaluate the edits.

	X: torch.tensor, shape=(1, d, length)
		A tensor where the second dimension is the number of categories (e.g., 4 
		for DNA) and the third dimension is the length of the sequence, and is
		one-hot encoded.

	X_hat: torch.tensor, shape=(1, d, length)
		A tensor of the same shape as `X` except that it contains the proposed
		edits.

	threshold: float, optional
		A threshold on the maximum change in model output that removing an edit
		can have. Default is 1.

	verbose: bool, optional
		Whether to print out the index and delta at each iteration.

	Returns
	-------
	X_m: torch.tensor, shape=(1, d, length)
		A tensor of the same shape as `X_hat` except with some of the edits
		reverted back to what they were in `X`.
	"""

	model = model.eval()
	X_hat = torch.clone(X_hat)
	if X1_hat is not None:
		X1_hat = torch.clone(X1_hat)
	else:
		X1_hat = None
	
	diff_idxs_ = torch.where((X != X_hat).sum(axis=1) > 0)[1]
	diff_idxs = set([idx.item() for idx in diff_idxs_])
	n, n_total = 0, len(diff_idxs)

	print(f"Initial number of differences to consider for pruning: {n_total}")
	model = model.to(X_hat.device)
 
	initial_semifreddo_model = Semifreddo(model=model,
                               slice_0_padded_seq=X_hat, 
                               edited_indices_slice_0=input_mask_slices_0,
                               saved_temp_output_path=semifreddo_temp_output_path,
                               slice_1_padded_seq=X1_hat,
                               edited_indices_slice_1=input_mask_slices_1,
                               batch_size=1,
                               cropping_applied=cropping_applied)
	
	y_hat = initial_semifreddo_model.forward()

	# FAST INITIAL PRUNING PHASE
	to_remove = set()
	loss_diffs = []
 
	for idx in diff_idxs:
		X_mod = torch.clone(X_hat)
		X_mod[0, :, idx] = X[0, :, idx]

		modified_semifreddo_model = Semifreddo(model=model,
											slice_0_padded_seq=X_mod,
											edited_indices_slice_0=input_mask_slices_0,
											saved_temp_output_path=semifreddo_temp_output_path,
											slice_1_padded_seq=X1_hat,
											edited_indices_slice_1=input_mask_slices_1,
											batch_size=1,
											cropping_applied=cropping_applied)

		y_mod = modified_semifreddo_model.forward()

		loss_diff_score = output_loss(y_hat, y_mod)
		loss_diffs.append(float(loss_diff_score))
  
		if loss_diff_score <= fast_threshold:
			to_remove.add(idx)
	# Apply fast pruning
	for idx in to_remove:
		X_hat[0, :, idx] = X[0, :, idx]
		diff_idxs.remove(idx)

	print(f"Fast pruning phase removed {len(to_remove)} edits out of {n_total}.")
		
	# update n_total after fast phase
	n_total = len(diff_idxs)
	print(f"{n_total} edits left for greedy pruning.")	
 
	# # for i in range(n_total):
	# for i in range(1):
	# 	print(f"\nStarting pruning iteration {i+1}, remaining differences: {len(diff_idxs)}")
	# 	tic = time.time()
	# 	best_score, best_idx = float("inf"), -1

	# 	for idx in diff_idxs:
	# 		X_mod = torch.clone(X_hat)
	# 		X_mod[0, :, idx] = X[0, :, idx] #removing an edit

		# 	modified_semifreddo_model = Semifreddo(model=model,
		# 					slice_0_padded_seq=X_mod, 
		# 					edited_indices_slice_0=input_mask_slices_0,
		# 					saved_temp_output_path=semifreddo_temp_output_path,
		# 					slice_1_padded_seq=X1_hat,
		# 					edited_indices_slice_1=input_mask_slices_1,
		# 					batch_size=1,
		# 					cropping_applied=cropping_applied)
	
		# 	y_mod = modified_semifreddo_model.forward()
			
		# 	loss_diff_score = output_loss(y_hat, y_mod)
   
		# 	if loss_diff_score < best_score:
		# 		best_score = loss_diff_score
		# 		best_idx = idx 
		# 		print(f"Best edit to remove this round: idx={best_idx}, loss difference={best_score:.4f}")
				
		# if best_score < threshold:
		# 	diff_idxs.remove(best_idx)
		# 	X_hat[0, :, best_idx] = X[0, :, best_idx]
		# 	n += 1

		# 	if verbose:
		# 		print("# Pruned: {}/{}\tPruned Index: {}\tLoss Difference: {:4.4}\tTime: {:4.4}s".format(n, n_total, best_idx, 
		# 			best_score, time.time() - tic))

		# else:
		# 	break
	
	# print(f"\nFinished pruning. Final number of edits kept: {len(diff_idxs)} out of {n_total} initial edits.")
 	
	return X_hat, diff_idxs, loss_diffs
