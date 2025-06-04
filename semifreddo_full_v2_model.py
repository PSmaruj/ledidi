import torch


def check_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(f"[{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


class Semifreddo():
    def __init__(self, 
                 model,
                 slice_0_padded_seq, 
                 edited_indices_slice_0,
                 saved_temp_output_path, 
                 slice_1_padded_seq=None,
                 edited_indices_slice_1=None,
                 batch_size=1,
                 cropping_applied=32
                 ):
        
        # model in interference mode
        self.model = model.eval()
        
        self.slice_0_padded_seq = slice_0_padded_seq
        self.slice_1_padded_seq = slice_1_padded_seq
        self.edited_indices_slice_0 = edited_indices_slice_0
        self.edited_indices_slice_1 = edited_indices_slice_1
        self.saved_temp_output_path = saved_temp_output_path
        self.batch_size = batch_size
        self.cropping_applied = cropping_applied


    def forward(self):    
        
        self.model = self.model.eval()
        device = next(self.model.parameters()).device
        
        # SLICE 0
        # passing an edited sequence though the top of model
        sub_x_0 = self.model.conv_block_1(self.slice_0_padded_seq.to(device))
        sub_x_0 = self.model.conv_tower(sub_x_0)
        
        # SLICE 1
        if self.slice_1_padded_seq is not None:
            sub_x_1 = self.model.conv_block_1(self.slice_1_padded_seq)
            sub_x_1 = self.model.conv_tower(sub_x_1)
        
        # loading saved output (for one sequence)
        x = torch.load(self.saved_temp_output_path, weights_only=True, map_location=device)
        x = x.clone()  # Clone it to avoid modifying the original tensor

        if x.shape[0] != self.batch_size:
            x = x.repeat(self.batch_size, 1, 1)
        
        # slice 0
        edited_slice_0_start = min(self.edited_indices_slice_0) + self.cropping_applied
        edited_slice_0_end = max(self.edited_indices_slice_0) + self.cropping_applied 
        
        # replacing +-1 bin               
        x[:, :, edited_slice_0_start-1:edited_slice_0_end+2] = sub_x_0[:,:,1:-1].squeeze(-1)
        
        # slice 1
        if self.slice_1_padded_seq is not None:
            edited_slice_1_start = min(self.edited_indices_slice_1) + self.cropping_applied
            edited_slice_1_end = max(self.edited_indices_slice_1) + self.cropping_applied

            # replacing +-1 bin
            x[:, :, edited_slice_1_start-1:edited_slice_1_end+2] = sub_x_1[:,:,1:-1].squeeze(-1)
            
        # to get reverse_bool of correct shape
        x, reverse_bool = self.model.stochastic_reverse_complement(x)
        
        x = self.model.residual1d_block1(x)
        x = self.model.residual1d_block2(x) 
        x = self.model.residual1d_block3(x)
        x = self.model.residual1d_block4(x)
        x = self.model.residual1d_block5(x)
        x = self.model.residual1d_block6(x)
        x = self.model.residual1d_block7(x)
        x = self.model.residual1d_block8(x)
        
        # added in V2
        x = self.model.residual1d_block9(x)
        x = self.model.residual1d_block10(x)
        x = self.model.residual1d_block11(x)
        
        x = self.model.conv_reduce(x)
        x = self.model.one_to_two(x)
        
        x = self.model.conv2d_block(x)
        x = self.model.symmetrize_2d(x)
        
        x = self.model.residual2d_block1(x)
        x = self.model.residual2d_block2(x)
        x = self.model.residual2d_block3(x)
        x = self.model.residual2d_block4(x)
        x = self.model.residual2d_block5(x)
        x = self.model.residual2d_block6(x)
        
        # added in V2
        x = self.model.squeeze_excite(x)
        
        x = self.model.cropping_2d(x)
        x = self.model.upper_tri(x, reverse_complement_flags=reverse_bool)
        x = self.model.final(x)
        
        return x