import torch


def check_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(f"[{tag}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


class Semifreddo():
    def __init__(self, sequence, edited_index, model, saved_out_path, batch_size):
        self.sequence = sequence
        self.edited_index = edited_index
        
        # model in interference mode
        self.model = model.eval()
        
        self.saved_out_path = saved_out_path
        self.batch_size = batch_size
    
    def forward(self):    
        
        self.model = self.model.eval()
        
        #passing an edited sequence though the top of model
        sub_x = self.model.conv_block_1(self.sequence)
        sub_x = self.model.conv_tower(sub_x)

        # loading saved output (for one sequence)
        x = torch.load(self.saved_out_path, weights_only=True)
        x = x.clone()  # Clone it to avoid modifying the original tensor

        if self.batch_size > 1:
            x = x.repeat(self.batch_size, 1, 1)
        
        # x[:, :, self.edited_index] = sub_x[:,:,2].squeeze(-1) #2 becaise, there are 5 bins, we need the middle one
        # for now, 3 bins get updated
        x[:, :, self.edited_index-1:self.edited_index+1] = sub_x[:,:,1:3].squeeze(-1)
        
        x = self.model.residual1d_block1(x)
        x = self.model.conv_reduce(x)
        x = self.model.one_to_two(x)
        x = self.model.concat_dist(x)
        x = self.model.conv2d_block(x)
        x = self.model.symmetrize_2d(x)
        x = self.model.residual2d_block1(x)
        x = self.model.residual2d_block2(x)
        x = self.model.residual2d_block3(x)
        x = self.model.residual2d_block4(x)
        x = self.model.cropping_2d(x)
        x = self.model.upper_tri(x)
        x = self.model.final(x)
        
        return x