from structured_stable_diffusion.models.diffusion.ddpm import LatentDiffusion
from collections import defaultdict
import torch
#@Wenxuan
class model_manager():
    
    def __init__(self, **kwargs) -> None:
        """
        Create a manager for globally logging and managing layers and attributes of LatentDiffusion 
        """
        self.cross_attn_layers = []
        
        #set save attention maps
        self.save_map = kwargs.pop("save_attn_maps", False)
        self.set_save_map(self.save_map)
        
        self.option = kwargs.pop("option", None)
        #set cross-attention type
        if "struc" in self.option and "key" in self.option and "value" in self.option:
            self.set_struct_key_val()
        elif "struc" in self.option and "key" in self.option:
            self.set_struct_key()
        else:
            pass #do nothing, multi_qkv will automatically modify value matrices with mutiple conditionings
        
        self.noun_idx = kwargs.pop("noun_idx", None)
        #set perturbation
        if "perturb" in self.option:
            self.perturb_strength = float(self.option.split("_")[-1])
            self.component_to_perturb = self.option.split("_")[-2]

    def find_cross_attn_layers(self):
        """
        Find all cross-attention layers in the model
        """
        assert isinstance(self.model, LatentDiffusion), \
            "child class must be a sampler and model must be a LatentDiffusion object. Check inheritance hierarchy"
        
        if not self.cross_attn_layers:
            #empty list
            for name, module in self.model.model.diffusion_model.named_modules():
                module_name = type(module).__name__
                #attn1 is self-attention and attn2 is cross-attention
                if module_name == 'CrossAttention' and 'attn2' in name:
                    self.cross_attn_layers.append((name, module))
        
        return self.cross_attn_layers
    
    def save_attention_maps(self, cur_step, max_steps, **kwargs):
        #log the last denoising step only by default
        log_freq = kwargs.pop("log_freq", 0)
        is_log = cur_step % log_freq == 0 if log_freq > 0 else cur_step == max_steps
        
        if is_log:
            assert isinstance(self.model, LatentDiffusion), \
            "child class must be a sampler and model must be a LatentDiffusion object. Check inheritance hierarchy"

            for name, module in self.find_cross_attn_layers():
                self.attn_maps[name].append(module.attn_maps)
                self.v_matrix[name].append(module.v_matrix)

    def set_save_map(self, save_map: bool):
        """
        This overrides the save_map param in config to save troubles
        """
        assert isinstance(self.model, LatentDiffusion), \
            "child class must be a sampler and model must be a LatentDiffusion object. Check inheritance hierarchy"
        
        for name, module in self.find_cross_attn_layers():
            module.save_map = save_map
        self.attn_maps = defaultdict(list)
        self.v_matrix = defaultdict(list)

    def test_log_attn(self):
        """
        Tester
        """
        self.save_attention_maps(cur_step=49, max_steps=49)
        assert self.attn_maps != defaultdict(list), "log_attn_maps is not working properly"
        self.attn_maps = defaultdict(list)

        self.save_attention_maps(cur_step=5, max_steps=10, log_freq=5)
        assert self.attn_maps != defaultdict(list), "log_attn_maps is not working properly"
        self.attn_maps = defaultdict(list)

        self.log_attn_test_passed = True   

    def set_struct_key(self):
        """
        Notify all cross-attn layers to add&average key matrices instead of value matrices
        """
        for name, module in self.find_cross_attn_layers():
            module.struct_key = True
    
    def set_struct_key_val(self):
        """
        Notify all cross-attn layers to add&average key matrices and value matrices
        """
        for name, module in self.find_cross_attn_layers():
            module.struct_key_val = True

    def set_perturb_strength(self, strength, noun_idx):
        """
        @param strength: float, strength of perturbation. 
            Will use gaussian noise with std = original_std * strength
            and mean = 0
        """
        component = self.component_to_perturb
        assert self.noun_idx is not None, "must provide noun_idx to perturb selected columns or rows"
        assert component in ["key", "value"], "can only perturb key or value matrices"
        
        #functional programming :)
        def perturbed_forward(self, strength, noun_idx):
            if component == "key":
                out_layer = self.to_k
            elif component == "value":
                out_layer = self.to_v
            
            def forward(x):
                x = out_layer(x)
                #x: [batch_size, num_heads, seq_len, dim]
                #perturb selected rows or columns
                if component == "key":
                    x[:, :, noun_idx, :] += torch.randn_like(x[..., :, noun_idx]) * strength
                elif component == "value":
                    x[:, :, :, noun_idx] += torch.randn_like(x[..., noun_idx, :]) * strength
                return x
            
            return forward
        
        for name, module in self.find_cross_attn_layers():
            if component == "key":
                module.to_k = perturbed_forward(module, strength, component, noun_idx)
            elif component == "value":
                module.to_v = perturbed_forward(module, strength, component, noun_idx)

    def on_step_end(self, cur_step, max_steps, **kwargs):
        if self.save_map:
            if not getattr(self, 'log_attn_test_passed', False):
                self.test_log_attn()
            self.save_attention_maps(cur_step=cur_step, max_steps=max_steps, **kwargs)