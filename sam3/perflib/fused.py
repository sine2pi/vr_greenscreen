
import torch

addmm_act_op = torch.ops.aten._addmm_activation

def addmm_act(activation_type, linear_layer, mat1, beta=1, alpha=1):

    if torch.is_grad_enabled():
        bias = linear_layer.bias
        weight = linear_layer.weight
        orig_shape = mat1.shape
        
        if mat1.ndim > 2:
            mat1_flat = mat1.view(-1, orig_shape[-1])
        else:
            mat1_flat = mat1

        res = torch.addmm(bias, mat1_flat, weight.t(), beta=beta, alpha=alpha)
        
        if mat1.ndim > 2:
            new_shape = orig_shape[:-1] + (res.shape[-1],)
            res = res.view(new_shape)
            
        if activation_type in [torch.nn.ReLU, torch.nn.functional.relu]:
            return torch.relu(res)
        elif activation_type in [torch.nn.GELU, torch.nn.functional.gelu]:
            return torch.nn.functional.gelu(res)
        elif activation_type in [torch.nn.SiLU, torch.nn.functional.silu]:
            return torch.nn.functional.silu(res)
        else:
            raise NotImplementedError(f"Fallback training activation for {activation_type} not implemented.")

    self = linear_layer.bias.detach()
    mat2 = linear_layer.weight.detach()
    self = self.to(torch.bfloat16)
    mat1 = mat1.to(torch.bfloat16)
    mat2 = mat2.to(torch.bfloat16)
    mat1_flat = mat1.view(-1, mat1.shape[-1])
    
    if activation_type in [torch.nn.functional.relu, torch.nn.ReLU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
        
    if activation_type in [torch.nn.functional.gelu, torch.nn.GELU]:
        y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
        return y.view(mat1.shape[:-1] + (y.shape[-1],))
        
    raise ValueError(f"Unexpected activation {activation_type}")

# def addmm_act(activation, linear, mat1):
#     if torch.is_grad_enabled():
#         out = linear(mat1)
#         if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
#             return torch.nn.functional.relu(out)
#         if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
#             return torch.nn.functional.gelu(out)
#         raise ValueError(f"Unexpected activation {activation}")

#     self = linear.bias.detach()
#     mat2 = linear.weight.detach()
#     self = self.to(torch.bfloat16)
#     mat1 = mat1.to(torch.bfloat16)
#     mat2 = mat2.to(torch.bfloat16)
#     mat1_flat = mat1.view(-1, mat1.shape[-1])
#     if activation in [torch.nn.functional.relu, torch.nn.ReLU]:
#         y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=False)
#         return y.view(mat1.shape[:-1] + (y.shape[-1],))
#     if activation in [torch.nn.functional.gelu, torch.nn.GELU]:
#         y = addmm_act_op(self, mat1_flat, mat2.t(), beta=1, alpha=1, use_gelu=True)
#         return y.view(mat1.shape[:-1] + (y.shape[-1],))
#     raise ValueError(f"Unexpected activation {activation}")
