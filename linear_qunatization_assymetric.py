import torch
def qunatize(tensor, dtype=torch.int8):
    r_min=tensor.min().item()
    r_max=tensor.max().item()
    q_min=torch.iinfo(dtype).min
    q_max=torch.iinfo(dtype).max 

    scale_val=(r_max-r_min)/(q_max-q_min)
    zero_val = int((q_min -(r_min/scale_val)))
    
    qunat_tensor = torch.round((tensor/scale_val)) + zero_val
    return qunat_tensor

torch.manual_seed(12)
tensor_a = torch.randint(1,10000,(5,5),dtype=torch.float32)/17
tensor_b = torch.randint(1,10000,(5,3),dtype=torch.float32)/3

print(tensor_a)
print(qunatize(tensor_a))
print(tensor_b)
print(qunatize(tensor_b))
