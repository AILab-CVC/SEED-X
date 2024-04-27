import deepspeed
from transformers import AutoConfig
from transformers.deepspeed import is_deepspeed_zero3_enabled
from torch import nn


def remove_mismatched_weights(model, pretrained_state_dict):
    own_state = model.state_dict()
    mismatch_keys = []

    for name in list(pretrained_state_dict.keys()):
        if name not in own_state or own_state[name].shape != pretrained_state_dict[name].shape:
            mismatch_keys.append(name)
            pretrained_state_dict.pop(name)

    return pretrained_state_dict, mismatch_keys


def load_zero3_checkpoint(module: nn.Module, state_dict, prefix="", error_msgs = [], top=True):
    # check if zero3 
    
    zero3_enabled = is_deepspeed_zero3_enabled()
    # print(f'zero3_enabled: {zero3_enabled}')

    if not is_deepspeed_zero3_enabled():

        state_dict, mismatch_keys = remove_mismatched_weights(module, state_dict)



        info = module.load_state_dict(state_dict, strict=False)


        if len(mismatch_keys) > 0:
            print("shape mismatch keys: ", mismatch_keys)


        if len(info.missing_keys) > 0:
            print("missing keys: ", info.missing_keys)
        
        if len(info.unexpected_keys) > 0:
            print("unexpected keys: ", info.unexpected_keys)

    else:
        # error_msgs = []
        local_metadata = {}
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
    
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            params_name = [k for k in state_dict.keys() if k in named_parameters]
            ## named buffer for layers like batchnorm
            named_buffers = dict(module.named_buffers(prefix=prefix[:-1], recurse=False))
            buffers_to_gather = [named_buffers[k] for k in state_dict.keys() if k in named_buffers]

            if len(params_to_gather) > 0 or len(buffers_to_gather)>0:
                # if len(buffers_to_gather)>0:
                #     print("loading buffers")
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    # if torch.distributed.get_rank() == 0:
                    # if only rank0, then module's buffer will not be syncd
                    # for k, v in zip(params_name, params_to_gather):
                        # log the shape of the loaded weights
                        # print(f'loading {k} with shape {v.shape}')
                    module._load_from_state_dict(*args)

                
            # if len (error_msgs) > 0:
            #     print(error_msgs)
        
        for name, child in module._modules.items():
            if child is not None:
                load_zero3_checkpoint(child, state_dict, prefix + name + ".", top=False)
        
        if top:
            if len(error_msgs) > 0:
                print('loading zero3 model weights meets error messages!')
                print(error_msgs)
            else:
                print('loading zero3 model weights success!')
                