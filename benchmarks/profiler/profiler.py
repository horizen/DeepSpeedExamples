import os
import torch.cuda.profiler as prof
import torch.cuda.nvtx as nvtx
import torch.nn.modules.module as nn
import torch.optim.optimizer as optim

ti_hook_step = 0

def _get_rank():
    return os.getenv("RANK")

def _range_pop1(arg0):
    nvtx.range_pop()
    return None
   
def _range_pop2(arg0, arg1):
    nvtx.range_pop()
    return None 

def _range_pop3(arg0, arg1, arg2):
    nvtx.range_pop()
    return None

def _range_pop4(arg0, arg1, arg2, arg3):
    nvtx.range_pop()
    return None

def _before_module_state_dict(module, prefix, keep_vars):
    nvtx.range_push("rank{}-epoch{}-{}-state".format(_get_rank(), ti_hook_step, module.__class__.__name__))  


def _before_forward(module, input):
    if ti_hook_step == 0:
        nvtx.range_pop()
        module.register_state_dict_pre_hook(_before_module_state_dict)
        module._register_state_dict_hook(_range_pop4)
    nvtx.range_push("rank{}-epoch{}-{}-forward".format(_get_rank(), ti_hook_step, module.__class__.__name__)) 
    return None


def _before_backward(module, grad_output):
    nvtx.range_push("rank{}-epoch{}-{}-backward".format(_get_rank(), ti_hook_step, module.__class__.__name__)) 
    return None


def _before_optim_state_dict(optimizer, state_dict):
    nvtx.range_push("rank{}-epoch{}-{}-state".format(_get_rank(), ti_hook_step, optimizer.__class__.__name__)) 
    return None 


def _before_step(optimizer, args, kwargs):
    if ti_hook_step == 0:
        optimizer.register_state_dict_pre_hook(_before_optim_state_dict)
        optimizer.register_state_dict_post_hook(_range_pop2)
    nvtx.range_push("rank{}-epoch{}-{}-step".format(_get_rank(), ti_hook_step, optimizer.__class__.__name__)) 
    return None


def _after_step(optimizer, args, kwargs):
    nvtx.range_pop()
    ti_hook_step = ti_hook_step + 1
    return None

def pre_hook():
    nn.register_module_forward_pre_hook(_before_forward)
    nn.register_module_forward_hook(_range_pop3)
    nn.register_module_full_backward_pre_hook(_before_backward)
    nn.register_module_full_backward_hook(_range_pop3)
    optim.register_optimizer_step_pre_hook(_before_step)
    optim.register_optimizer_step_post_hook(_after_step)
    prof.start()
    nvtx.range_push("rank{}-init".format(_get_rank()))


def post_hook():
    if ti_hook_step == 0:
        nvtx.range_pop()
    prof.stop()

