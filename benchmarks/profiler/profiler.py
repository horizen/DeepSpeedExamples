import os
from packaging.version import Version
import torch
import torch.cuda.profiler as prof
import torch.cuda.nvtx as nvtx
import torch.nn.modules.module as nn
import torch.optim.optimizer as optim

ti_hook_step = 0
ti_init_handler = None
_ti_model = None

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
    global ti_hook_step
    nvtx.range_push("rank{}-epoch{}-{}-state".format(_get_rank(), ti_hook_step, module.__class__.__name__))  
    return None


def _before_forward(module, input):
    global ti_hook_step
    if ti_hook_step >= 2:
        nvtx.range_push("rank{}-epoch{}-{}-forward".format(_get_rank(), ti_hook_step, module.__class__.__name__)) 
        
    return None


def _after_forward(module, input, output):
    global ti_hook_step
    if ti_hook_step >= 2:
        nvtx.range_pop()
    return None


def _init_module(module, input):
    global ti_init_handler
    ti_init_handler.remove()
    nvtx.range_pop()

    global _ti_model 
    if _ti_model is None:
        _ti_model = module
    
        _ti_model.register_forward_pre_hook(_before_forward)
        _ti_model.register_forward_hook(_after_forward)
        _ti_model.register_state_dict_pre_hook(_before_module_state_dict)
        _ti_model._register_state_dict_hook(_range_pop4) 
    

def _before_backward(module, grad_output):
    global ti_hook_step
    nvtx.range_push("rank{}-epoch{}-{}-backward".format(_get_rank(), ti_hook_step, module.__class__.__name__)) 
    return None


def _before_optim_state_dict(optimizer, state_dict):
    global ti_hook_step
    nvtx.range_push("rank{}-epoch{}-{}-state".format(_get_rank(), ti_hook_step, optimizer.__class__.__name__)) 
    return None


def _before_step(optimizer, args, kwargs):
    global ti_hook_step
    if ti_hook_step >= 2:
        nvtx.range_push("rank{}-epoch{}-{}-step".format(_get_rank(), ti_hook_step, optimizer.__class__.__name__)) 
        if Version(torch.__version__) >= Version("2.3.0"):
            optimizer.register_state_dict_pre_hook(_before_optim_state_dict)
            optimizer.register_state_dict_post_hook(_range_pop2)
    return None


def _after_step(optimizer, args, kwargs):
    global ti_hook_step
    if ti_hook_step >= 2:
        nvtx.range_pop()
    ti_hook_step = ti_hook_step + 1
    return None


def pre_hook():
    global ti_init_handler
    ti_init_handler = nn.register_module_forward_pre_hook(_init_module)
    optim.register_optimizer_step_pre_hook(_before_step)
    optim.register_optimizer_step_post_hook(_after_step)
    prof.start()
    nvtx.range_push("rank{}-init".format(_get_rank()))


def post_hook():
    prof.stop()

