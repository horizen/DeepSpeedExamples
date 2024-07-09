import os
import sys
from datetime import datetime
from packaging.version import Version
import torch
import torch.cuda.profiler as prof
import torch.cuda.nvtx as nvtx
import torch.nn.modules.module as nn
import torch.nn.modules.loss as F
import torch.optim.optimizer as optim
from torch.profiler import ProfilerAction,ProfilerActivity
import time

ti_hook_step = 0
_schedule_fn = None
_ti_model = None
_ti_loss = None
_ti_action = {}
_hook_start_step = 1
_hook_end_step = 10
_torch_prof = None
_hook_start = 2e20
_deepspeed_model = None

def log(msg):
    now = datetime.now()
    timestr = now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    print("{}: [{}] {}".format(timestr, _get_rank(), msg))


def _get_rank():
    return os.getenv("RANK")


def _get_profiler():
    return os.getenv("TI_PROFILER", default="nsys")


def _range_push(name):
    global _schedule_fn
    if _schedule_fn() == ProfilerAction.RECORD:
        global _ti_action
        id = nvtx.range_start(name)
        _ti_action[name] = id


def _range_pop(name):
    global _schedule_fn
    if _schedule_fn() == ProfilerAction.RECORD:
        global _ti_action
        id = _ti_action.get(name)
        if id is not None:
            nvtx.range_end(id)
            del _ti_action[name]


def _before_module_state_dict(module, prefix, keep_vars):
    _range_push("model-state")


def _after_module_state_dict(arg0, arg1, arg2, arg3):
    _range_pop("model-state")


def _before_forward(module, input):
    _range_push("forward")


def _after_forward(module, input, output):
    _range_pop("forward")


def _before_backward(module, grad_output):
    _range_push("backward")


def _after_backward(param):
    _range_pop("backward")


def _before_optim_state_dict(optimizer, state_dict):
    _range_push("optim-state")


def _after_optim_state_dict(arg0, arg1):
    _range_pop("optim-state")


def _before_step(optimizer, args, kwargs):
    _range_push("step")
    if Version(torch.__version__) >= Version("2.3.0"):
        optimizer.register_state_dict_pre_hook(_before_optim_state_dict)
        optimizer.register_state_dict_post_hook(_after_optim_state_dict)
    return None


def _after_step(optimizer, args, kwargs):
    global ti_hook_step, _hook_start_step, _hook_end_step, _hook_start
    global _torch_prof
    if _get_profiler() == "nsys":
        _range_pop("step")

    if hasattr(_torch_prof, "step"):
        _torch_prof.step()

    ti_hook_step = ti_hook_step + 1
    if ti_hook_step == _hook_start_step and _get_profiler() == "nsys":
        _hook_start = time.time()
        log("start profiler at step {}".format(ti_hook_step))
        _torch_prof.start()
    if ti_hook_step >= _hook_end_step or time.time() - _hook_start > 300:
        log("stop profiler at step {}".format(ti_hook_step))
        _torch_prof.stop()
        sys.exit(0)
    
    return None


def prepare_nsys_profiler(schedule_fn):
    optim.register_optimizer_step_pre_hook(_before_step)

    _ti_init_handler = None
    def _init(module, input):
        log("finish setup profiler")
        _ti_init_handler.remove()
        return None

    _ti_init_loss_handler = None
    def _init_loss(module, input):
        global _ti_loss
        global ti_hook_step
        if _ti_loss is None and module.training and ti_hook_step >= 1 and _ti_model is not None:
            for fn in F.__all__:
                if fn == module.__class__.__name__:
                    _ti_init_loss_handler.remove() 
                    _ti_loss = module
                    if _deepspeed_model is None:
                        _ti_loss.register_full_backward_pre_hook(_before_backward)
                        log("init profiler {}".format(module.__class__.__name__))
                    return None
        return None

    _ti_init_model_handler = None
    def _init_module(module, input):
        global _ti_model 
        global ti_hook_step, _deepspeed_model
        if _ti_model is None and module.training and ti_hook_step >= 1:
            _ti_init_model_handler.remove()
            _ti_model = module
            try:
                from deepspeed import DeepSpeedEngine
                if isinstance(module, DeepSpeedEngine):
                    log("deepspeed has built-in nvtx, skip hook")
                    _deepspeed_model = module
            except:
                pass
            if _deepspeed_model is None:
                log("init profiler {}".format(module.__class__.__name__))
                _ti_model.register_forward_pre_hook(_before_forward)
                _ti_model.register_forward_hook(_after_forward)
                _ti_model.register_state_dict_pre_hook(_before_module_state_dict)
                _ti_model._register_state_dict_hook(_after_module_state_dict)
                for name, param in _ti_model.named_parameters():
                    if param.requires_grad:
                        log("register hook {}, {}, {}".format(name, type(param), param.size()))
                        if hasattr(param, "register_post_accumulate_grad_hook"):
                            param.register_post_accumulate_grad_hook(_after_backward)
                        else:
                            param.register_hook(_after_backward)
                        break
        return None

    _ti_init_handler = nn.register_module_forward_pre_hook(_init)
    _ti_init_model_handler = nn.register_module_forward_pre_hook(_init_module)
    _ti_init_loss_handler = nn.register_module_forward_pre_hook(_init_loss)

    def wrapper_scheduler_fn():
        global ti_hook_step
        action = schedule_fn(ti_hook_step)
        if action == ProfilerAction.RECORD_AND_SAVE:
            return ProfilerAction.RECORD
        else:
            return action
    
    global _schedule_fn
    _schedule_fn = wrapper_scheduler_fn

    global _torch_prof
    _torch_prof = prof

def prepare_torch_profiler(schedule_fn):
    global _torch_prof
    _torch_prof = torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
                                         schedule=schedule_fn,
                                         profile_memory=True,
                                         on_trace_ready=torch.profiler.tensorboard_trace_handler(os.getenv("PROFILER_RESULT_FILE"))) 
    _torch_prof.start()
    log("finish setup profiler") 


def pre_hook():
    log("start setup profiler")
    
    # https://pytorch.org/docs/stable/profiler.html#torch.profiler.schedule
    hook_wait = int(os.getenv("PROFILER_WAIT", default="0"))
    hook_warmup = int(os.getenv("PROFILER_WARMUP", default="0"))
    hook_active = int(os.getenv("PROFILER_ACTIVE", default="10"))
    hook_repeat = int(os.getenv("PROFILER_REPEAT", default="1"))
    hook_skip_first = int(os.getenv("PROFILER_SKIP_FIRST", default="5"))
    assert (hook_repeat > 0 and hook_skip_first > 0), "Invalid profiler schedule arguments, PROFILER_REPEAT AND PROFILER_SKIP_FIRT must be >0"

    global _hook_start_step, _hook_end_step
    _hook_start_step = hook_skip_first + hook_wait + hook_warmup
    _hook_end_step = hook_skip_first + hook_repeat*(hook_wait+hook_warmup+hook_active)

    schedule_fn = torch.profiler.schedule(skip_first=hook_skip_first, wait=hook_wait, warmup=hook_warmup, active=hook_active, repeat=hook_repeat)
    optim.register_optimizer_step_post_hook(_after_step)

    if _get_profiler() == "nsys":
        prepare_nsys_profiler(schedule_fn)
    else:
        prepare_torch_profiler(schedule_fn)


def post_hook():
    global _torch_prof
    if _torch_prof is not None:
        _torch_prof.stop()
