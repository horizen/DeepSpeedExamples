支持以下格式

### 示例1：
```shell
python -m torch.distributed.run $dist_args $training_script $training_args
```

### 示例2：
```shell
torchrun $dist_args $training_script $training_args
```

### 示例3：
```shell
deepspeed $dist_args $training_script $training_args
```

### 示例4：
```shell
horovodrun $dist_args $training_script $training_args
```

### 示例4：
```shell
mpirun $dist_args $training_script $training_args
```
