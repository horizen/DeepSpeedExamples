#/bin/bash

install_nsys() {
  nsys --version
  if [[ $? != "0" ]]; then
    lsb_release -a 2>&1 | grep "Ubuntu" &> /dev/null
    # Ubuntu Distribution
    if [ $? == 0 ]; then
      rm -rf /etc/apt/sources.list.d/cuda.list
      apt update && apt install -y --no-install-recommends gnupg
      echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
      apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
      apt update && apt install nsight-systems-cli
    fi
    cat /etc/redhat-release  &> /dev/null
    # CentOS/tlinux Distribution
    if [ $? == 0 ]; then
      rpm --import https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
      sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-*
      sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-*
      dnf install -y 'dnf-command(config-manager)'
      version=$(source /etc/os-release; echo $VERSION_ID)
      if [ -f /etc/tlinux-release ]; then
        if [[ "$version" == "2.4" ]]; then
          version=7
        elif [[ "$version" == "3.1" ]]; then
          version=8
        fi
      fi
      dnf config-manager --add-repo "https://developer.download.nvidia.com/devtools/repos/rhel${version}/$(rpm --eval '%{_arch}' | sed s/aarch/arm/)/"
      dnf install -y nsight-systems-cli
    fi
  fi
}

install_coscmd() {
  coscmd --version
  if [[ $? != "0" ]]; then
    pip install coscmd
  fi
}

get_script() {
  local script_file="$1"
  if [ ! -f "$script_file" ]; then
    echo "Error: File '$script_file' not found!" >> "$log_file"
    return 1
  fi

  {
    command_buffer=""
    while IFS= read -r line || [ -n "$line" ]; do
      if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
      fi
      if [[ "$line" =~ \\$ ]]; then
        command_buffer+="${line%\\} "
      else
        command_buffer+="$line"
        env_vars=$(echo "$command_buffer" | awk '{for(i=1;i<=NF;i++){if($i ~ /=/){printf "%s ", $i}else{break}}}')
        command=$(echo "$command_buffer" | awk '{for(i=1;i<=NF;i++){if($i !~ /=/){for(j=i;j<=NF;j++){printf "%s ", $j}; break}}}')
        cmd=$(echo "$command" | sed 's/^[ \t]*//' | cut -d ' ' -f 1)
        found=false
        if [[ $cmd == "cd" ]]; then
          $command_buffer
        elif [[ $cmd == "bash" ]] || [[ $cmd == "sh" ]] || [[ $cmd == "." ]] || [[ $cmd =~ ^.+.sh$ ]]; then
          called_script=$(echo "$command_buffer" | awk '{print $2}')
          echo into $called_script
          get_script $called_script
        elif [[ $cmd == "torchrun" ]] || [[ $cmd == "mpirun" ]] || [[ $cmd == "deepspeed" ]] || [[ $cmd == "horovodrun" ]]; then
          found=true
        elseø
          echo $command | grep -E -q "torch.distributed.run"
          if [[ "$?" == "0" ]] && ([[ $cmd =~ ^.*python$ ]] || [[ $cmd =~ ^.*python3$ ]]); then
            found=true
          fi
        fi
        if [[ $found == "true" ]]; then
          if [[ $TI_PROFILER == "nsys" ]]; then
            profiler_args="nsys profile -t cuda,nvtx,osrt \
                            --capture-range=cudaProfilerApi -s cpu --cudabacktrace=true \
                            --nic-metrics=true \
                            -x true -w true \
                            -o /tmp/nsys_profile"
                            #--python-sampling-frequency=2000 --python-backtrace=cuda --python-sampling=true \
                            #--cpuctxsw=process-tree --event-sample=system-wide --os-events=2,3,4,6 --event-sampling-interval=1000 \
          fi
          for arg in $command; do
            if [[ "$arg" =~ ^.*\.py$ ]]; then
              echo -e "from profiler import pre_hook, post_hook\npre_hook()" >> $train_file
              cat $arg >> $train_file
              echo "post_hook()" >> $train_file
              if [[ $cmd == "mpirun" ]] || [[ $cmd == "deepspeed" ]] || [[ $cmd == "horovodrun" ]]; then
                command=$(echo "$command" | sed "s#$arg#$profiler_args $train_file#g")
              else
                command=$(echo "$profiler_args $command" | sed "s#$arg#$train_file#g")
              fi
              break
            fi
          done
          command_buffer="PYTHONPATH=/tmp:$PYTHONPATH $env_vars NCCL_DEBUG=INFO NVSHMEM_NVTX=common $command"
        fi
        echo $command_buffer >> $cmd_file
        command_buffer=""
      fi
    done
  } < "$script_file"
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script_file>"
    exit 1
fi


install_coscmd
install_nsys

TI_PROFILER=${TI_PROFILER:-nsys}
train_file="/tmp/.start.py"
cmd_file="/tmp/.start.sh"
> $train_file
> $cmd_file

get_script "$1"

shift

. $cmd_file $@