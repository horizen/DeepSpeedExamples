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

if nvidia-smi >/dev/null 2>&1; then
    echo 'no driver install'
    exit 102
fi

install_nsys
nsys --version
if [[ $? != "0" ]]; then
    echo 'install nsight failed'
    exit 103
fi

install_coscmd
coscmd --version
if [[ $? != "0" ]]; then
    echo 'install coscmd failed'
    exit 104
fi