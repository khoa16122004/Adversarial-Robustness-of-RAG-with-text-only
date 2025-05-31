#!/bin/bash

# Cập nhật hệ thống
sudo apt update && sudo apt upgrade -y

# Cài đặt screen
sudo apt install screen -y


# Tạo file requirements.txt
cat <<EOF > requirements.txt
pymoo
textattack
tqdm
nvitop
EOF

# Cài đặt các gói Python
pip install -r requirements.txt