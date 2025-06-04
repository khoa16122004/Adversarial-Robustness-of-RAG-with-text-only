#!/bin/bash
sudo apt update && sudo apt upgrade -y
sudo apt install screen -y

cat <<EOF > requirements.txt
pymoo
textattack
tqdm
nvitop
EOF
streamlit

pip install -r requirements.txt