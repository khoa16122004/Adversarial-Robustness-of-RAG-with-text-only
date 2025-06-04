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
gdown

pip install -r requirements.txt
gdown --folder 11WyPiasrLD-LLPIN3eJnCijjIfe_sVCA
