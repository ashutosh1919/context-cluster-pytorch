# Installing Torch related packages
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1+cu116 torchtext==0.14.1 -f https://download.pytorch.org/whl/torch_stable.html

# Installing lxml
sudo apt-get install python-lxml

# Installing requirements.txt
pip install -r requirements.txt
pip install --upgrade lxml

# Cloning Context-Cluster repository for code usage
git clone https://github.com/ma-xu/Context-Cluster.git context_cluster