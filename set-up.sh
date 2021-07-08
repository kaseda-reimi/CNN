#sudo apt-get update
#sudo apt-get install -yq git python3.7 python3-venv
#git clone 

git config --global user.name "kaseda-reimi"

sudo apt-get install tmux

python3 -m venv .venv

. .venv/bin/activate

pip install --upgrade pip
pip install numpy
pip install matplotlib
pip install tensorflow
pip install scikit-learn


#tmux
#ulimit 