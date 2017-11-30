#!/bin/bash  
echo "start init"  
rm -rf /usr/bin/python
ln -s /usr/bin/python3.5 /usr/bin/python
apt-get install python-pip
apt-get install python3-pip
apt-get install python3-tk
git config --global user.name "wenliang303"
git config --global user.email 64457570@qq.com
pip install matplotlib
pip install h5py
pip install tensorflow
pip install scipy
pip install scikit-learn
pip install pillow
pip install Ipython
pip install pydot
pip install keras