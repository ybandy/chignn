ORIG_PWD=${PWD}
export DGL_HOME=`realpath ${ORIG_PWD}/dgl`

#git clone https://github.com/dmlc/dgl.git
cd ${DGL_HOME}
#git checkout 3d16000b4170fa741ed9e9667f22ba84d3493026
#git submodule update --init --recursive

bash script/build_dgl.sh -g 2>&1 | tee ${ORIG_PWD}/build_dgl.log

cd python
python setup.py install
python setup.py build_ext --inplace

cd ${ORIG_PWD}
