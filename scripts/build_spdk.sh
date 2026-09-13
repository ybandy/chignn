git clone https://github.com/spdk/spdk
pushd spdk
git checkout eccfee9a908175f87fb38527195245f79d181fdb
git submodule update --init --recursive
sudo scripts/pkgdep.sh
#sudo apt install -y meson python3-pyelftools
pip install pyelftools
./configure \
    --without-idxd --without-crypto --without-fio --without-xnvme --without-vhost --without-virtio --without-vfio-user --without-dpdk-compressdev --without-rbd --without-ublk --without-rdma --without-fc --without-daos --without-iscsi-initiator --without-vtune --without-ocf --without-uring --without-dpdk-uadk --without-uring-zns --without-fsdev --without-nvme-cuse --without-raid5f --without-wpdk --without-usdt --without-sma --without-avahi --without-golang --without-aio-fsdev
make
popd

mkdir -p spdk_lib
pushd spdk_lib
python3 ../scripts/combine_spdk_libs.py
popd
