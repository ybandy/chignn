import re, os
from pathlib import Path


SPDK_LIB_DIR = Path('../spdk/build/lib')

SPDK_LIBS = \
"""
#LDFLAGS += -lspdk_sock_posix
#LDFLAGS += -lspdk_thread
LDFLAGS += -lspdk_nvme
LDFLAGS += -lspdk_keyring
LDFLAGS += -lspdk_sock
LDFLAGS += -lspdk_trace
LDFLAGS += -lspdk_rpc
LDFLAGS += -lspdk_jsonrpc
LDFLAGS += -lspdk_json
LDFLAGS += -lspdk_dma
#LDFLAGS += -lspdk_vmd
LDFLAGS += -lspdk_util
LDFLAGS += -lspdk_log
LDFLAGS += -lspdk_env_dpdk
"""


DPDK_LIB_DIR = Path('../spdk/dpdk/build/lib')

DPDK_LIBS = \
"""
LDFLAGS += -lrte_bus_pci
#LDFLAGS += -lrte_bus_vdev
#LDFLAGS += -lrte_cmdline
#LDFLAGS += -lrte_compressdev
#LDFLAGS += -lrte_cryptodev
#LDFLAGS += -lrte_dmadev
LDFLAGS += -lrte_eal
#LDFLAGS += -lrte_ethdev
#LDFLAGS += -lrte_hash
LDFLAGS += -lrte_kvargs
LDFLAGS += -lrte_log
#LDFLAGS += -lrte_mbuf
LDFLAGS += -lrte_mempool
#LDFLAGS += -lrte_mempool_ring
#LDFLAGS += -lrte_meter
#LDFLAGS += -lrte_net
LDFLAGS += -lrte_pci
#LDFLAGS += -lrte_power
#LDFLAGS += -lrte_power_acpi
#LDFLAGS += -lrte_power_amd_pstate
#LDFLAGS += -lrte_power_cppc
#LDFLAGS += -lrte_power_intel_pstate
#LDFLAGS += -lrte_power_intel_uncore
#LDFLAGS += -lrte_power_kvm_vm
#LDFLAGS += -lrte_rcu
#LDFLAGS += -lrte_reorder
LDFLAGS += -lrte_ring
#LDFLAGS += -lrte_security
LDFLAGS += -lrte_telemetry
#LDFLAGS += -lrte_timer
#LDFLAGS += -lrte_vhost
"""


def my_system(cmd):
	print(cmd)
	os.system(cmd)

def combine_libraries(lib_dir, makefile_lib_str, combined_lib_name):
	if os.path.isfile(combined_lib_name):
		os.remove(combined_lib_name)
	libraries = []
	for line in makefile_lib_str.split('\n'):
		m = re.search(r'^LDFLAGS.+\-l(\S+)', line)
		if(m is not None):
			libraries.append(f'lib{m[1]}.a')
	libraries_str = [str(lib_dir.joinpath(lib)) for lib in libraries]
	for lib in libraries_str:
		my_system(f'ar x {lib}')
	my_system(f'ar rcs {combined_lib_name} *.o')
	my_system('rm *.o')

SPDK_LIB_NAME = 'libspdk.a'
combine_libraries(SPDK_LIB_DIR, SPDK_LIBS, SPDK_LIB_NAME)

DPDK_LIB_NAME = 'libdpdk.a'
combine_libraries(DPDK_LIB_DIR, DPDK_LIBS, DPDK_LIB_NAME)
