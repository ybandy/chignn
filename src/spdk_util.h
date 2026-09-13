#include <stdint.h>
#include <vector>

#include "spdk/nvme.h"

namespace spdk_util {


struct drive {
	struct spdk_nvme_ctrlr *ctrlr;
	struct spdk_nvme_ns *ns;
	struct spdk_nvme_qpair **qpairs;
};

int init(int num_qpairs, int queue_depth, const char *lcore_map);
int num_drives();
struct drive* get_drive(int drive_id);
void fini();

} // namespace spdk_util
