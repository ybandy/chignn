#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "spdk_util.h"

#define BLOCK_SIZE (512)


namespace spdk_util {


struct ctrlr_entry {
	struct spdk_nvme_ctrlr *ctrlr;
	char name[1024];
};

static std::vector<struct ctrlr_entry *> g_controllers;
static std::vector<struct drive *> g_namespaces;
static int g_num_qpairs;


static void register_ns(struct spdk_nvme_ctrlr *ctrlr, struct spdk_nvme_ns *ns)
{
	if (!spdk_nvme_ns_is_active(ns)) {
		return;
	}

	struct drive *entry = (struct drive *)malloc(sizeof(struct drive));
	if (entry == NULL) {
		perror("drive malloc");
		exit(1);
	}

	entry->ctrlr = ctrlr;
	entry->ns = ns;
	g_namespaces.push_back(entry);

	uint32_t sector_size = spdk_nvme_ns_get_sector_size(ns);
	assert(sector_size == BLOCK_SIZE);

	printf("  Namespace ID: %d size: %juGB (sector size %uB)\n", spdk_nvme_ns_get_id(ns),
	       spdk_nvme_ns_get_size(ns) / 1000000000, sector_size);
}

static bool probe_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	                 struct spdk_nvme_ctrlr_opts *opts)
{
	printf("Attaching to %s\n", trid->traddr);
	return true; // false indicates not to use this device
}

static void attach_cb(void *cb_ctx, const struct spdk_nvme_transport_id *trid,
	                  struct spdk_nvme_ctrlr *ctrlr, const struct spdk_nvme_ctrlr_opts *opts)
{
	struct ctrlr_entry *entry = (struct ctrlr_entry *)malloc(sizeof(struct ctrlr_entry));
	if (entry == NULL) {
		perror("ctrlr_entry malloc");
		exit(1);
	}
	printf("Attached to %s\n", trid->traddr);

	const struct spdk_nvme_ctrlr_data *cdata = spdk_nvme_ctrlr_get_data(ctrlr);
	snprintf(entry->name, sizeof(entry->name), "%-20.20s (%-20.20s)", cdata->mn, cdata->sn);

	entry->ctrlr = ctrlr;
	g_controllers.push_back(entry);

	for (int nsid = spdk_nvme_ctrlr_get_first_active_ns(ctrlr); nsid != 0;
	     nsid = spdk_nvme_ctrlr_get_next_active_ns(ctrlr, nsid)) {
		struct spdk_nvme_ns *ns = spdk_nvme_ctrlr_get_ns(ctrlr, nsid);
		if (ns == NULL) {
			continue;
		}
		register_ns(ctrlr, ns);
	}
}


int init(int num_qpairs, int queue_depth, const char *lcore_map)
{
	g_num_qpairs = num_qpairs;

	struct spdk_env_opts opts;
	opts.opts_size = sizeof(opts);
	spdk_env_opts_init(&opts);
	opts.name = "spdk_util";
	opts.core_mask = NULL;
	opts.lcore_map = lcore_map;
	if (spdk_env_init(&opts) < 0) {
		fprintf(stderr, "Unable to initialize SPDK env\n");
		return 1;
	}

	printf("Initializing NVMe Controllers\n");
	struct spdk_nvme_transport_id trid = {};
	spdk_nvme_trid_populate_transport(&trid, SPDK_NVME_TRANSPORT_PCIE);
	snprintf(trid.subnqn, sizeof(trid.subnqn), "%s", SPDK_NVMF_DISCOVERY_NQN);
	int rc = spdk_nvme_probe(&trid, NULL, probe_cb, attach_cb, NULL);
	if (rc != 0) {
		fprintf(stderr, "spdk_nvme_probe() failed\n");
		return rc;
	}

	if(g_controllers.empty()) {
		fprintf(stderr, "no NVMe controllers found\n");
		return 1;
	}

	// make sure drive IDs are consistent across runs
	std::sort(g_namespaces.begin(), g_namespaces.end(),
	          [](const struct drive *a, const struct drive *b) {
			      const struct spdk_nvme_ctrlr_data *d_a = spdk_nvme_ctrlr_get_data(a->ctrlr);
			      const struct spdk_nvme_ctrlr_data *d_b = spdk_nvme_ctrlr_get_data(b->ctrlr);
				  return strncmp(reinterpret_cast<const char*>(d_a->sn),
				                 reinterpret_cast<const char*>(d_b->sn), 20) < 0; // sort by drive serial numbers
			  });

	for(size_t j = 0; j < g_namespaces.size(); j++)
	{
		auto drive = g_namespaces[j];
		const struct spdk_nvme_ctrlr_data *cdata = spdk_nvme_ctrlr_get_data(drive->ctrlr);
		printf("Drive %2.1lu: %-20.20s (%-20.20s)\n", j, cdata->mn, cdata->sn);

		struct spdk_nvme_io_qpair_opts opts;
		spdk_nvme_ctrlr_get_default_io_qpair_opts(drive->ctrlr, &opts, sizeof(opts));
		if (opts.io_queue_requests < queue_depth) {
			opts.io_queue_requests = queue_depth;
		}
		opts.delay_cmd_submit = true; // important for performance!!

		drive->qpairs = (struct spdk_nvme_qpair**)malloc(sizeof(struct spdk_nvme_qpair*) * num_qpairs);
		for(int i = 0; i < num_qpairs; i++) {
			drive->qpairs[i] = spdk_nvme_ctrlr_alloc_io_qpair(drive->ctrlr, &opts, sizeof(opts));
			if (drive->qpairs[i] == NULL) {
				printf("ERROR: spdk_nvme_ctrlr_alloc_io_qpair() failed\n");
				return 2;
			}
		}
	}

	printf("Initialization complete.\n");
	return 0;
}

int num_drives()
{
    return static_cast<int>(g_namespaces.size());
}

struct drive* get_drive(int drive_id)
{
    return g_namespaces[drive_id];
}

void fini()
{
	struct spdk_nvme_detach_ctx *detach_ctx = NULL;

	for(auto drive : g_namespaces) {
		for(int i = 0; i < g_num_qpairs; i++) {
			spdk_nvme_ctrlr_free_io_qpair(drive->qpairs[i]);
		}
		free(drive->qpairs);
		free(drive);
	}

	for(auto ctrlr_entry : g_controllers) {
		spdk_nvme_detach_async(ctrlr_entry->ctrlr, &detach_ctx);
		free(ctrlr_entry);
	}

	if (detach_ctx) {
		spdk_nvme_detach_poll(detach_ctx);
	}

 	spdk_env_fini();
}

} // namespace spdk_util
