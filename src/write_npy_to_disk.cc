#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <vector>
#include <chrono>
#include <thread>

#include "spdk_util.h"
#include "spdk/likely.h"

#include "npy_reader.h"

#define BLOCK_SIZE (512UL)


using spdk_util::drive;


struct work_param {
    uint32_t core;
    uint32_t queue_id;
    uint32_t drive_id;
    uint32_t num_contexts;
    uint64_t begin;
    uint64_t end;
    int *node_ids;
    void *data;
    uint64_t feature_size;
    uint64_t num_ios_processed;
    uint64_t elapsed_time_nsec;
};

struct qpair_context {
    struct spdk_nvme_ns *ns;
    struct spdk_nvme_qpair *qpair;
    uint64_t last_index;
	uint64_t num_submitted;
	uint64_t num_completed;
    int *node_ids;
    void *data;
    uint64_t last_data_index;
};

struct io_context {
    struct qpair_context *qctx;
    uint32_t drive_id;
    uint32_t num_drives;
    uint64_t end;
	uint64_t index;
    uint64_t data_index;
    uint64_t feature_size;
    uint64_t unused;
	void *buf;
};


static int g_mode = 0;
static uint64_t max_lba = UINT64_MAX;


static std::chrono::nanoseconds::rep get_nsec()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static void set_affinity(int core_id)
{
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(core_id, &cpu_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
}


static void submit_io(struct io_context *ctx);

static void read_complete(void *arg, const struct spdk_nvme_cpl *completion)
{
	struct io_context *ctx = (struct io_context *)arg;
    struct qpair_context *qctx = ctx->qctx;
    const uint64_t feature_size = ctx->feature_size;
    uint64_t index = qctx->last_index + 1;

    if(spdk_unlikely(spdk_nvme_cpl_is_error(completion)))
    {
		fprintf(stderr, "I/O error status: %s\n", spdk_nvme_cpl_get_status_string(&completion->status));
		fprintf(stderr, "Read I/O failed, aborting run\n");
		exit(1);
	}
    
    const uint32_t num_drives = ctx->num_drives;
    if(g_mode != 1)
    {
        memcpy((uint8_t *)qctx->data + ctx->data_index * feature_size, ctx->buf, feature_size);
    }
    qctx->num_completed++;

    int *node_ids = qctx->node_ids;
    for(; index < ctx->end; index++) {
        if(ctx->drive_id == node_ids[index] % num_drives) break;
    }
    if(spdk_likely(index < ctx->end)) {
        ctx->index = index;
        ctx->data_index = qctx->last_data_index + 1;
        submit_io(ctx);
    } else {
      	qctx->last_index = ctx->end;
        spdk_free(ctx->buf);
        free(ctx);
    }
}


static __thread uint64_t seed = 0;

static void submit_io(struct io_context *ctx)
{
    struct qpair_context *qctx = ctx->qctx;
    const uint32_t num_drives = ctx->num_drives;
    uint64_t node_id = qctx->node_ids[ctx->index];
    assert(ctx->drive_id == node_id % num_drives);
    uint64_t lba = g_mode == 2 ? spdk_rand_xorshift64(&seed) % max_lba : node_id / num_drives;
    qctx->num_submitted++;
    qctx->last_index = ctx->index;
    qctx->last_data_index = ctx->data_index;
    int rc = spdk_nvme_ns_cmd_read(qctx->ns, qctx->qpair,
                                   ctx->buf,
                                   lba, 1, read_complete, ctx, 0);
    if(spdk_unlikely(rc != 0))
    {
        fprintf(stderr, "starting read I/O failed\n");
        exit(1);
    }
}

static void *read_data(void *param)
{
    struct work_param *p = (struct work_param*)param;
    const uint32_t queue_id = p->queue_id;
    const uint32_t drive_id = p->drive_id;
    const uint32_t num_contexts = p->num_contexts;
    const uint32_t num_drives = spdk_util::num_drives();
    const uint64_t feature_size = p->feature_size;

    set_affinity(p->core);

    seed = spdk_rand_xorshift64_seed();

    struct drive *drive = spdk_util::get_drive(drive_id);
    const int32_t numa_id = spdk_nvme_ctrlr_get_numa_id(drive->ctrlr);

    struct qpair_context qctx;
    qctx.last_index = 0;
    qctx.num_submitted = 0;
    qctx.num_completed = 0;
    qctx.ns = drive->ns;
    qctx.qpair = drive->qpairs[queue_id];
    qctx.node_ids = p->node_ids;
    qctx.data = p->data;
    //qctx.last_data_index = num_contexts;

    auto start_time = get_nsec();

    uint64_t index = p->begin;
    const uint64_t end = p->end;
    for(uint32_t i = 0; i < num_contexts; i++)
    {
        for(; index < end; index++) {
            if(drive_id == p->node_ids[index] % num_drives) break;
        }
        if(index >= end) break;

        struct io_context *ctx = (struct io_context *)malloc(sizeof(struct io_context));
        ctx->drive_id = drive_id;
        ctx->num_drives = num_drives;
        ctx->end = end;
        ctx->index = index;
        ctx->qctx = &qctx;
        ctx->data_index = i;
        ctx->feature_size = feature_size;
        ctx->buf = spdk_zmalloc(BLOCK_SIZE, BLOCK_SIZE, NULL, numa_id, SPDK_MALLOC_DMA);
        submit_io(ctx);
        index++;
    }

    while(true)
    {
        if(spdk_unlikely(qctx.last_index >= end && qctx.num_completed >= qctx.num_submitted))
        {
            break;
        }
        spdk_nvme_qpair_process_completions(qctx.qpair, 0);
    }

    //spdk_nvme_ctrlr_free_io_qpair(qctx.qpair);
    assert(qctx.num_submitted == qctx.num_completed);
    p->num_ios_processed = qctx.num_completed;
    p->elapsed_time_nsec = static_cast<uint64_t>(get_nsec() - start_time);
    return NULL;
}


struct write_param
{
    uint32_t drive_id;
    const char *input_file;
    uint64_t mask;
};

static void write_complete(void *arg, const struct spdk_nvme_cpl *completion)
{
	int *is_completed = (int*)arg;

	if(spdk_nvme_cpl_is_error(completion))
    {
		fprintf(stderr, "I/O error status: %s\n", spdk_nvme_cpl_get_status_string(&completion->status));
		fprintf(stderr, "Write I/O failed, aborting run\n");
		*is_completed = 2;
		exit(1);
	}

	*is_completed = 1; // if no error, then complete
}

static void *write_data(void *param)
{
    struct write_param *p = (struct write_param *)param;
    const uint64_t mask = p->mask;
    const uint32_t drive_id = p->drive_id;
    struct drive *drive = spdk_util::get_drive(drive_id);
    const int num_drives = spdk_util::num_drives();

    set_affinity(drive_id); // use drive_id as core ID

    NpyReader reader;
    reader.open(p->input_file);
    const size_t num_nodes = reader.get_num_nodes();

    const uint32_t num_features_to_batch = 64;
    int32_t numa_id = spdk_nvme_ctrlr_get_numa_id(drive->ctrlr);
    void *buf = spdk_zmalloc(BLOCK_SIZE * num_features_to_batch, BLOCK_SIZE, NULL, numa_id, SPDK_MALLOC_DMA);

    uint64_t lba = 0;
    uint64_t node_id = drive_id;
    while(node_id < num_nodes)
    {
        for(uint32_t i = 0; i < num_features_to_batch; i++)
        {
            reader.read_feature(node_id, (uint8_t *)buf + i * BLOCK_SIZE);
            node_id += num_drives;
            if(node_id >= num_nodes) break;
        }
        int is_completed = 0;
        int rc = spdk_nvme_ns_cmd_write(drive->ns, drive->qpairs[0], buf,
		        						lba, num_features_to_batch,
				    					write_complete, &is_completed, 0);
        if(rc != 0)
        {
            fprintf(stderr, "starting write I/O failed\n");
            exit(1);
        }
    	lba += num_features_to_batch;
        while(!is_completed)
        {
            spdk_nvme_qpair_process_completions(drive->qpairs[0], 0);
        }
    }
    spdk_free(buf);
    reader.close();
    return NULL;
}


struct stats_param
{
    uint64_t begin;
    uint64_t end;
    int *node_ids;
    int *index_map;
    void *data;
    const char *input_file;
    int num_mismatches;
};

static void *count_mismatches(void *param)
{
    struct stats_param *p = (struct stats_param*)param;
    const uint64_t begin = p->begin;
    const uint64_t end = p->end;
    int *node_ids = p->node_ids;
    int *index_map = p->index_map;
    uint8_t *data = (uint8_t *)p->data;

    NpyReader reader;
    reader.open(p->input_file);
    const size_t feature_size = reader.get_feature_size();
    uint8_t *feature = new uint8_t [feature_size];

    int num_mismatches = 0;
    for(uint64_t i = begin; i < end; i++)
    {
        const int node_id = node_ids[i];
        if(reader.read_feature(node_id, feature) != 0)
        {
            fprintf(stderr, "error reading %lu-th feature\n", i);
            reader.close();
            num_mismatches = -1;
            break;
        }

        uint8_t *buf = data + (uint64_t)index_map[i] * feature_size;
        if(memcmp(buf, feature, feature_size))
        {
            num_mismatches++;
        }
    }
    delete [] feature;
    reader.close();
    p->num_mismatches = num_mismatches;
    return NULL;
}

static void check_results(uint64_t count, int *node_ids, int *index_map, void *result, const char *input_file)
{
    auto start = get_nsec();

    const int num_threads = 32;
    struct stats_param *params = (struct stats_param *)malloc(num_threads * sizeof(struct stats_param));
    pthread_t tids[num_threads];
    for(int i = 0; i < num_threads; i++)
    {
        params[i].begin = count * i / num_threads;
        params[i].end = count * (i + 1) / num_threads;
        params[i].input_file = input_file;
        params[i].node_ids = node_ids;
        params[i].index_map = index_map;
        params[i].data = result;
        pthread_create(&tids[i], NULL, count_mismatches, &params[i]);
    }
    for(int i = 0; i < num_threads; i++) pthread_join(tids[i], NULL);

    int num_mismatches = 0;
    for(int i = 0; i < num_threads; i++)
    {
        if(params[i].num_mismatches < 0)
        {
            fprintf(stderr, "error while counting mismatches\n");
            exit(1);
        }
        num_mismatches += params[i].num_mismatches;
    }
    free(params);

    auto end = get_nsec();
    auto t = end - start;
    printf("# mismatches = %d  (counted in %f sec)\n", num_mismatches, t * 1e-9);
}

int main(int argc, char *argv[])
{
    if(argc < 6)
    {
        fprintf(stderr, "%s input.npy test_count verify num_contexts num_qpairs mode\n", argv[0]);
        fprintf(stderr, "  test_count=0: write\n");
        fprintf(stderr, "  test_count>0: random read test\n");
        fprintf(stderr, "  mode=0: normal\n");
        fprintf(stderr, "  mode=1: no memcpy\n");
        fprintf(stderr, "  mode=2: random LBA\n");
        return 0;
    }
    const char *input_file = argv[1];
    const int test_count = atoi(argv[2]);
    const bool verify = atoi(argv[3]) > 0;
    const int num_contexts = atoi(argv[4]);
    const int num_qpairs = atoi(argv[5]);
    g_mode = argc >= 7 ? atoi(argv[6]) : 0;

    unsigned int num_lcores = std::thread::hardware_concurrency();
    char lcore_map[128];
    sprintf(lcore_map, "0-%d", num_lcores - 1);
    printf("lcore_map: %s\n", lcore_map);
	if(spdk_util::init(num_qpairs, num_contexts, lcore_map))
    {
		fprintf(stderr, "failed to initialize SPDK\n");
		return 1;
    }
    const int num_drives = spdk_util::num_drives();
    for(int i = 0; i < num_drives; i++)
    {
        struct spdk_nvme_ns *ns = spdk_util::get_drive(i)->ns;
        uint64_t ns_size = spdk_nvme_ns_get_size(ns);
	    uint32_t sector_size = spdk_nvme_ns_get_sector_size(ns);
        printf("Drive %d: capacity %lu GB, sector size %u\n", i, ns_size / 1000000000UL, sector_size);
        assert(sector_size == BLOCK_SIZE);
        if(max_lba > ns_size) max_lba = ns_size;
    }
    max_lba /= BLOCK_SIZE;
    printf("Max LBA: %lu\n", max_lba);

    spdk_unaffinitize_thread(); // allow other threads to run on arbitrary cores

    NpyReader reader;
    reader.open(input_file, true);
    const size_t num_nodes = reader.get_num_nodes();
    const size_t feature_size = reader.get_feature_size();
    assert(feature_size <= BLOCK_SIZE);
    reader.close();

    if(test_count == 0)
    {
        auto start = get_nsec();

        struct write_param *params = (struct write_param *)malloc(num_drives * sizeof(struct write_param));
        pthread_t tids[num_drives];
        for(uint64_t j = 0; j < 1; j++)
        {
            for(int i = 0; i < num_drives; i++)
            {
                params[i].mask = j;
                params[i].drive_id = i;
                params[i].input_file = input_file;
                pthread_create(&tids[i], NULL, write_data, &params[i]);
            }
            for(int i = 0; i < num_drives; i++) pthread_join(tids[i], NULL);
        }
        free(params);

        auto end = get_nsec();
        auto t = end - start;
        printf("Wrote data in %f sec (%.2f MIOPS)\n", t * 1e-9, num_nodes * 1000.0 / (double)t);
    }

    const uint64_t count = test_count == 0 ? num_nodes : test_count;
    int *node_ids = (int *)malloc(count * sizeof(int));
    srand(time(NULL));
    for(int i = 0; i < count; i++)
    {
        node_ids[i] = test_count == 0 ? i : (int)(rand() % num_nodes);
    }
    void *result = malloc(count * feature_size);
    if(result == NULL)
    {
        fprintf(stderr, "failed to allocate data: void *result\n");
        exit(1);
    }

    int *result_index = (int *)calloc(num_drives + 1, sizeof(int));
    // count the number of nodes to be fetched from each drive
    //for(int i = 0; i < count; i++)
    //{
    //    result_index[(node_ids[i] % num_drives) + 1]++;
    //}
    //for(int i = 0; i <= num_drives; i++) printf("%d\n", result_index[i]);
    // the following replaces the above code by taking multiple queue pairs into account
    int **queue_index = (int **)malloc(num_qpairs * sizeof(int *));
    for(int queue_id = 0; queue_id < num_qpairs; queue_id++)
    {
        queue_index[queue_id] = (int *)malloc(num_drives * sizeof(int));
        // record the intermediate counting state
        // this should come before updating result_index
        for(int i = 0; i < num_drives; i++)
        {
            queue_index[queue_id][i] = result_index[i+1];
        }
        uint64_t begin = count * queue_id / num_qpairs;
        uint64_t end = count * (queue_id + 1) / num_qpairs;
        for(uint64_t i = begin; i < end; i++)
        {
            result_index[(node_ids[i] % num_drives) + 1]++;
        }
    }
    // accumulate to get the array indices
    // example with 10 nodes and 4 drives
    // 0  3  4  2  1 (initial given above)
    // 0  3  4  2  3
    // 0  3  4  6  7
    // 0  3  7  9 10 (final array indices)
    for(int i = num_drives - 1; i > 0; i--)
    {
        for(int j = i + 1; j <= num_drives; j++) result_index[j] += result_index[i];
    }

    auto start = get_nsec();

    const int num_threads = num_drives * num_qpairs;
    struct work_param *params = (struct work_param *)malloc(num_threads * sizeof(struct work_param));
    pthread_t tids[num_threads];
    int i = 0;
    for(int drive_id = 0; drive_id < num_drives; drive_id++)
    {
        for(int queue_id = 0; queue_id < num_qpairs; queue_id++)
        {
            params[i].core = i;
            params[i].queue_id = queue_id;
            params[i].drive_id = drive_id;
            params[i].begin = count * queue_id / num_qpairs;
            params[i].end = count * (queue_id + 1) / num_qpairs;
            params[i].node_ids = node_ids;
            params[i].num_contexts = num_contexts;
            params[i].feature_size = feature_size;
            params[i].data = (uint8_t *)result + (uint64_t)(result_index[drive_id] + queue_index[queue_id][drive_id]) * feature_size;
            pthread_create(&tids[i], NULL, read_data, &params[i]);
            i++;
        }
    }
    for(i = 0; i < num_threads; i++) pthread_join(tids[i], NULL);

    auto end = get_nsec();
    auto t = end - start;
    printf("%.2f MIOPS (%f sec)\n", count * 1000.0 / (double)t, t * 1e-9);

    uint64_t sum_ios = 0;
    double sum_miops = 0;
    for(int i = 0; i < num_threads; i++)
    {
        double miops = params[i].num_ios_processed * 1000.0 / (double)params[i].elapsed_time_nsec;
        printf("drive %d, queue %d: %lu IOs processed (%.2f MIOPS)\n", params[i].drive_id, params[i].queue_id, params[i].num_ios_processed, miops);
        sum_ios += params[i].num_ios_processed;
        sum_miops += miops;
    }
    printf("%lu IOs, %.2f MIOPS\n", sum_ios, sum_miops);
    //assert(sum_ios == count);
    free(params);

    if(verify)
    {
        int *index_map = (int *)malloc(count * sizeof(int));
        for(int i = 0; i < count; i++)
        {
            int drive_id = node_ids[i] % num_drives;
            index_map[i] = result_index[drive_id];
            result_index[drive_id]++;
            //printf("%d, %d\n", node_ids[i], index_map[i]);
        }
        check_results(count, node_ids, index_map, result, input_file);
        free(index_map);
    }

    free(node_ids);
    free(result);
    free(result_index);
    for(int i = 0; i < num_qpairs; i++) free(queue_index[i]);
    free(queue_index);
	spdk_util::fini();
    return 0;
}
