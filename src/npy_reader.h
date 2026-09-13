#include <stdio.h>
#include <stdint.h>

class NpyReader
{
public:
    int open(const char* input_file, bool print_info = false);
    void close();
    int read_feature(uint64_t node_id, void *out);
    size_t get_feature_size() { return feature_size; };
    size_t get_num_nodes() { return num_nodes; };

private:
    size_t header_size;
    size_t feature_size;
    size_t num_nodes;
    FILE *fp = nullptr;
};
