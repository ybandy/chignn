#include <iostream>
#include <fstream>
#include <regex>

#include "npy_reader.h"


int NpyReader::open(const char* input_file, bool print_info)
{
    std::ifstream file(input_file);
    if(!file.is_open())
    {
        fprintf(stderr, "failed to open file: %s", input_file);
        return 1;
    }
    std::string header;
    std::getline(file, header);
    file.close();

    close();
    if((fp = fopen(input_file, "rb")) == NULL)
    {
        fprintf(stderr, "failed to open file: %s", input_file);
        return 1;
    }

    // Get prefix length for computing feature offset,
    // add one for new-line character.
    header_size = header.size() + 1;

    const std::regex rx{ "<f(\\d+).+\\((\\d+),\\s*(\\d+)\\)" };
    std::smatch match;
    if(!std::regex_search(header, match, rx) || match.size() != 3 + 1)
    {
        fprintf(stderr, "failed to parse header\n");
        return 2;
    }
    const int dtype = stoi(match[1]);
    num_nodes = stoll(match[2]);
    const int dim = stoi(match[3]);
    if(print_info)
    {
        printf("%lu nodes, %d dimensions in %s\n", num_nodes, dim, (dtype == 2 ? "float16" : "float32"));
    }

    feature_size = dim * dtype;
    return 0;
}

void NpyReader::close()
{
    if(fp) fclose(fp);
    fp = nullptr;
}

int NpyReader::read_feature(uint64_t node_id, void *out)
{
    if(!fp) return 3;
    off_t offset = header_size + feature_size * node_id;
    if(fseeko(fp, offset, SEEK_SET) != 0)
    {
        fprintf(stderr, "seek error\n");
        return 1;
    }
    if(fread(out, feature_size, 1, fp) != 1)
    {
        fprintf(stderr, "read error\n");
        return 2;
    }
    return 0;
}
