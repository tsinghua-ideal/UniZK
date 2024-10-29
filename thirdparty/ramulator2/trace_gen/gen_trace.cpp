#include <fstream>
#include <vector>
#include <cstdint>
#include <iostream>

enum OpRecordType {
    READ = 0,
    WRITE = 1
};

class OpRecord {
public:
    OpRecord(int id, OpRecordType type, int delay, const std::vector<int>& dependencies, uint64_t addr, int size)
        : id(id), type(type), delay(delay), dependencies(dependencies), addr(addr), size(size) {}

    int id;
    OpRecordType type;
    int delay;
    std::vector<int> dependencies;
    uint64_t addr;
    int size;
};
const char magic_word[8] = "BINFILE";
const uint32_t bufsize = 8;

void append_record(std::ofstream& file, const OpRecord& op) {
    file.write(reinterpret_cast<const char*>(&op.id), bufsize);
    file.write(reinterpret_cast<const char*>(&op.addr), bufsize);
    file.write(reinterpret_cast<const char*>(&op.type), bufsize);
    file.write(reinterpret_cast<const char*>(&op.delay), bufsize);
    file.write(reinterpret_cast<const char*>(&op.size), bufsize);

    uint64_t dep_len = op.dependencies.size();
    file.write(reinterpret_cast<const char*>(&dep_len), bufsize);
    for (int dep : op.dependencies) {
        file.write(reinterpret_cast<const char*>(&dep), bufsize);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " dep_count output.bin\n";
        return -1;
    }

    int dep_count = std::stoi(argv[1]);
    int nRecord = 100;
    std::vector<OpRecord> recs;

    std::ofstream file(argv[2], std::ios::binary);
    if (!file) {
        std::cerr << "Unable to open file for writing.\n";
        return 0;
    }
    file.write(magic_word, bufsize);

    for (int i = 0; i < nRecord; ++i) {
        std::vector<int> deps;
        if (i >= dep_count) {
            for (int dep = 1; dep < dep_count; ++dep) {
                deps.push_back(i / dep_count * dep_count - dep);
            }
        }
        append_record(file, OpRecord(i, READ, /* delay */ 0, deps, 0x1000 + i * 64, 64));
    }

    file.close();
    return 0;
}
