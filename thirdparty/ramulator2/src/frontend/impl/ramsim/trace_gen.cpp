#include <unordered_set>
#include <type_traits>
#include <fstream>

#include "ramsim.h"
#include "frontend/frontend.h"

namespace Ramulator {

class TraceInputStream {
    private:
        std::ifstream file;
        uint64_t expectRecordId;
        uint64_t reqIdCnt;

    public:
        TraceInputStream(const std::string& traceFileName)
            : file(traceFileName.c_str(), std::ios::in | std::ios::binary) {
            reset();
        }

        bool is_open() const { return file.is_open(); }

        bool checkMagic() {
            const uint32_t size = 8;
            char magic[8];
            file.read(reinterpret_cast<char *>(&magic), size);
            return strcmp(magic, "BINFILE") == 0;
        }

        void reset() {
            file.clear();
            file.seekg(0);
            bool correctMagic = checkMagic();
            assert(correctMagic);
            expectRecordId = 0;
            reqIdCnt = 0;
        }

        bool parseRecord(MemReq* record, bool* eof) {
            *eof = file.peek() == EOF;
            if (*eof) return false;
            const uint32_t size = 8;
            file.read(reinterpret_cast<char *>(&record->id), size);
            file.read(reinterpret_cast<char *>(&record->addr), size);
            file.read(reinterpret_cast<char *>(&record->type), size);
            file.read(reinterpret_cast<char *>(&record->delay), size);
            file.read(reinterpret_cast<char *>(&record->size), size);
            uint64_t depCount;
            file.read(reinterpret_cast<char *>(&depCount), size);
            record->deps.resize(depCount);
            for (size_t i = 0; i < depCount; i++) {
                file.read(reinterpret_cast<char *>(&record->deps[i]), size);
            }
            record->ids_0 = reqIdCnt;
            record->ids.resize(ceil(record->size / 64.0));
            for (size_t i = 0; i < record->ids.size(); i++) {
                record->ids[i] = reqIdCnt;
                reqIdCnt++;
            }
            return true;
        }

        bool next(MemReq* record, bool* eof) {
            bool good = parseRecord(record, eof);

            if (good) {
                // Check continuous ID.
                if (record->id != expectRecordId) {
                    panic("OpRecord id is not continuous, should be 0x%lx but 0x%lx encountered.",
                            expectRecordId, record->id);
                }
                expectRecordId++;
            }

            return good;
        }
};

class TraceGen : public IFrontEnd, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, TraceGen, "TraceGen", "TraceGen")

private:
    // runtime states
    bool finish;
    int m_id;
    RamsimCallback m_callback;
    Logger_t m_logger;

    TraceInputStream* traceInput = nullptr;
    RamSim* ramsim = nullptr;
    std::vector<MemReq> pendReq; // a 1-entrry queue in case RamSim is stalled
    // parameter
    uint32_t maxPendEntry = 0;
    std::string traceFileName;

    bool handleRequest(MemReq& req) {
        // do nothing for a trace generator
        return true;
    };

    void readNextRecord() {
        if (pendReq.size() == 1) return;
        MemReq record;
        bool eof;
        if (traceInput->next(&record, &eof)) {
            pendReq.push_back(record);
        } else {
            if (!eof) panic("Trace file %s has error and has not ended properly.", traceFileName.c_str());
            finish = true;
        }
    }

public:
    void init() override {
        // initilize general config
        m_clock_ratio = param<uint>("clock_ratio").required();
        finish = false;
        m_id = 0;
        m_logger = Logging::create_logger("TraceGen");
        m_callback = [this](MemReq& req) { return this->handleRequest(req); };
        // initialize parameters
        traceFileName = param<std::string>("traceFileName").desc("traceFileName").required();
        // initialize trace input and ramsim
        traceInput = new TraceInputStream(traceFileName);
        if (!traceInput->is_open())
            panic("Trace file %s failed to open.", traceFileName.c_str());
        ramsim = dynamic_cast<RamSim*>(create_child_ifce<IFrontEnd>());
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
        ramsim->setup(this, m_callback, memory_system);
    }

    void tick() override {
        if (is_finished()) return;
        ramsim->tick();
        for (auto it = pendReq.begin() ; it < pendReq.end(); it++) {
            if (ramsim->send(*it)) {
                pendReq.erase(it);
            }
        }
        readNextRecord();
    };

    bool is_finished() override {
        if (finish && ramsim->is_finished())
            printf("TraceGen finished in cycle %ld\n", ramsim->get_cycles());
        return finish && ramsim->is_finished();
    };
};

} // namespace Ramulator