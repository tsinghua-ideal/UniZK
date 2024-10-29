#ifndef     RAMSIM_H
#define     RAMSIM_H

#include <unordered_set>
#include <type_traits>
#include <fstream>

#include "frontend/frontend.h"

#define print(args...) \
{ \
    fprintf(stdout, args); \
    fprintf(stdout, "\n"); \
    fflush(stdout); \
}

#define PANIC_EXIT_CODE (112)
#define panic(args...) \
{ \
    print(args); \
    exit(PANIC_EXIT_CODE); \
}

#define RAMSIM_READREQ 0
#define RAMSIM_WRITEREQ 1

namespace Ramulator {

struct MemReq {
    uint64_t id;
    Addr_t addr;
    int type;
    uint32_t delay;
    uint32_t size;
    std::vector<uint64_t> deps;
    
    uint64_t ids_0;
    std::vector<uint64_t> ids;
};

struct MemReqInfo {
    // request
    MemReq req;
    // runtime
    bool depSolved;
    Clk_t minIssueCycle;
};
using RamsimCallback = std::function<bool(MemReq&)>;

class RamSim : public IFrontEnd, public Implementation {
    RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, RamSim, "RamSim", "RamSim")

private:
    // runtime states
    bool finish;
    int m_id;
    std::function<void(Request&)> ramulator_handler;
    Logger_t m_logger;

    std::unordered_map<uint64_t, MemReqInfo> pendReqInfo;
    std::unordered_map<uint64_t, MemReq> inflightReq;
    std::vector<MemReq> pendResp;
    uint64_t maxReqId = 0;
    RamsimCallback m_callback; // default callback
    std::unordered_map<uint64_t, RamsimCallback> reqCallback;
    IMemorySystem* m_memory_system = nullptr;

    // parameter
    uint32_t maxPendEntry = 0;

    // stats
    uint64_t s_total_mem_req = 0;

    bool sendRequest(MemReq &req) {
        // m_logger->info("sendRequest id {} addr {} cycle {} ids_0 {} ids {} size {}", req.id, req.addr, m_clk, req.ids_0, req.ids.size(), req.size);
        assert(req.ids.size() != 0);
        for (auto it = req.ids.begin() ; it != req.ids.end();) {
            bool ok = m_memory_system->send(Request(req.addr + (*it - req.ids_0) * 64,
                                        req.type,
                                        m_id,
                                        ramulator_handler,
                                        req.id,
                                        req.ids.size() == 1));
            if (ok) {
                ++s_total_mem_req;
                it = req.ids.erase(it);
                // break;
            } else {
                // printf("not erase id %ld\n", *it);
                break;
            }
        }
        bool ok = req.ids.empty();
        if (ok) {
            if (req.type == RAMSIM_READREQ)
                inflightReq.emplace(req.id, req);
            else {
                assert(req.type == RAMSIM_WRITEREQ);
                pendResp.emplace_back(req);
            }
        }
        return ok;
    }

    bool checkDepSolved(MemReq &req) {
        for (auto& dep : req.deps) {
            if (pendReqInfo.count(dep) || inflightReq.count(dep)) {
                return false;
            }
        }
        return true;
    }

    void updateDepSolved(){
        for (auto& [reqId, it] : pendReqInfo) {
            if (it.depSolved) continue;
            if (checkDepSolved(it.req)) {
                // m_logger->info("id {} ready cycle {}", reqId, m_clk);
                it.depSolved = true;
                it.minIssueCycle = m_clk + it.req.delay;
            }
        }
    }

    void handleRequest(Request& req) {
        if (req.last_subid) {
            // m_logger->info("handleRequest id {} addr {}", req.id, req.addr);
            assert(inflightReq.count(req.id));
            assert(req.type_id == RAMSIM_READREQ);
            pendResp.emplace_back(inflightReq[req.id]);
            inflightReq.erase(req.id);
            updateDepSolved();
        }
    };

public:
    void init() override {
        // initilize general config
        finish = false;
        m_id = 0;
        m_logger = Logging::create_logger("RamSim");
        ramulator_handler = [this](Request& req) { return this->handleRequest(req); };
        m_clk = 0;
        // initialize parameters
        maxPendEntry = param<std::uint32_t>("maxPendEntry").desc("maxPendEntry").required();
        bool enableLogging = param<bool>("enableLogging").desc("enableLogging").default_val(false);
        if (!enableLogging) m_logger->set_level(spdlog::level::off);
        // initialize stats
        register_stat(s_total_mem_req).name("s_total_mem_req");
    };

    void setup(IFrontEnd* frontend, RamsimCallback callback, IMemorySystem* memory_system) {
        m_callback = callback;
        m_memory_system = memory_system;
    }

    bool send(MemReq& req) {
        return send(req, m_callback);
    }

    bool send(MemReq& req, RamsimCallback callback) {
        if (pendReqInfo.size() + pendResp.size() > maxPendEntry) return false;
        maxReqId = std::max(maxReqId, req.id);
        bool depSolved = checkDepSolved(req);
        Clk_t minIssueCycle = depSolved? m_clk + req.delay : -1UL;
        assert(!pendReqInfo.count(req.id));
        pendReqInfo.emplace(req.id, MemReqInfo{req, depSolved, minIssueCycle});
        reqCallback.emplace(req.id, callback);
        return true;
    }

    void tick() override {
        m_clk += 1;
        if (is_finished()) return;
        // send pend requests
        for (auto it = pendReqInfo.begin() ; it != pendReqInfo.end();) {
            auto reqId = it->first;
            auto& info = it->second;
            if (!info.depSolved || m_clk < info.minIssueCycle) {
                // m_logger->info("id {} addr {} depSolved {} minIssueCycle {} curClk {}",
                //         info.req.id, info.req.addr, info.depSolved, info.minIssueCycle, m_clk);
                ++it;
                continue;
            }
            if (sendRequest(info.req)) {
                bool isWrite = info.req.type == RAMSIM_WRITEREQ;
                it = pendReqInfo.erase(it);
                if (isWrite)
                    updateDepSolved();
            } else {
                ++it;
            }
        }
        // send pend responses
        for (auto it = pendResp.begin() ; it < pendResp.end(); it++) {
            // print("check reqCallback id %ld count %lx", it->id, reqCallback.count(it->id));
            assert(reqCallback.count(it->id));
            if (reqCallback[it->id](*it)) {
                // print("erase reqCallback id %ld", it->id);
                reqCallback.erase(it->id);
                pendResp.erase(it);
            }
        }
    };

    bool is_finished() override {
        return pendReqInfo.empty() && inflightReq.empty() && pendResp.empty();
    };

    uint64_t get_cycles() {
        return m_clk;
    }

    uint64_t get_total_mem_req() {
        return s_total_mem_req;
    }
};

} // namespace Ramulator

#endif   // RAMSIM_H
