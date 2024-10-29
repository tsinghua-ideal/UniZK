#include "base/base.h"
#include "base/request.h"
#include "base/config.h"
#include "frontend/frontend.h"
#include "memory_system/impl/generic_DRAM_system.cpp"
#include "frontend/impl/ramsim/ramsim.h"

class MemCpyAccel
{
public:
    Ramulator::RamSim *ramsim;
    Ramulator::GenericDRAMSystem *mem_system;
    std::vector<std::pair<Ramulator::MemReq, Ramulator::RamsimCallback>> pendReq;
    Ramulator::RamsimCallback rdHandler, wrHandler;

    uint64_t id;
    const uint32_t accSize = 64;
    Ramulator::Clk_t m_clk;

    MemCpyAccel(std::string config_path)
    {
        YAML::Node config = Ramulator::Config::parse_config_file(config_path, {});
        ramsim = dynamic_cast<Ramulator::RamSim *>(Ramulator::Factory::create_frontend(config));
        mem_system = dynamic_cast<Ramulator::GenericDRAMSystem *>(Ramulator::Factory::create_memory_system(config));
        rdHandler = [this](Ramulator::MemReq &req)
        { return this->handleRead(req); };
        wrHandler = [this](Ramulator::MemReq &req)
        { return this->handleWrite(req); };
        ramsim->connect_memory_system(mem_system);
        mem_system->connect_frontend(ramsim);
        ramsim->setup(ramsim, rdHandler, mem_system);
        id = 0;
        m_clk = 0;
    };

    ~MemCpyAccel()
    {
        ramsim->IFrontEnd::finalize();
        mem_system->IMemorySystem::finalize();
        delete ramsim;
        delete mem_system;
    };

    bool handleRead(Ramulator::MemReq &req)
    {
        // std::cout << "Finish memcpy read addr " << req.addr << " size " << req.size << std::endl;
        return true;
    }

    bool handleWrite(Ramulator::MemReq &req)
    {
        // std::cout << "Finish memcpy write addr " << req.addr << " size " << req.size << std::endl;
        return true;
    }

    void start(Ramulator::Addr_t src, Ramulator::Addr_t dst, uint32_t size)
    {
        for (Ramulator::Addr_t offset = 0; offset < size; offset += accSize)
        {
            Ramulator::MemReq rd{id, src + offset, RAMSIM_READREQ, 0, accSize, std::vector<uint64_t>(),0, std::vector<uint64_t>(1)};
            id += 1;
            bool rd_success = ramsim->send(rd, rdHandler);
            if (rd_success == false)
            {
                pendReq.push_back(std::make_pair(rd, rdHandler));
            }
            // std::cout << "Start memcpy addr " << rd.addr << " size " << rd.size << std::endl;
            Ramulator::MemReq wr{id, dst + offset, RAMSIM_WRITEREQ, 0, accSize, std::vector<uint64_t>(1, id-1),0, std::vector<uint64_t>(1)};
            id += 1;
            bool wr_success = ramsim->send(wr, wrHandler);
            if (wr_success == false)
            {
                pendReq.push_back(std::make_pair(wr, wrHandler));
            }
        }
    }

    void tick()
    {
        m_clk += 1;
        ramsim->tick();
        mem_system->tick();
        for (auto it = pendReq.begin(); it < pendReq.end(); it++)
        {
            if (ramsim->send(it->first, it->second))
            {
                pendReq.erase(it);
            }
        }
    }

    bool is_finished()
    {
        // printf("ramsim->is_finished() %d\n", ramsim->is_finished());
        // printf("pendReq.empty() %d\n", pendReq.empty());
        return ramsim->is_finished() && pendReq.empty();
    }

    void showStats()
    {
        std::cout << "-------" << std::endl;
        std::cout << "Total cycles: " << mem_system->get_clock() << std::endl;
        std::cout << "Total read num: " << mem_system->get_s_num_read_requests() << std::endl;
        std::cout << "Total write num: " << mem_system->get_s_num_write_requests() << std::endl;
        std::cout << "Total ramsim serve num: " << ramsim->get_total_mem_req() << std::endl;
        std::cout << "-------" << std::endl;
    }
};

int main(int argc, char **argv)
{
    int size = 0;
    scanf("%d", &size);
    MemCpyAccel accel("accel.yaml");
    accel.start(0x10000000, 0x20000000, size);
    while (!accel.is_finished())
    {
        accel.tick();
        // printf("Tick %ld\n", accel.m_clk);
    }
    accel.showStats();
    return 0;
}