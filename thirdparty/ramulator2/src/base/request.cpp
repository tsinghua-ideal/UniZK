#include "base/request.h"

namespace Ramulator {

Request::Request(Addr_t addr, int type): addr(addr), type_id(type) {};

Request::Request(AddrVec_t addr_vec, int type): addr_vec(addr_vec), type_id(type) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback):
addr(addr), type_id(type), source_id(source_id), callback(callback) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, uint64_t id):
addr(addr), type_id(type), source_id(source_id), callback(callback), id(id) {};

Request::Request(Addr_t addr, int type, int source_id, std::function<void(Request&)> callback, uint64_t id, bool last_subid):
addr(addr), type_id(type), source_id(source_id), callback(callback), id(id), last_subid(last_subid) {};
}        // namespace Ramulator

