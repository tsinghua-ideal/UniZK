#!/usr/bin/python3
from enum import IntEnum
import random
import sys
magic_word = b'BINFILE\0'

class OpRecordType(IntEnum):
    READ = 0
    WRITE = 1

class OpRecord:
    def __init__(self, id, type, delay, dependencies, addr, size):
        self.id = id
        self.type = type
        self.delay = delay
        self.dependencies = dependencies
        self.addr = addr
        self.size = size

def save_trace(op_list, filename):
    bufsize = 8
    with open(filename, "wb") as f:
        f.write(magic_word)
        for op in op_list:
            f.write(op.id.to_bytes(bufsize, "little"))
            f.write(op.addr.to_bytes(bufsize, "little"))
            f.write(int(op.type).to_bytes(bufsize, "little"))
            f.write(op.delay.to_bytes(bufsize, "little"))
            f.write(op.size.to_bytes(bufsize, "little"))
            f.write(len(op.dependencies).to_bytes(bufsize, "little"))
            for dep in op.dependencies:
                f.write(dep.to_bytes(bufsize, "little"))

if len(sys.argv) != 3:
    print("Usage", sys.argv[0], "dep_count output.bin")
    sys.exit(-1)

dep_count = int(sys.argv[1])
nRecord = 100
recs = []
for i in range(nRecord):
    deps = []
    if i >= dep_count:
        for dep in range(1, dep_count):
            deps.append(i // dep_count * dep_count - dep)
    recs.append(OpRecord(i, OpRecordType.READ, 0, deps, 0x1000 + i * 64, 64))
# op1 = OpRecord(0, 2, 10, [], 0x1000, 4, 0)
# op2 = OpRecord(1, 2, 10, [], 0x1004, 4, 0)
# op3 = OpRecord(2, 0, 1, [1, 2], 0x1008, 4, 0)
# op4 = OpRecord(3, 3, 10, [3], 0x100c, 4, 0)

save_trace(recs, sys.argv[2])