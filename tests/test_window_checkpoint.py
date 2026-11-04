#!/usr/bin/env python3
"""The short-term file on disk must never name a memory that long-term memory
already holds, and a window left over its maximum must drain on open without
promoting anything twice.

Runs against a throwaway store in a temp directory. No fixtures, no framework:
    python tests/test_window_checkpoint.py
"""
import contextlib, io, json, os, shutil, sys, tempfile

LIB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, LIB); sys.path.insert(0, os.path.join(LIB, "V2"))

store = tempfile.mkdtemp(prefix="asmc-test-")
os.environ["ASMC_STORE_DIR"] = store
from ASMC_API import create_memory


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def ltm_keys(mem):
    # Read through the library's own handle: LMDB refuses a second open of the
    # same environment inside one process.
    with mem._mm._ltm._env.begin() as txn:
        return {(json.loads(v)["inputText"], json.loads(v).get("timeDate")) for k, v in txn.cursor() if k.isdigit()}


def stm_file(path):
    f = os.path.join(path, "MemoryStructures", "STM", "stm.json")
    if not os.path.exists(f):      # nothing has been checkpointed yet
        return []
    return json.load(open(f))["entries"]


try:
    WINDOW = 10
    mem = quiet(create_memory, max_entries=WINDOW, db_path=store)   # db_path and max_entries now take effect
    assert mem._mm._stm.STM_MAX == WINDOW, "max_entries ignored"

    # 1. After every add, the file on disk and long-term memory are disjoint.
    for i in range(WINDOW + 7):
        quiet(mem.add_experience, situation=f"prompt number {i} about topic {i % 3}", response=f"answer {i}")
        on_disk = {(e["inputText"], e.get("timeDate")) for e in stm_file(store)}
        assert not (on_disk & ltm_keys(mem)), f"after add {i}: the window file names a promoted memory"
    assert len(stm_file(store)) <= WINDOW
    before = ltm_keys(mem)
    assert len(before) == 7, f"expected 7 promotions, found {len(before)}"
    with mem._mm._ltm._env.begin() as txn:
        already = json.loads(txn.get(b"0"))                 # a record long-term memory holds
    quiet(mem._mm._stm._checkpoint)   # a clean exit flushes the window; do it explicitly here
    mem._mm._ltm._env.close()         # so the second open below is the only one in this process
    del mem

    # 2. A stale file (what an unclean death left behind): over the window, and holding
    #    one entry long-term memory already has. Open -> drained, nothing promoted twice.
    entries = stm_file(store)
    fresh = dict(entries[-1]); fresh["inputText"] = "a genuinely unpromoted entry"; fresh["timeDate"] = "2000-01-01T00:00:00+00:00"
    stale = [already, fresh] + entries                      # WINDOW + 2 entries
    json.dump({"entries": stale}, open(os.path.join(store, "MemoryStructures", "STM", "stm.json"), "w"))
    mem = quiet(create_memory, max_entries=WINDOW, db_path=store)
    after = ltm_keys(mem)
    assert mem._mm._stm.count() == WINDOW, f"window not drained: {mem._mm._stm.count()}"
    assert len(stm_file(store)) == WINDOW, "drained window not written to disk"
    assert after - before == {("a genuinely unpromoted entry", "2000-01-01T00:00:00+00:00")}, "wrong promotions on drain"
    assert len(after) == len(before) + 1, "a memory was promoted twice"
    print("PASS: checkpoint after promotion, drain on open, db_path and max_entries honoured")
finally:
    shutil.rmtree(store, ignore_errors=True)
