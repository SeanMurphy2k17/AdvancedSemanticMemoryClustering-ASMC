"""A store with MemoryStructures/PROTECTED refuses to be cleared; one without still can be.

Run: python tests/test_protected_store.py   (uses temporary stores only)
"""
import os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def _store_with(n_entries, protected):
    tmp = tempfile.mkdtemp(prefix="asmc-protect-")
    os.environ["ASMC_STORE_DIR"] = tmp
    for mod in [m for m in list(sys.modules) if m.startswith("V2") or m == "ASMC_API"]:
        del sys.modules[mod]  # the modules read ASMC_STORE_DIR when built
    from ASMC_API import create_memory
    api = create_memory(max_entries=50, db_path=tmp, verbose=False)
    for i in range(n_entries):
        api.add_experience(f"situation {i} about tide pools", f"answer {i} about bioluminescence")
    if protected:
        marker = api._mm._ltm.protected_marker()
        with open(marker, "w") as f:
            f.write("test\n")
    return tmp, api


def main():
    tmp, api = _store_with(3, protected=True)
    before = api.get_statistics()
    r = api.clear_memory(confirm=True)
    assert r.get("refused") and not r.get("success"), r
    after = api.get_statistics()
    assert before == after, (before, after)
    raised = False
    try:
        api._mm._ltm._clear()
    except PermissionError:
        raised = True
    assert raised, "_clear must raise on a protected store"
    shutil.rmtree(tmp)

    tmp, api = _store_with(3, protected=False)
    r = api.clear_memory(confirm=True)
    assert r.get("success"), r
    assert api.get_statistics()["stm_entries"] == 0
    shutil.rmtree(tmp)
    print("PASS: protected store refuses clear_memory and _clear; unprotected store still clears")


if __name__ == "__main__":
    main()
