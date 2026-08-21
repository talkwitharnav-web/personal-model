"""Peak-RSS reporting with no new dependencies.

RAM is this machine's real bottleneck (15.52 GB) and Python dies SILENTLY when it runs tight -
no traceback, the process just vanishes (onboarding section 2). Every long corpus job should
therefore report what it actually peaked at, so "this streams safely" is a measurement rather
than a claim.

psutil is not installed and deliberately not being added: requirements.txt lists plain `torch`
with no index URL, so any casual pip install risks pulling CPU-only torch over the CUDA build.
Windows exposes the number through psapi via ctypes, which is stdlib.
"""
import ctypes
import ctypes.wintypes
import sys


class _MemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.wintypes.DWORD),
        ("PageFaultCount", ctypes.wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def memory_mb():
    """(current, peak) resident memory in MB. Returns (None, None) if unavailable."""
    values = memory_detail()
    if values is None:
        return None, None
    return values["working_set"], values["peak_working_set"]


def memory_detail():
    """Working set AND private committed bytes.

    Working set counts memory-mapped file pages the OS happens to be holding, which is NOT the
    program's own demand: the same code writing a memmap measured 2,572 / 3,476 / 3,775 MB peak
    on three consecutive runs purely from system pressure. Private (commit) bytes exclude
    file-backed pages, so they measure what the process actually requires and cannot be evicted.

    That distinction is the whole point of the combine rewrite: anonymous allocations from
    concatenate + fancy-index CANNOT be evicted and cause the silent deaths in onboarding
    section 2, while mapped file pages can always be written back and reclaimed.
    """
    if sys.platform != "win32":
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            return {"working_set": None, "peak_working_set": peak,
                    "private": None, "peak_private": None}
        except Exception:
            return None

    counters = _MemoryCounters()
    counters.cb = ctypes.sizeof(counters)

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    # GetCurrentProcess returns the pseudo-handle (HANDLE)-1. Without an explicit restype
    # ctypes assumes c_int, which truncates it on 64-bit and the call then fails silently.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    handle = kernel32.GetCurrentProcess()

    # Modern Windows exposes this from kernel32 as K32GetProcessMemoryInfo; psapi.dll forwards
    # to it. Try kernel32 first, fall back to psapi for older builds.
    function = None
    for library, name in ((kernel32, "K32GetProcessMemoryInfo"),
                          (ctypes.WinDLL("psapi", use_last_error=True),
                           "GetProcessMemoryInfo")):
        try:
            function = getattr(library, name)
            break
        except AttributeError:
            continue
    if function is None:
        return None

    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MemoryCounters), ctypes.wintypes.DWORD]
    function.restype = ctypes.wintypes.BOOL
    if not function(handle, ctypes.byref(counters), counters.cb):
        return None
    mb = 1024 ** 2
    return {
        "working_set": counters.WorkingSetSize / mb,
        "peak_working_set": counters.PeakWorkingSetSize / mb,
        "private": counters.PagefileUsage / mb,
        "peak_private": counters.PeakPagefileUsage / mb,
    }


def report(label=""):
    values = memory_detail()
    if values is None:
        print(f"  memory: unavailable {label}")
        return None
    if values["private"] is None:
        print(f"  memory: peak {values['peak_working_set']:,.0f} MB {label}")
        return values["peak_working_set"]
    print(f"  memory: working set {values['working_set']:,.0f} MB "
          f"(peak {values['peak_working_set']:,.0f}) | "
          f"private {values['private']:,.0f} MB "
          f"(peak {values['peak_private']:,.0f}) {label}")
    return values["peak_private"]


class _SystemMemory(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.wintypes.DWORD),
        ("dwMemoryLoad", ctypes.wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def system_free_mb():
    """Free physical RAM for the WHOLE MACHINE, in MB, or None.

    Needed by anything that runs alongside training: this machine has 15.52 GB and Python dies
    SILENTLY when it runs short (onboarding section 2). A monitoring tool that loads a model
    without checking this can kill the run it exists to watch, which is strictly worse than
    having no monitor.
    """
    if sys.platform != "win32":
        return None
    try:
        status = _SystemMemory()
        status.dwLength = ctypes.sizeof(status)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return status.ullAvailPhys / 1024 ** 2
    except Exception:
        return None


if __name__ == "__main__":
    report("(baseline interpreter)")
