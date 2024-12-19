import threading

# Goal is to eventually remove these global vars
# For now they're here
lock = threading.Lock()
cond = threading.Condition(lock)
maxcount = None
waitset = set()
signalset = set()
TLCvars = {}
IO_inputs = []
IO_outputs = []
IO_running = set()


def checkcontinue(_step: int) -> bool:
    return not checkstop(_step)


def checkstop(_step: int) -> bool:
    return maxcount is not None and _step >= maxcount
