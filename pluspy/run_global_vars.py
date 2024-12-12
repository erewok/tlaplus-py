import threading

# Goal is to eventually remove these global vars
# For now they're here
lock = threading.Lock()
cond = threading.Condition(lock)
maxcount = None
step = 0
waitset = set()
signalset = set()
TLCvars = {}
IO_inputs = []
IO_outputs = []
IO_running = set()
