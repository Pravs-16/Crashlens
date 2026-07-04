"""Central configuration for CrashLens."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "smd"
ARTIFACT_DIR = ROOT / "artifacts"

MACHINE = "machine-1-1"

# The 38 telemetry metrics recorded for every SMD machine, in column order.
# Names follow the SMD documentation (CPU, load averages, memory, disk,
# swap-in/out, network interface, TCP/UDP counters).
METRIC_NAMES = [
    "cpu_r", "load_1", "load_5", "load_15",
    "mem_shmem", "mem_u", "mem_u_e", "total_mem",
    "disk_q", "disk_r", "disk_rb", "disk_svc", "disk_u", "disk_w",
    "disk_wa", "disk_wb",
    "si", "so",
    "eth1_fi", "eth1_fo", "eth1_pi", "eth1_po",
    "tcp_tw", "tcp_use", "active_opens", "curr_estab",
    "in_errs", "in_segs", "listen_overflows", "out_rsts", "out_segs",
    "passive_opens", "retransegs", "tcp_timeouts",
    "udp_in_dg", "udp_out_dg", "udp_rcv_buf_errs", "udp_snd_buf_errs",
]

N_FEATURES = len(METRIC_NAMES)

# training
VAL_FRACTION = 0.1
BATCH_SIZE = 256
MAX_EPOCHS = 60
PATIENCE = 5
LR = 1e-3
SEED = 42

# detection
SMOOTH_WINDOW = 5      # rolling-median smoothing of the anomaly score
MIN_SEGMENT_GAP = 2    # merge predicted segments separated by <= this many steps
TOP_K_METRICS = 5      # metrics listed per anomaly in reports
