"""Independent sync daemon for hosts/containers without an active systemd bus."""
import signal
import sys
from .client import run

if __name__=='__main__':
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    run(sys.argv[1],sys.argv[2],_fd=int(sys.argv[3]))
