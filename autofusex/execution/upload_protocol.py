"""Execution upload wire versions are independent of capture/classifier versions."""
CURRENT = 2
SUPPORTED = (1, 2)
CAPTURE_CURRENT = 2

def version(value, supported=SUPPORTED):
    if type(value) is not int or value not in supported:
        raise ValueError('unsupported protocol version; supported: ' + ','.join(map(str, supported)))
    return value
