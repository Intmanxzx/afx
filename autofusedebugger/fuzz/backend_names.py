"""Display-only backend names; no execution dependencies."""
def display_backend(name):
    return {'npu-eager':'eager-npu', 'cpu-eager': 'cpu', 'triton': 'triton-npu',
            'ascendc': 'AF · 模式未记录',
            'cpu-compile': 'CPU · 编译诊断'}.get(name, name)


