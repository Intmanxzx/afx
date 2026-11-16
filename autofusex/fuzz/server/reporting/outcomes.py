"""Independent accuracy and measurement states, never a performance pass claim."""
def outcomes(attempt, evaluation):
    shapes = attempt['shapes']
    accuracy = [s['correctness'] for s in shapes]
    performance = [s['performance'] for s in shapes]
    if not evaluation.get('correctness', True):
        correctness = 'not_requested'
    elif 'fail' in accuracy:
        correctness = 'fail'
    elif 'error' in accuracy:
        correctness = 'error'
    elif accuracy and all(s == 'pass' for s in accuracy):
        correctness = 'pass'
    elif attempt['status'] == 'unsupported':
        correctness = 'unsupported'
    elif attempt['status'] == 'error':
        correctness = 'error'
    else:
        correctness = 'not_checked'
    if 'error' in performance:
        measurement = 'error'
    elif performance and all(s == 'measured' for s in performance):
        measurement = 'measured'
    elif 'measured' in performance:
        measurement = 'partial'
    elif evaluation.get('performance'):
        measurement = 'not_completed'
    else:
        measurement = 'not_requested'
    return {'correctness_status': correctness, 'measurement_status': measurement}
