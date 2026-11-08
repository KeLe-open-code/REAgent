import hashlib
from pathlib import Path


def evaluation_code_fingerprint():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob('*.py')):
        digest.update(path.name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()
