"""One atomic JSON store for suite results and restart checkpoints."""
import json
from pathlib import Path
import shutil

ACTIVE = None


class ResultStore:
    def __init__(self, path):
        self.path = Path(path)
        self.backup_path = self.path.with_name(
            f'{self.path.stem}.backup{self.path.suffix}')
        self.data = None
        for source in (self.path, self.backup_path):
            if source.exists():
                try:
                    self.data = json.loads(source.read_text(encoding='utf-8'))
                    break
                except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
        if self.data is None:
            self.data = {
                'schema_version': 1, 'seeds': [44, 43, 42], 'jobs': {},
                'experiments': {}, 'sessions': []}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False), encoding='utf-8')
        if self.path.exists():
            backup_temporary = self.backup_path.with_suffix('.tmp')
            shutil.copyfile(self.path, backup_temporary)
            backup_temporary.replace(self.backup_path)
        temporary.replace(self.path)


def get(key):
    return ACTIVE.data['experiments'].get(str(key)) if ACTIVE is not None else None


def put(key, result):
    if ACTIVE is not None:
        ACTIVE.data['experiments'][str(key)] = result
        ACTIVE.save()


def enabled():
    return ACTIVE is not None
