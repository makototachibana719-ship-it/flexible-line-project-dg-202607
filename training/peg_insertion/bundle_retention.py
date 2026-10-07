"""Flat-file bundles: metadata-last commit, numeric retention, fail-closed journals.

Three filenames cannot be renamed atomically. A journal detects interrupted
transactions; readers must reject pending transactions, never accept partials.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import tempfile

PATTERN=re.compile(r'checkpoint_(0|[1-9][0-9]*)(?:_online_replay\.pkl|_resume_metadata\.json)?')
JOURNAL='.bundle_transaction.json'

def paths(directory,step):
    return tuple(Path(directory)/name for name in
        (f'checkpoint_{step}',f'checkpoint_{step}_online_replay.pkl',f'checkpoint_{step}_resume_metadata.json'))

def inventory(directory):
    directory=Path(directory)
    if (directory/JOURNAL).exists():raise RuntimeError('pending bundle transaction; manual recovery required')
    groups={}
    for path in directory.iterdir():
        if not path.name.startswith('checkpoint_'):continue
        match=PATTERN.fullmatch(path.name)
        if not match or path.is_symlink() or not path.is_file():raise ValueError('unrecognized checkpoint member: '+path.name)
        groups.setdefault(int(match.group(1)),set()).add(path.name)
    for step,names in groups.items():
        if names!={p.name for p in paths(directory,step)}:raise ValueError('incomplete preexisting bundle: '+str(step))
    return sorted(groups)

@contextmanager
def writer_lock(directory):
    import fcntl
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.bundle_writer.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)

def move_transaction(directory,moves):
    """Metadata is last on publish, first on retire. Exceptions roll back."""
    directory=Path(directory);journal=directory/JOURNAL
    with journal.open('x') as stream:
        json.dump([[str(src),str(dst)] for src,dst in moves],stream)
        stream.flush();os.fsync(stream.fileno())
    moved=[]
    try:
        for src,dst in moves:
            if dst.exists():raise FileExistsError(dst)
            os.rename(src,dst);moved.append((src,dst))
    except BaseException:
        for src,dst in reversed(moved):os.rename(dst,src)
        journal.unlink()
        raise
    journal.unlink()

def prune_locked(directory,keep_bundles=20):
    if type(keep_bundles) is not int or keep_bundles<1:raise ValueError('positive keep_bundles required')
    directory=Path(directory);steps=inventory(directory)
    retired=[]
    for step in steps[:-keep_bundles]:
        # Dedicated transaction area; no broad checkpoint-prefix deletion.
        quarantine=Path(tempfile.mkdtemp(prefix='.bundle_retired_',dir=directory))
        members=paths(directory,step)
        move_transaction(directory,[(src,quarantine/src.name) for src in reversed(members)])
        # Logical bundle is now fully retired. Delete only its three exact files.
        for src in members:(quarantine/src.name).unlink()
        quarantine.rmdir();retired.append(step)
    return retired

def retain_bundles(directory,keep_bundles=20):
    with writer_lock(directory):return prune_locked(directory,keep_bundles)

def save_transaction(directory,step,save,keep_bundles=20):
    if type(step) is not int or step<0:raise ValueError('nonnegative global checkpoint step required')
    if type(keep_bundles) is not int or keep_bundles<1:raise ValueError('positive keep_bundles required')
    directory=Path(directory).resolve()
    with writer_lock(directory):
        steps=inventory(directory)
        if steps and step<=steps[-1]:raise ValueError('checkpoint step must advance globally')
        stage=Path(tempfile.mkdtemp(prefix='.bundle_stage_',dir=directory))
        # On serialization failure preserve the hidden stage for diagnosis;
        # nothing partial is published to the checkpoint namespace.
        save(stage)
        if inventory(stage)!=[step]:raise ValueError('staged bundle incomplete')
        move_transaction(directory,list(zip(paths(stage,step),paths(directory,step))))
        stage.rmdir()
        prune_locked(directory,keep_bundles)
        return paths(directory,step)[0]
