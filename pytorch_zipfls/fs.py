import os
import stat
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


@contextmanager
def change_dir(dirpath):
    dirpath = str(dirpath)
    cwd = os.getcwd()
    try:
        os.chdir(dirpath)
        yield
    finally:
        os.chdir(cwd)


def make_symlink_if_not_exists(src: Path, dst: Path, overwrite=False):
    ''':param override: if destination exists and it is already a symlink
    then overwrite it.'''

    src, dst = Path(src), Path(dst)

    if not overwrite:
        dst.symlink_to(src)
        return

    while True:
        try:
            s = dst.lstat()
            break
        except FileNotFoundError:
            try:
                dst.symlink_to(src)
                return
            except FileExistsError:
                continue

    if not stat.S_ISLNK(s.st_mode):
        raise FileExistsError("{} exists and is not a symlink".format(dst))

    with TemporaryDirectory(dir=str(dst.parent)) as tmpdir:
        tmplink = Path(tmpdir, 'x')
        tmplink.symlink_to(src)
        tmplink.rename(dst)


def mkdir_p(dirname):
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != 17:
            raise e

