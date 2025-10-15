from contextlib import contextmanager
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Generator
from cloudpathlib import AnyPath, CloudPath


@contextmanager
def staged_upload(dest: Path | CloudPath) -> Generator[Path, None, None]:
    """Temp directory that uploads to cloud/local path on exit."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        yield tmp_path

        # Handle CloudPath and local Path differently
        if isinstance(dest, CloudPath):
            dest.upload_from(tmp_path)
        else:
            # For local paths, use shutil.copytree
            dest = Path(dest)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(tmp_path, dest)


@contextmanager
def staged_download(source: Path | CloudPath) -> Generator[Path, None, None]:
    """Temp directory that downloads from cloud/local path on entry."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Handle CloudPath and local Path differently
        if isinstance(source, CloudPath):
            source.download_to(tmp_path)
        else:
            # For local paths, use shutil.copytree
            source = Path(source)
            shutil.copytree(source, tmp_path, dirs_exist_ok=True)

        yield tmp_path

