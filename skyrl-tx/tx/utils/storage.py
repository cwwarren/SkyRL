from contextlib import contextmanager
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
from typing import Generator
from cloudpathlib import AnyPath, CloudPath


def create_tar_archive(checkpoint_dir: Path) -> io.BytesIO:
    """Create a tar.gz archive from a directory.

    Args:
        checkpoint_dir: Directory to archive

    Returns:
        BytesIO buffer containing tar.gz data
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for p in checkpoint_dir.iterdir():
            if p.is_file():
                tar.add(p, arcname=p.name)
    tar_buffer.seek(0)
    return tar_buffer


@contextmanager
def staged_upload(dest: Path | CloudPath) -> Generator[Path, None, None]:
    """Temp directory that creates a tar.gz archive and uploads on exit.

    Args:
        dest: Destination path for the tar.gz file
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        yield tmp_path

        # Create tar archive of temp directory contents
        tar_buffer = create_tar_archive(tmp_path)

        # Write the tar file (handles both local and cloud storage)
        dest = AnyPath(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            f.write(tar_buffer.read())


@contextmanager
def staged_download(source: Path | CloudPath) -> Generator[Path, None, None]:
    """Temp directory that downloads and extracts tar.gz archive on entry.

    Args:
        source: Source path for the tar.gz file
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download and extract tar archive (handles both local and cloud storage)
        source = AnyPath(source)
        with source.open("rb") as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                tar.extractall(tmp_path, filter="data")

        yield tmp_path


def download_file(source: AnyPath) -> io.BytesIO:
    """Download a file from storage and return it as a BytesIO object.

    Args:
        source: Source path for the file (local or cloud)

    Returns:
        BytesIO object containing the file contents
    """
    buffer = io.BytesIO()
    with source.open("rb") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer
