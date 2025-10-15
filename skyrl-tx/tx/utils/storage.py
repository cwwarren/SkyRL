from contextlib import contextmanager
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
from tempfile import TemporaryDirectory
from typing import Generator
from cloudpathlib import AnyPath, CloudPath


def create_tar_archive(checkpoint_dir: Path) -> tuple[io.BytesIO, int]:
    """Create a tar.gz archive from a directory.

    Args:
        checkpoint_dir: Directory to archive

    Returns:
        Tuple of (BytesIO buffer containing tar.gz data, size of archive)
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for p in checkpoint_dir.iterdir():
            if p.is_file():
                tar.add(p, arcname=p.name)
    tar_size = tar_buffer.tell()
    tar_buffer.seek(0)
    return tar_buffer, tar_size


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
        tar_buffer, _ = create_tar_archive(tmp_path)

        # Upload/copy the tar file
        if isinstance(dest, CloudPath):
            # For CloudPath, write to a temp file then upload
            tar_file = tmp_path / "archive.tar.gz"
            with open(tar_file, "wb") as f:
                f.write(tar_buffer.read())
            dest.upload_from(tar_file)
        else:
            # For local path, write the tar file directly
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                f.write(tar_buffer.read())


@contextmanager
def staged_download(source: Path | CloudPath) -> Generator[Path, None, None]:
    """Temp directory that downloads and extracts tar.gz archive on entry.

    Args:
        source: Source path for the tar.gz file
    """
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Download the tar file to a separate temporary file if needed
        if isinstance(source, CloudPath):
            with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tar_file:
                tar_file_path = Path(tar_file.name)
                source.download_to(tar_file_path)

                # Extract tar archive
                with tarfile.open(tar_file_path, "r:gz") as tar:
                    tar.extractall(tmp_path, filter="data")
        else:
            tar_file_path = Path(source)

            # Extract tar archive
            with tarfile.open(tar_file_path, "r:gz") as tar:
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
