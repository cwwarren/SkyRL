from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator
from cloudpathlib import AnyPath, CloudPath


@contextmanager
def staged_upload(dest: Path | CloudPath) -> Generator[Path, None, None]:
    """Temp directory that uploads to cloud/local path on exit."""
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        yield tmp_path
        
        AnyPath(dest).upload_from(tmp_path)

