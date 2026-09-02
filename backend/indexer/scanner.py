import os
import fnmatch
from typing import List
from backend.indexer.models import ScannedFile

IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "dist",
    "build",
    "node_modules",
}

SECRET_PATTERNS = [
    ".env*",
    "*.pem",
    "*.key",
    "secrets.json",
]


def is_secret_file(filename: str) -> bool:
    """
    Check if a filename matches any of the defined secret patterns.
    """
    name_lower = filename.lower()
    for pattern in SECRET_PATTERNS:
        # Match pattern against lowercased filename for case-insensitive matching
        if fnmatch.fnmatch(name_lower, pattern.lower()):
            return True
    return False


def scan_repository(repo_path: str, max_size_kb: int = 500) -> List[ScannedFile]:
    """
    Recursively scan a repository path, ignoring specified directories and
    secret patterns, and filtering out files larger than max_size_kb.
    """
    scanned_files = []
    max_size_bytes = max_size_kb * 1024
    
    # Resolve absolute path of repo_path
    repo_path = os.path.abspath(repo_path)
    
    for root, dirs, files in os.walk(repo_path):
        # Prune ignored directories in-place to avoid descending into them
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if is_secret_file(file):
                continue
                
            abs_path = os.path.abspath(os.path.join(root, file))
            
            try:
                size_bytes = os.path.getsize(abs_path)
            except OSError:
                # If we cannot access the file or get its size, skip it
                continue
                
            if size_bytes > max_size_bytes:
                continue
                
            # Compute relative path and normalize separator to forward slash
            rel_path = os.path.relpath(abs_path, repo_path)
            rel_path_normalized = rel_path.replace("\\", "/")
            abs_path_normalized = abs_path.replace("\\", "/")
            
            _, ext = os.path.splitext(file)
            
            scanned_files.append(
                ScannedFile(
                    relative_path=rel_path_normalized,
                    absolute_path=abs_path_normalized,
                    extension=ext,
                    size_bytes=size_bytes
                )
            )
            
    return scanned_files
