"""
Shared file-path substring patterns recognized by both the organization
policy engine (backend/policy/evaluator.py, which allows/blocks changes to
these paths) and the git diff risk heuristic (backend/vcs/git_manager.py,
which scores changes to these paths as elevated risk). Extracted here so a
new dependency manager, CI provider, or config format only needs to be
added in one place instead of drifting between two independently
maintained lists.
"""

# Dependency manifest / package-lock filenames across common ecosystems.
DEPENDENCY_MANIFEST_PATTERNS = [
    "requirements.txt",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "Gemfile",
    "pom.xml",
    "build.gradle",
]

# CI/CD pipeline configuration paths.
CI_CD_PATTERNS = [
    ".github/",
    ".gitlab-ci",
    "Jenkinsfile",
    ".circleci/",
    ".travis.yml",
    "azure-pipelines.yml",
]

# General infrastructure/environment configuration paths.
INFRA_CONFIG_PATTERNS = [
    "docker-compose",
    "Dockerfile",
    "Makefile",
    ".env",
    ".env.",
    "alembic.ini",
]
