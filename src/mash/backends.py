from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from collections.abc import Callable, Sequence
from functools import cache
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mash.config import data_dir, matlab_root

if TYPE_CHECKING:
    from mash.session import Output

Log = Callable[[str], None]

RIG_VERSION = "0.11.0"
R_REPOS = "https://cloud.r-project.org"


def run(cmd: Sequence[str | Path], log: Log, env: dict[str, str] | None = None) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        env=env,
        text=True,
        errors="replace",
    )
    assert proc.stdout
    for line in proc.stdout:
        log(line.rstrip())
    if proc.wait():
        raise RuntimeError(f"{Path(str(cmd[0])).name} exited with status {proc.returncode}")


def download(url: str, log: Log) -> Path:
    log(f"downloading {url}")
    target = Path(tempfile.mkdtemp(prefix="mash-")) / url.rsplit("/", 1)[-1]
    urllib.request.urlretrieve(url, target)
    return target


@cache
def _version(exe: str, extra_env: tuple[tuple[str, str], ...] = ()) -> str:
    env = {**os.environ, **dict(extra_env)}
    out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60, env=env).stdout
    m = re.search(r"version (\d+\.\d+(\.\d+)?)", out)
    if not m:
        raise RuntimeError(f"could not read the version of {exe}")
    return m.group(1)


def minor(version: str) -> str:
    return ".".join(version.split(".")[:2])


def r_string(path: Path | str) -> str:
    return "'" + str(path).replace("\\", "\\\\").replace("'", "\\'") + "'"


class Backend:
    name = ""
    title = ""
    alias = ""
    file_ext = ""
    marker = "# %%"
    lexer = ""
    start_timeout = 120.0
    interrupt_mode = "signal"
    installable = True
    missing = "not installed"
    variables_code = ""
    _install_lock = threading.Lock()

    def status(self, cwd: Path) -> tuple[bool, str]:
        raise NotImplementedError

    def launch(self, cwd: Path, share_name: str) -> tuple[list[str], dict[str, str]]:
        raise NotImplementedError

    def install_summary(self) -> str:
        return ""

    def setup_action(self) -> str:
        return f"install {self.title}"

    def install(self, log: Log) -> None:
        with self._install_lock:
            self._install(log)

    def _install(self, log: Log) -> None:
        raise NotImplementedError

    def startup_code(self) -> str | None:
        return None

    def snapshot_code(self, base: Path) -> str | None:
        return None

    def failed(self, outputs: list[Output]) -> bool:
        return any(o.kind == "error" for o in outputs)

    def desktop_url(self, prev: str, line: str) -> str | None:
        return None

    def notebook_kernelspec(self, language_info: dict[str, Any]) -> dict[str, str]:
        raise NotImplementedError


class MatlabBackend(Backend):
    name = "matlab"
    title = "MATLAB"
    alias = "matlab"
    file_ext = ".m"
    marker = "%%"
    lexer = "matlab"
    start_timeout = 300.0
    interrupt_mode = "message"
    installable = False
    variables_code = "whos"

    def status(self, cwd: Path) -> tuple[bool, str]:
        try:
            return True, str(matlab_root())
        except RuntimeError as e:
            return False, str(e)

    def launch(self, cwd: Path, share_name: str) -> tuple[list[str], dict[str, str]]:
        root = matlab_root()
        env = dict(os.environ)
        # the kernel runs matlab-proxy-app from PATH; it sits next to mash's interpreter
        paths = [str(root / "bin"), str(Path(sys.executable).parent), env["PATH"]]
        env["PATH"] = os.pathsep.join(paths)
        env["MWI_CUSTOM_MATLAB_ROOT"] = str(root)
        env["MWI_USE_EXISTING_LICENSE"] = "True"
        env.setdefault("MWI_LOG_LEVEL", "WARN")
        quoted = str(cwd).replace("'", "''")
        env["MWI_MATLAB_STARTUP_SCRIPT"] = f"cd('{quoted}'); matlab.engine.shareEngine('{share_name}');"
        return [sys.executable, "-m", "jupyter_matlab_kernel", "-f", "{connection_file}"], env

    def install_summary(self) -> str:
        return "MATLAB needs a licensed MathWorks install; mash only installs the MATLAB Engine for Python"

    def _install(self, log: Log) -> None:
        root = matlab_root()
        engine = root / "extern" / "engines" / "python"
        if not engine.is_dir():
            raise RuntimeError(f"no Python engine under {root}; set MASH_MATLAB_ROOT to a MATLAB install")
        log(f"installing MATLAB Engine for Python from {engine}")
        run(["uv", "pip", "install", "--python", sys.executable, engine], log)

    def startup_code(self) -> str | None:
        return "%%matlab new_session"

    def snapshot_code(self, base: Path) -> str | None:
        quoted = str(base.with_suffix(".mat")).replace("'", "''")
        return f"save('{quoted}')"

    def failed(self, outputs: list[Output]) -> bool:
        # the MATLAB kernel reports errors and warnings as stderr
        return any(o.kind == "error" or (o.kind == "stream" and o.data["name"] == "stderr") for o in outputs)

    def desktop_url(self, prev: str, line: str) -> str | None:
        if "Access MATLAB at" in prev and line.strip().startswith("http"):
            return line.strip()
        return None

    def notebook_kernelspec(self, language_info: dict[str, Any]) -> dict[str, str]:
        return {"name": "jupyter_matlab_kernel", "display_name": "MATLAB Kernel", "language": "matlab"}


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def has_ipykernel(venv: Path) -> bool:
    return (
        bool(glob(str(venv / "lib" / "python*" / "site-packages" / "ipykernel")))
        or (venv / "Lib" / "site-packages" / "ipykernel").is_dir()
    )


def uv_exe() -> str | None:
    return os.environ.get("UV") or shutil.which("uv")


class PythonBackend(Backend):
    name = "python"
    title = "Python"
    alias = "py"
    file_ext = ".py"
    lexer = "python"
    variables_code = "%whos"
    missing = "no Python venv"
    packages = ("ipykernel", "numpy", "pandas", "matplotlib")

    def default_env(self) -> Path:
        return data_dir() / "python" / "venv"

    def _locate(self, cwd: Path) -> tuple[Path, Path | None, bool, str] | None:
        # (interpreter, venv, has ipykernel, where it came from)
        override = os.environ.get("MASH_PYTHON")
        if override:
            py = Path(override)
            venv = py.parents[1] if (py.parents[1] / "pyvenv.cfg").exists() else None
            return py, venv, True, "MASH_PYTHON"
        venv = cwd / ".venv"
        if venv_python(venv).exists():
            return venv_python(venv), venv, has_ipykernel(venv), "workspace .venv"
        venv = self.default_env()
        if venv_python(venv).exists():
            return venv_python(venv), venv, has_ipykernel(venv), str(venv)
        return None

    def status(self, cwd: Path) -> tuple[bool, str]:
        found = self._locate(cwd)
        if found is None:
            return False, f"no .venv in {cwd} and no mash venv at {self.default_env()}"
        _, _, kernel, where = found
        if kernel:
            return True, where
        if uv_exe() is None:
            return False, f"{where} has no ipykernel and uv is not on PATH"
        return True, f"{where} (ipykernel layered in by uv)"

    def launch(self, cwd: Path, share_name: str) -> tuple[list[str], dict[str, str]]:
        found = self._locate(cwd)
        if found is None:
            raise RuntimeError(self.status(cwd)[1])
        py, venv, kernel, _ = found
        env = dict(os.environ)
        env.pop("PYTHONHOME", None)
        env.pop("VIRTUAL_ENV", None)
        if kernel:
            if venv is not None:
                env["VIRTUAL_ENV"] = str(venv)
                env["PATH"] = f"{py.parent}{os.pathsep}{env['PATH']}"
            return [str(py), "-m", "ipykernel_launcher", "-f", "{connection_file}"], env
        uv = uv_exe()
        if uv is None:
            raise RuntimeError("uv is not on PATH")
        # --no-sync keeps uv from changing the project env; --with puts ipykernel in a throwaway overlay
        argv = [uv, "run", "--no-sync", "--with", "ipykernel", "python", "-m", "ipykernel_launcher"]
        return [*argv, "-f", "{connection_file}"], env

    def install_summary(self) -> str:
        return (
            f"uv creates a Python 3.13 venv at {self.default_env()} with {', '.join(self.packages)} "
            "(about 150 MB). Workspaces with their own .venv keep using it."
        )

    def setup_action(self) -> str:
        return "create a Python venv"

    def _install(self, log: Log) -> None:
        uv = uv_exe()
        if uv is None:
            raise RuntimeError("uv is not on PATH; install it from https://docs.astral.sh/uv/")
        venv = self.default_env()
        if not venv_python(venv).exists():
            run([uv, "venv", "--python", "3.13", venv], log)
        run([uv, "pip", "install", "--python", venv_python(venv), *self.packages], log)

    def notebook_kernelspec(self, language_info: dict[str, Any]) -> dict[str, str]:
        return {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"}


R_VARIABLES = """\
invisible(local({
  e <- globalenv(); n <- ls(e)
  size <- function(v) if (is.null(dim(v))) as.character(length(v)) else paste(dim(v), collapse = "x")
  if (length(n)) print(data.frame(name = n,
    class = vapply(n, function(x) class(get(x, envir = e))[1], ""),
    size = vapply(n, function(x) size(get(x, envir = e)), ""), row.names = NULL), row.names = FALSE)
  else cat("no variables\\n")
}))"""


class RBackend(Backend):
    name = "r"
    title = "R"
    alias = "r"
    file_ext = ".R"
    lexer = "r"
    variables_code = R_VARIABLES

    def root(self) -> Path:
        return data_dir() / "r"

    def rig_env(self) -> dict[str, str]:
        root = self.root()
        return {
            "RIG_MODE": "user",
            "RIG_R_INSTALL_DIR": str(root / "versions"),
            "RIG_BINARY_DIR": str(root / "bin"),
            "RIG_CACHE_DIR": str(root / "cache"),
        }

    def managed_r(self) -> Path | None:
        links = [Path(p) for p in glob(str(self.root() / "bin" / "R-[0-9]*"))]
        if not links:
            return None
        return max(links, key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])

    def find_r(self) -> tuple[Path, str] | None:
        override = os.environ.get("MASH_R")
        if override:
            return Path(override), "MASH_R"
        managed = self.managed_r()
        if managed:
            return managed, "installed by mash with rig"
        exe = shutil.which("R")
        return (Path(exe), "PATH") if exe else None

    def kernel_lib(self, rbin: Path) -> Path:
        return self.root() / "kernel-lib" / minor(_version(str(rbin)))

    def status(self, cwd: Path) -> tuple[bool, str]:
        found = self.find_r()
        if found is None:
            return False, "R not found; mash can install it with rig"
        rbin, where = found
        try:
            version = _version(str(rbin))
        except Exception as e:
            return False, f"{rbin}: {e}"
        if (self.kernel_lib(rbin) / "IRkernel").is_dir() or _r_has_irkernel(str(rbin)):
            return True, f"R {version} ({where})"
        return False, f"R {version} ({where}) has no IRkernel; mash can install it"

    def launch(self, cwd: Path, share_name: str) -> tuple[list[str], dict[str, str]]:
        found = self.find_r()
        if found is None:
            raise RuntimeError(self.status(cwd)[1])
        rbin = found[0]
        # append the kernel library last so project libraries (renv) still come first
        boot = f".libPaths(c(.libPaths(), {r_string(self.kernel_lib(rbin))})); IRkernel::main()"
        return [str(rbin), "--no-echo", "-e", boot, "--args", "{connection_file}"], dict(os.environ)

    def install_summary(self) -> str:
        if self.find_r():
            return f"installs IRkernel and its packages into {self.root() / 'kernel-lib'} (a few MB)"
        return (
            f"rig {RIG_VERSION} installs the current R release in user mode (no sudo) under {self.root()}, "
            "then IRkernel (about 150 MB)"
        )

    def _install(self, log: Log) -> None:
        found = self.find_r()
        if found is None:
            self._install_r(log)
            found = self.find_r()
            if found is None:
                raise RuntimeError(f"rig finished but no R-x.y link appeared in {self.root() / 'bin'}")
        rbin = found[0]
        _version.cache_clear()
        lib = self.kernel_lib(rbin)
        lib.mkdir(parents=True, exist_ok=True)
        code = (
            f".libPaths({r_string(lib)}); r <- getOption('repos'); "
            "if (is.null(r) || identical(unname(r['CRAN']), '@CRAN@')) "
            f"options(repos = c(CRAN = {r_string(R_REPOS)})); "
            f"install.packages('IRkernel', lib = {r_string(lib)}); "
            f"if (!requireNamespace('IRkernel', lib.loc = {r_string(lib)}, quietly = TRUE)) quit(status = 1)"
        )
        run([rbin, "--no-save", "--no-restore", "--no-init-file", "--no-echo", "-e", code], log)
        _r_has_irkernel.cache_clear()

    def _install_r(self, log: Log) -> None:
        plat = {"darwin": "macos", "linux": "linux"}.get(sys.platform)
        if plat is None:
            raise RuntimeError(f"mash can't install R on {sys.platform}; install R, then run mash-setup r")
        machine = platform.machine().lower()
        arch = "x86_64" if machine in ("x86_64", "amd64") else ("arm64" if plat == "macos" else "aarch64")
        # fetch the release tarball: rig's install.sh also writes a receipt into the user's own rig config
        name = f"rig-{plat}-{arch}-{RIG_VERSION}.tar.gz"
        archive = download(f"https://github.com/r-lib/rig/releases/download/v{RIG_VERSION}/{name}", log)
        with tarfile.open(archive) as tar:
            tar.extractall(self.root() / "rig", filter="data")
        env = {**os.environ, **self.rig_env(), "HOME": str(self.root() / "rig-home")}
        # rig fails if these are missing, and writes ~/.profile, its config and pak under HOME
        for key in ("RIG_R_INSTALL_DIR", "RIG_BINARY_DIR", "RIG_CACHE_DIR", "HOME"):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        run([self.root() / "rig" / "bin" / "rig", "add", "release", "--without-pak"], log, env)

    def notebook_kernelspec(self, language_info: dict[str, Any]) -> dict[str, str]:
        return {"name": "ir", "display_name": "R", "language": "R"}


@cache
def _r_has_irkernel(rbin: str) -> bool:
    proc = subprocess.run(
        [rbin, "--no-echo", "-e", "quit(status = !requireNamespace('IRkernel', quietly = TRUE))"],
        capture_output=True,
        timeout=60,
    )
    return proc.returncode == 0


class JuliaBackend(Backend):
    name = "julia"
    title = "Julia"
    alias = "jl"
    file_ext = ".jl"
    lexer = "julia"
    start_timeout = 180.0
    variables_code = "Main.InteractiveUtils.varinfo()"

    def root(self) -> Path:
        return data_dir() / "julia"

    def juliaup_env(self) -> dict[str, str]:
        return {"JULIAUP_DEPOT_PATH": str(self.root() / "juliaup-depot")}

    def find_julia(self) -> tuple[Path, str] | None:
        override = os.environ.get("MASH_JULIA")
        if override:
            return Path(override), "MASH_JULIA"
        managed = self.root() / "juliaup" / "bin" / "julia"
        if managed.exists():
            return managed, "installed by mash with juliaup"
        exe = shutil.which("julia")
        return (Path(exe), "PATH") if exe else None

    def launcher_env(self, julia: Path) -> dict[str, str]:
        return self.juliaup_env() if julia == self.root() / "juliaup" / "bin" / "julia" else {}

    def julia_env(self, julia: Path) -> dict[str, str]:
        return {**os.environ, **self.launcher_env(julia)}

    def kernel_env(self, version: str) -> Path:
        return self.root() / "kernel-env" / f"v{minor(version)}"

    def _version(self, julia: Path) -> str:
        return _version(str(julia), tuple(self.launcher_env(julia).items()))

    def status(self, cwd: Path) -> tuple[bool, str]:
        found = self.find_julia()
        if found is None:
            return False, "julia not found; mash can install it with juliaup"
        julia, where = found
        try:
            version = self._version(julia)
        except Exception as e:
            return False, f"{julia}: {e}"
        manifest = self.kernel_env(version) / "Manifest.toml"
        if manifest.exists() and "IJulia" in manifest.read_text():
            return True, f"Julia {version} ({where})"
        return False, f"Julia {version} ({where}) has no IJulia for mash; mash can install it"

    def launch(self, cwd: Path, share_name: str) -> tuple[list[str], dict[str, str]]:
        found = self.find_julia()
        if found is None:
            raise RuntimeError(self.status(cwd)[1])
        julia = found[0]
        env = self.julia_env(julia)
        depot = self.root() / "depot"
        # a leading empty entry keeps the user's default depots first; IJulia lives in mash's depot
        env["JULIA_DEPOT_PATH"] = f"{os.pathsep}{depot}"
        env["JULIA_LOAD_PATH"] = os.pathsep.join(
            ["@", "@v#.#", "@stdlib", str(self.kernel_env(self._version(julia)))]
        )
        boot = "import IJulia; IJulia.run_kernel()"
        return [str(julia), "-i", "--color=yes", "--project=@.", "-e", boot, "{connection_file}"], env

    def install_summary(self) -> str:
        what = f"IJulia into {self.root() / 'kernel-env'}"
        if self.find_julia():
            return f"installs and precompiles {what} (a few minutes)"
        return f"juliaup installs the current Julia release under {self.root()}, then {what} (about 400 MB)"

    def _install(self, log: Log) -> None:
        found = self.find_julia()
        if found is None:
            self._install_julia(log)
            found = self.find_julia()
            if found is None:
                raise RuntimeError("juliaup finished but julia was not found")
        julia = found[0]
        _version.cache_clear()
        env = self.julia_env(julia)
        env["JULIA_DEPOT_PATH"] = str(self.root() / "depot")
        env["IJULIA_NODEFAULTKERNEL"] = "1"
        kenv = self.kernel_env(self._version(julia))
        kenv.mkdir(parents=True, exist_ok=True)
        run([julia, f"--project={kenv}", "-e", 'import Pkg; Pkg.add("IJulia"); Pkg.precompile()'], log, env)

    def _install_julia(self, log: Log) -> None:
        env = dict(os.environ)
        env.update(self.juliaup_env())
        script = download("https://install.julialang.org", log)
        args = ["--yes", f"--path={self.root() / 'juliaup'}", "--add-to-path=no"]
        run(["sh", script, *args, "--background-selfupdate=0", "--startup-selfupdate=0"], log, env)

    def notebook_kernelspec(self, language_info: dict[str, Any]) -> dict[str, str]:
        version = minor(str(language_info.get("version", "")))
        return {"name": f"julia-{version}", "display_name": f"Julia {version}", "language": "julia"}


BACKENDS: dict[str, Backend] = {
    b.name: b for b in (MatlabBackend(), PythonBackend(), RBackend(), JuliaBackend())
}
ALIASES = {"py": "python", "python3": "python", "jl": "julia", "rlang": "r"}


def get_backend(language: str) -> Backend:
    key = language.strip().lower()
    key = ALIASES.get(key, key)
    if key not in BACKENDS:
        raise ValueError(f"unknown language {language!r}; choose from {', '.join(BACKENDS)}")
    return BACKENDS[key]


def default_language() -> str:
    return get_backend(os.environ.get("MASH_DEFAULT_KERNEL") or "python").name


def language_status(cwd: Path) -> list[tuple[Backend, bool, str]]:
    return [(b, *b.status(cwd)) for b in BACKENDS.values()]
