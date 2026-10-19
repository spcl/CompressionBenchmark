"""Download the pinned LAGRANTO sources and record their provenance."""

import sys
import urllib.request
from pathlib import Path

TAG = "2.7"
SVN_REVISION = 47
DOWNLOAD_TIMEOUT_SECONDS = 120
SOURCE_PATHS = (
    "caltra/caltra.f",
    "lib/inter.f",
    "lib/ioinp_cdo.f",
    "lib/iotra.f",
    "lib/libcdfio.f",
    "lib/libcdfplus.f",
    "lib/times.f",
)


def download_sources(work):
    base = f"https://svn.iac.ethz.ch/pub/lagranto.ecmwf/!svn/bc/{SVN_REVISION}/tags/{TAG}/"
    with (work / "caltra.build-info").open("w") as manifest:
        manifest.write(f"LAGRANTO ECMWF {TAG}; SVN revision {SVN_REVISION}; CDO input reader\n")
        for path in SOURCE_PATHS:
            url = base + path
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                data = response.read()
            (work / Path(path).name).write_bytes(data)
            manifest.write(url + "\n")


if __name__ == "__main__":
    download_sources(Path(sys.argv[1]))
