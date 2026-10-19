#!/usr/bin/env bash
#SBATCH --job-name=compbench-build
#SBATCH --partition=preemptable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=460000
#SBATCH --time=03:00:00
#SBATCH --output=compbench-build-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
IMAGE_OUTPUT=${IMAGE_OUTPUT:-${SCRATCH:?Set SCRATCH}/compbench.sqsh}
IMAGE_TAG=${IMAGE_TAG:-compbench:latest}
[[ -f containers/Containerfile ]]
# Keep previously tested images intact; choose a new IMAGE_OUTPUT when rebuilding.
[[ ! -e "$IMAGE_OUTPUT" ]] || { echo "Image already exists: $IMAGE_OUTPUT" >&2; exit 1; }
mkdir -p "$(dirname "$IMAGE_OUTPUT")"
srun --ntasks=1 --mpi=none bash -c '[[ $(uname -m) == aarch64 ]]' || {
    echo "Build on a native Alps ARM64 node" >&2; exit 1;
}
srun --ntasks=1 --mpi=none podman build --platform=linux/arm64 \
    -f containers/Containerfile -t "$IMAGE_TAG" .
srun --ntasks=1 --mpi=none bash containers/import_image.sh "$IMAGE_TAG" "$IMAGE_OUTPUT"
srun --ntasks=1 --mpi=none podman image inspect "$IMAGE_TAG" --format '{{.Id}}'
sha256sum "$IMAGE_OUTPUT"
