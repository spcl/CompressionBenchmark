#!/usr/bin/env bash
# Scope the Enroot Podman cleanup workaround to this import invocation.
set -euo pipefail
IMAGE_TAG=${1:?Provide the Podman image tag}
IMAGE_OUTPUT=${2:?Provide the output SquashFS path}
WORK=$(mktemp -d)
trap 'rm -rf -- "$WORK"' EXIT
# Some Enroot versions hard-code `docker rm` in their Podman EXIT trap.
# Route it to Podman, whose `rm -f` also tolerates an already removed container.
cat > "$WORK/docker" <<'SH'
#!/usr/bin/env bash
# Enroot removes its current working directory before running this cleanup.
cd /
exec podman "$@"
SH
chmod +x "$WORK/docker"
PATH="$WORK:$PATH" enroot import -o "$IMAGE_OUTPUT" "podman://$IMAGE_TAG"
unsquashfs -s "$IMAGE_OUTPUT"
