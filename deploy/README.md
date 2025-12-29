# Getting Started with Docker Compose

## Install Dependencies

You need:

- Docker Engine
- Docker Compose v2 (`docker compose ...`)
- (Optional, for NIM services) NVIDIA drivers + NVIDIA Container Toolkit

On Ubuntu, Docker Compose v2 is typically provided by the Docker *compose plugin* (package names vary by distro).

Sanity checks:

```bash
docker --version
docker compose version
```

## Set your personal NGC CLI API KEY Environment Variable

An NGC personal run key (with permissions scoped properly) is required
to download and run the NIMs.

```bash
export NGC_CLI_API_KEY=<YOUR NGC PERSONAL RUN KEY>
```

## Log in to nvcr.io

```bash
# Use the literal '$oauthtoken' for username and your NGC_CLI_API_KEY for the password
echo "$NGC_CLI_API_KEY" | docker login nvcr.io --username='$oauthtoken' --password-stdin
```



## Create your NIM cache and set its permissions

NIMs cache any model data to your local disk so that subsequent start times are faster.
You can put this cache anywhere on your system that you have permissions, but the recommended
location is `~/.cache/nim`. Note: the cache **must** have global read-write permissions so that
the NIM can write into it, so we strongly recommend against putting any files requiring higher security
in the NIM cache directory.

```bash
## Create the nim cache directory
mkdir -p ~/.cache/nim

## Make the NIM cache writable by the NIM
chmod -R 777 ~/.cache/nim

## Set an environment variable so that docker compose can find the cache
export HOST_NIM_CACHE=~/.cache/nim
```

## Start the Docker Compose Configuration

Recommended (auto-selects the right compose file for your platform):

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

If you want the “batteries included” flow that may also start host-native wrapper services (and optionally provision minimal assets on ARM64), use:

```bash
./scripts/start_everything.sh
```

Stop:

```bash
./scripts/stop_everything.sh
```

Notes:
- The dashboard stacks expose the MCP Server on `http://localhost:${MCP_SERVER_HOST_PORT:-8011}`.
- The AMD64 NIM stack will download large model assets into `HOST_NIM_CACHE` on first start (can take hours).

This will first pull the containers then start them. When the containers start they will
pull the model for each NIM. This process can take several hours; in the case of AlphaFold2
and other large models, **expect the model download step to take from three to seven hours**
even on a fast internet connection.
