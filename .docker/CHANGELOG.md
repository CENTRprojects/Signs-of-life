# Changelog: Docker and deployment (`.docker/`)

Changes to the images, compose stack and environment files. For the release summary and upgrade steps, see the root
[CHANGELOG.md](../CHANGELOG.md). For crawler code changes, see [app_domains/CHANGELOG.md](../app_domains/CHANGELOG.md).

## [Unreleased] — branch `2024.10.29-LBB-Updates` (compared to `main`)

### New

- **Grafana stack** in `docker-compose.yml`:
  - `grafana` (`grafana/grafana-oss`) on port 3000, configured by `conf/grafana/grafana.ini` and
    `conf/grafana/grafana_env`;
  - `renderer` (`grafana/grafana-image-renderer`) on port 8081;
  - data and plugins in `docker_volumes/grafana/`.
- **`crawler_no-screenshoots` service.**
  - A second crawler (2 replicas) that reads `.env_noscreenshots`, so it can run with sampling turned off.
  - It writes its data to `docker_volumes/crawler/data_no-screenshoots`.
  - `.env_noscreenshots` is git-ignored, and no example file is provided yet.
- **Helper scripts:**

  | Script | Purpose |
  |---|---|
  | `build_clean_docker.sh <version>` | Builds a clean image (`--no-cache`), tagged `docker-crawler:<version>_<YYYYMMDD>` |
  | `build_and_run_docker.sh` | `docker compose up --force-recreate --build -d`, then follows the logs |
  | `docker_show_logs.sh` | Follows the last 100 lines of the compose logs |

- **New settings in `.env.example`** (all documented inline):
  - Sampling: `DO_SAMPLING`, `SAMPLING_RATE`.
  - Request engine: `MAX_WORKERS`, `BATCH_SPLIT`, `MAX_PROCESSES`, `REQUEST_RATE_LIMIT`, `MAX_MB_SINGLE_URL`
    (20), `CONTROLLER_LIMIT` (5000).
  - Memory guard: `MEM_HIGH_PCT`, `MEM_CRITICAL_PCT`, `MEM_LOW_PCT`, `MIN_PROCESSES`, `MIN_WORKERS`,
    `MIN_WORKERS_POST_PROCESSING`.
- **Placeholders** in compose for a future NS resolver and an ELK stack.

### Changed

#### Compose (`docker-compose.yml`)

- **The crawler runs pre-built, pinned images** and no longer builds from source:
  - `crawler` uses `docker-crawler:v1.1.6_20260520`;
  - `crawler_no-screenshoots` uses `docker-crawler:v1.1.10_20260924`.

  The `build:` section and the `../app_domains` source mount are commented out, because the code is now inside
  the image.
- **Volumes moved out of the source tree**, into `.docker/docker_volumes/crawler/`:

  | Container path | `main` | Now |
  |---|---|---|
  | `logging`, `output`, `inter`, `processing`, `completed`, `done` | `../<dir>` | `./docker_volumes/crawler/<dir>` |
  | `inter/urls`, `inter/revisit` | inside `../inter` | separate mounts |
  | `data` | `./data` | `./docker_volumes/crawler/data` |

- **`restart: unless-stopped`** on every service.
- **Database name:** `dbpostgres` now creates `signs_of_life_crawler` (was `signs_of_life`), and the `expose`
  entry was removed.
- **Platform:** `platform: linux/amd64` was removed from the crawler service.

#### Image (`Dockerfile`)

- **Chrome and ChromeDriver** now come from the Chrome-for-Testing builds, pinned to the latest **146.x**. They
  are installed under `/opt/chrome-linux64` and `/usr/local/bin`, replacing the `google-chrome-stable` apt
  package.
- **apt sources:** apt uses `archive.debian.org` for bullseye, because the base image's sources have been
  archived.
- **Packages:** installs `jq`, `unzip`, `curl` and the Chrome runtime libraries (`libnss3`, `libgbm1`,
  `libasound2`, …), with `--no-install-recommends` and the apt lists cleaned up.
- **Layer caching:** each `requirements_*.txt` is copied just before its `pip install` step.
- **`app_domains/` is copied into the image.**

#### Environment (`.env.example`)

- **Database name:** `POSTGRES_DB` is now `signs_of_life_crawler`. Also set `DBNAME` to the same value in your
  `.env`; it isn't in the example yet.
- **Postgres port:** `POSTGRES_PORT` is now `5432/TCP`.
- **Single example file:** the root `.env.example` was removed; `.docker/.env.example` is now the only one.
- **Git-ignored:** `.env_*`, `*.env` and `*.env.*`.
