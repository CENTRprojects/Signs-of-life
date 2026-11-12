# Changelog

Changes to the Signs of Life crawler, by release.
This file has the summary and the upgrade steps. The details are in two component changelogs:

- [app_domains/CHANGELOG.md](app_domains/CHANGELOG.md): the crawler code (pipeline, DNS, screenshots, memory, database).
- [.docker/CHANGELOG.md](.docker/CHANGELOG.md): images, compose, Grafana, environment files.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] — branch `2024.10.29-LBB-Updates` (compared to `main`)

20 commits, 36 files.

### What's new

- **DNS data extraction.** A, AAAA, NS, TXT, SOA, CAA, CNAME, RRSIG, MX and DMARC records, plus a DKIM check, for
  every domain. The results go into `final.csv` and the database.
- **Adaptive memory guard.** The number of processes and workers goes down when RAM is high and back up when it
  drops, to prevent out-of-memory kills. Every adjustment is logged.
- **More reliable screenshots.** The crawler waits until the page is ready, discards blank and error pages,
  retries failures, and keeps a list of pending screenshots that it retries later.
- **Placeholder detection for parked pages**, for template text left on registrar pages.
- **Grafana dashboards** and a second crawler service without screenshots, in Docker compose.
- **Pre-run sanity check** (`sanity_tests.py`) for dependency versions and the database connection.

### What's changed

- **Each container keeps its own logs and sampling output**, named after the container hostname.
- **Different defaults:** sampling, database upload and DNS extraction are now **on**.
- **Different Chrome:** a pinned Chrome-for-Testing 146 build, run in the new headless mode with
  bot-detection flags removed.
- **Compose runs pre-built, versioned images** instead of building from source. Data volumes moved under
  `.docker/docker_volumes/`.
- **JavaScript re-visit is now limited to sampled pages.**
- **Dependency changes:** `dnspython` 2.3.0, `inltk` 0.9, and new pins for numpy, nltk, murmurhash and
  bottleneck.

### Upgrade steps

1. Set `DBNAME=signs_of_life_crawler` in `.env`. Compose creates this database, but `config.py` still defaults to
   `signs_of_life`.
2. Copy the new keys from `.docker/.env.example` into your `.env`: the `MEM_*` and `MIN_*` keys,
   `MAX_MB_SINGLE_URL`, `CONTROLLER_LIMIT`, `DO_SAMPLING` and `SAMPLING_RATE`.
3. Build a versioned image with `.docker/build_clean_docker.sh <version>`, then update the `image:` tags in
   `docker-compose.yml`.
4. Move existing data from `../logging`, `../output`, `../inter` and the other root folders into
   `.docker/docker_volumes/crawler/`.
5. Update any tooling that reads log files by name, because the names now end in `_<container>.log`.
6. No manual database step is needed. The new DNS, DMARC, DKIM and MX columns (migrations 7–9) are added
   automatically on the first connection.
