# Changelog: crawler code (`app_domains/`)

Changes to the Python crawler. For the release summary and upgrade steps, see the root [CHANGELOG.md](../CHANGELOG.md).
For Docker and deployment, see [.docker/CHANGELOG.md](../.docker/CHANGELOG.md).

## [Unreleased] — branch `2024.10.29-LBB-Updates` (compared to `main`)

### New

#### DNS data extraction (`DNS_data_extraction.py`)

- New pipeline step, run in `main_domains.py` after parking classification.
- It queries each domain for these record types, each with its own switch in `config.py`:

  | Record | Switch | How it is queried |
  |---|---|---|
  | A, AAAA, NS, TXT, SOA, CAA, CNAME | `DO_DNS_<T>` | `RESOLVER_NAMESERVER` (default `9.9.9.9`), 2 s timeout, 5 s lifetime |
  | RRSIG | `DO_DNS_RRSIG` | DNSKEY query with EDNS DO bit |
  | MX | `DO_DNS_MX` | system resolver |
  | DMARC | `DO_DNS_DMARC` | TXT on `_dmarc.<domain>` |
  | DKIM | `DO_DNS_DKIM` | SRV query on `_domainkey.<domain>` sent to the domain's own NS. This checks existence only, with timeout `DNS_DKIM_TIMEOUT` (15 s). |

- `DO_DNS` turns all of them off together.
- Each record type adds three output columns: `dns_has_<T>`, `dns_value_<T>` and `dns_<T>_comment`. The comment
  holds errors such as `No answer`, `Timeout` or `No existing query name`.
- The columns are added to `EXPECTED_ORDER` (`main_domains.py`), `DEBUG_COLUMNS` and `TABLEAU_COLUMNS`
  (`formatting.py`).
- New database migrations, applied automatically by `output_processing.ConnectDB()`:
  - `sql/setup_crawler_db_7.sql`: A, AAAA, NS, TXT, SOA, CAA, CNAME and RRSIG columns.
  - `sql/setup_crawler_db_8.sql`: DMARC columns.
  - `sql/setup_crawler_db_9.sql`: DKIM and MX columns.

#### Adaptive memory guard (`memory_guard.py`)

- `MAX_PROCESSES`, `MAX_WORKERS` and `WORKERS_POST_PROCESSING` are now **ceilings**, no longer fixed values.
- A background thread checks memory every 5 s. It uses whichever is tighter: the container cgroup limit or the
  host RAM.

  | Memory use | Action |
  |---|---|
  | ≥ `MEM_CRITICAL_PCT` (90) | cut parallelism by 40%, then `gc.collect()` |
  | ≥ `MEM_HIGH_PCT` (80) | cut parallelism by 20% (at most every 30 s) |
  | < `MEM_LOW_PCT` (65) | raise parallelism by 20% toward the ceiling (60 s after the last cut) |

- Floors: `MIN_PROCESSES`, `MIN_WORKERS` and `MIN_WORKERS_POST_PROCESSING`.
- New helpers:
  - `run_adaptive()` runs the request batches.
  - `adaptive_parallel()` replaces `joblib.Parallel` in every pipeline step. It processes slices of 1000 items
    and re-checks memory before each slice.
  - `AdaptiveAsyncLimiter` throttles the async requests inside each worker process.
  - `release_workers()` kills idle joblib workers after classification, which frees the loaded ML models.
- Adjustments, new peaks and a summary for each file are logged to `logging/memory_<container>.log`.

#### Screenshots and sampling

- Before a screenshot (`_wait_for_page_ready`), the crawler waits for `document.readyState`, decoded images,
  loaded fonts and a lazy-load scroll, followed by a 2 s paint delay.
- Screenshots are validated (`_save_screenshot_if_valid`):
  - Chrome error pages (`chrome-error://`, `about:`, `#main-message`) are skipped.
  - Blank or single-colour images (pixel standard deviation < 8) are deleted.
- Failed screenshots are retried with a fresh driver (`_take_screenshot_with_new_driver`):
  - once at the end of each browser batch;
  - up to 3 more times after classification;
  - once more at the end of each file (`retry_pending_screenshots()`).
- Screenshots that still fail are saved to `pending_screenshots.json` in the sampling folder, so they can be
  re-run manually.
- New Chrome launches wait while free RAM is below 500 MB (`_wait_for_free_memory`).

#### Other additions

- **Placeholder detection** (`classification_parked.py`): on registrar pages, template text such as
  `"name": "Your Organization Name"` marks the page as parked (`PLACEHOLDERS`, `identify_placeholders()`).
- **`sanity_tests.py`**: checks the pinned versions of 30 dependencies and the database connection, and stops
  the run if numpy is wrong, a dependency is missing, or the database can't be reached.
- **`redirect_http_https_check.py`**: an HTTP→HTTPS redirect check. It is not yet called by the pipeline.
- **`utils.gather_url_status()`**: reads only `{url: to_revisit}` from the saved URL files, without loading the
  HTML.
- **New config keys** in `config.py`:
  - `CONTAINER_ID` (the hostname).
  - The DNS switches (`DO_DNS*`, `RESOLVER_NAMESERVER`, `DNS_DKIM_TIMEOUT`).
  - `SAVESCREENSHOOTSTODISK` and `UPLOADSCREENSHOOTSTOLABTOOLS`.

### Changed

#### Defaults (`config.py`)

| Key | `main` | Now |
|---|---|---|
| `DO_SAMPLING` | `False` | `True` |
| `USE_DB` | `False` | `True` |
| `DO_DNS` (+ all `DO_DNS_*`) | — | `True` |
| `SAMPLING_LOCAL_FOLDER` | `output/sampling` | `output/sampling/<file>_<container>/` (set per input file) |
| Sample upload to LABTOOLS | always | only when `UPLOADSCREENSHOOTSTOLABTOOLS=True` (default `False`) |

#### Request engine (`url_visitor.py`)

- **No second download.** For pages without `Content-Length`, the URL, history and status now come from the
  streamed response; the page is no longer fetched a second time.
- **Oversized pages are skipped.** If `Content-Length` is larger than `MAX_MB_SINGLE_URL`, the page is flagged
  `Page size too large error` and not downloaded.
- **Less memory per round.** `FileData.documents` is cleared before each visit round, and earlier runs are
  checked with `gather_url_status()` without loading HTML.
- **Sampled links are visited with the browser** at the end of each async request batch.
- **Parallelism adapts to memory.** `async_url_batch_visit` takes a `max_workers` argument from the memory
  guard.

#### Browser / Selenium (`url_visitor.initiate_browser_driver`)

- **Headless mode:** `--headless=new` instead of the old `headless`, and `webdriver.Chrome(options=…)` instead of
  the deprecated `chrome_options=`.
- **Lighter Chrome:** GPU, sync, translation, background networking and similar features are off, and the V8
  heap is capped at 128 MB per tab.
- **TLS:** certificate errors are ignored (`acceptInsecureCerts`).
- **Less bot detection:** `navigator.webdriver` is hidden, `AutomationControlled` is disabled, and the
  `enable-automation` switch is removed.
- **Load timeouts:**
  - The driver now has a page-load timeout.
  - When the timeout hits, the crawler calls `window.stop()` and continues, instead of failing.
  - The driver is always closed in a `finally` block, and is retried once if it fails to start.

#### Classification (`classification_parked.py`)

- **JS re-visit:** pages are re-visited with JavaScript only if they are both `to_revisit_with_js` **and**
  `to_sample`. On `main`, either was enough.
- **Registrar validation** is skipped when a placeholder is found.

#### Output and logging

- **Log files are named per container:** `main_perf_<container>.log`, `sam_p_<container>.log`,
  `req_p_<container>.log`, `db_perf_<container>.log` and so on.
- **`formatting.final_formatting()`** strips `;` from `final_url` and `target_url`.
- **`main_domains.py`:**
  - prints the process and available RAM;
  - frees the controller with `gc.collect()` after each chunk;
  - prints the start and finish times.
- **Removed:** `tracemalloc`, and direct `joblib.Parallel` / `tqdm` calls in the pipeline modules.

#### Dependencies

| Package | `main` | Now |
|---|---|---|
| `dnspython` | 1.16.0 | 2.3.0 |
| `inltk` | 0.8.1 | 0.9 |
| `numpy` (after torch) | — | 1.18.3 |
| `bottleneck` | — | 1.2.1 |
| `nltk` | — | 3.3.0 |
| `murmurhash` | — | 1.0.10 |

`requirements_before_torch.txt` now installs spaCy and murmurhash from binary wheels only
(`--only-binary=spacy,murmurhash`).
