"""
Wind resource backend for the Turbine Yield Estimator.

Fetches real long-term average wind speed data from NASA POWER (free, public,
no API key required) and converts it into hub-height Weibull parameters that
the frontend can use for AEP calculations.

Data source: NASA POWER Climatology API
https://power.larc.nasa.gov/docs/services/api/temporal/climatology/

NASA POWER gives ~50km grid resolution wind data derived from the MERRA-2
reanalysis, 1981-present. It's much coarser than Global Wind Atlas (250m,
terrain-corrected) but it's free, keyless, and reliable for a v1 backend.
Swapping in GWA later is a drop-in replacement for the fetch_wind_climatology
function below, once you've sorted API access with them.

Run locally:
    pip install fastapi uvicorn httpx --break-system-packages
    uvicorn wind_backend:app --reload --port 8000

Then the frontend can call: http://localhost:8000/api/wind-resource?lat=56.6&lon=-4.9&hub_height=80
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import math
import time
import asyncio

app = FastAPI(title="Wind Resource API")

# CORS restricted to the deployed frontend (plus localhost for development).
# The previous wildcard let any website on the internet call this API from
# their visitors' browsers, burning the free-tier quota. Note this only stops
# browser-based cross-origin use — direct curl/script access is unaffected
# (that would need auth or rate limiting, deliberately not added at this scale).
# If the frontend moves to a custom domain, add it here and redeploy.
ALLOWED_ORIGINS = [
    "https://windypins.netlify.app",
    # GitHub Pages staging (only active if Pages is enabled on the repo).
    "https://rowancgoffin-web.github.io",
    # Retained during the Netlify migration. Delete once the old site is gone.
    "https://gorgeous-swan-134cb1.netlify.app",
    "http://localhost:8000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Netlify deploy previews get a unique subdomain per PR/branch push
    # (deploy-preview-N--windypins.netlify.app and branch--windypins variants).
    # Previews are free under the credit model, so all iteration happens there;
    # this regex is what makes preview testing work against the live backend.
    allow_origin_regex=r"https://[a-z0-9-]+--windypins\.netlify\.app",
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Per-IP rate limiting (in-memory sliding window, no dependencies).
# CORS only restricts browsers; these endpoints are otherwise open GET proxies,
# and /api/era5-resource is expensive upstream (4 x 5-year hourly fetches).
# Limits are generous for humans, hostile to scripts. Memory is bounded.
import time as _time
from collections import defaultdict, deque

_RATE_BUCKETS: dict = defaultdict(deque)
_RATE_LIMITS = {  # endpoint-key: (max_requests, window_seconds)
    "era5": (10, 600),      # heavy: 10 per 10 min per IP
    "nearby": (30, 600),
    "default": (120, 600),
}

def _client_ip(request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _rate_check(request, key: str = "default"):
    limit, window = _RATE_LIMITS.get(key, _RATE_LIMITS["default"])
    now = _time.time()
    bucket = _RATE_BUCKETS[(key, _client_ip(request))]
    while bucket and bucket[0] < now - window:
        bucket.popleft()
    if len(bucket) >= limit:
        raise HTTPException(status_code=429, detail=f"Rate limit: {limit} requests per {window//60} min for this endpoint. The cache means repeated identical requests do not need re-fetching — please slow down.")
    bucket.append(now)
    # Bound total memory: drop oldest buckets if the table grows silly.
    if len(_RATE_BUCKETS) > 5000:
        for k in list(_RATE_BUCKETS.keys())[:1000]:
            del _RATE_BUCKETS[k]

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
NASA_POWER_MONTHLY_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# ERA5 via Open-Meteo's archive API: keyless, no registration, no async queue.
# This is why ERA5 is now viable when the CDS route was not. ~31 km resolution,
# hourly, 1940-present, with native 100 m wind components.
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# OpenStreetMap Overpass: neighbouring turbines for external wake (IEC 4.1b).
# Public Overpass instances filter requests without an identifying
# User-Agent (hence 406s from library-default UAs), and OSM usage policy
# requires one regardless. A mirror is tried before giving up.
# Multiple public Overpass instances, tried in order. Cloud-host IPs are
# shared, so the main instance often deprioritises/rejects them through no
# fault of ours — breadth of mirrors is the practical fix. Per-mirror timeout
# must keep the WORST-CASE total under the frontend's 45 s abort.
# (url, client_timeout_s). The budget is STAGGERED to fit the deployed
# frontend's 45 s abort: kumi gets the long slot because it is the mirror
# that historically served this backend's (shared, deprioritised) Render IP —
# slowly but successfully. A flat short timeout kills that working path;
# a flat long one blows the frontend budget. 6+25+6+6 = 43 s worst case.
OVERPASS_MIRRORS = [
    ("https://overpass-api.de/api/interpreter", 6.0),
    ("https://overpass.kumi.systems/api/interpreter", 25.0),
    ("https://overpass.private.coffee/api/interpreter", 6.0),
    ("https://maps.mail.ru/osm/tools/overpass/api/interpreter", 6.0),
]
OVERPASS_HEADERS = {"User-Agent": "windypins-backend/1.0 (+https://windypins.netlify.app)"}

# Simple in-memory cache: wind data doesn't change, so no need to hit NASA's
# servers repeatedly for the same location. Keyed by lat/lon rounded to 0.25
# degrees (roughly the native grid resolution) plus hub height.
_cache: dict[str, dict] = {}
CACHE_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days — this data is climatological, it won't go stale fast
CACHE_MAX_ENTRIES = 500  # was 5000: six caches x 5000 entries grew steadily on a
                         # long-lived free-tier instance and contributed to the
                         # Aug 2026 OOM. 500 still covers normal repeat use.

# At most two ERA5 fetches in flight at once. Each holds ~15 MB of hourly data
# after the streaming rework; unbounded concurrency was what turned a traffic
# spike into an out-of-memory restart. Extra callers queue rather than stack.
_ERA5_CONCURRENCY = asyncio.Semaphore(2)


def _cache_put(cache: dict, key: str, value: dict) -> None:
    """Insert with a crude size cap: evict the oldest ~20% when full. FIFO is
    fine here — entries are climatological and cheap to refetch."""
    if len(cache) >= CACHE_MAX_ENTRIES:
        for k in list(cache.keys())[: CACHE_MAX_ENTRIES // 5]:
            del cache[k]
    cache[key] = value


def _cache_key(lat: float, lon: float, hub_height: float, alpha: float) -> str:
    return f"{round(lat * 4) / 4}_{round(lon * 4) / 4}_{hub_height}_{alpha}"


def power_law_extrapolate(v_ref: float, h_ref: float, h_target: float, alpha: float) -> float:
    """
    Extrapolate wind speed from a reference height to hub height using the
    standard wind power law. alpha is user-supplied since, without site
    measurements at multiple heights, there's no way to derive a
    location-specific shear exponent — 0.14 is a common default for open
    onshore terrain, but forested or urban surroundings run higher (0.2-0.3+),
    and smooth offshore/flat terrain runs lower (~0.10-0.12).
    """
    return v_ref * (h_target / h_ref) ** alpha


def weibull_params_from_mean(v_mean: float, k: float = 2.0) -> tuple[float, float]:
    """
    Approximate Weibull scale (A) from mean wind speed for a given shape
    parameter k.
    """
    gamma_term = math.gamma(1 + 1 / k)
    A = v_mean / gamma_term
    return A, k


def estimate_k_from_monthly_variability(monthly_speeds: list[float]) -> float:
    """
    Rough estimate of the Weibull shape parameter k, used only when we don't
    have real per-sector data (i.e. no site centre / GWA rose set) — the
    NASA POWER climatology fallback previously just assumed k=2 (Rayleigh)
    everywhere, which is wrong: real k typically ranges ~1.5 (variable,
    storm-driven climates) to ~2.5+ (steady, consistently windy regimes).

    This uses the Justus & Mikhail (1976) empirical relation
        k ≈ (σ / mean)^-1.086
    which is normally applied to the standard deviation of the *actual*
    (hourly or sub-daily) wind speed distribution. We don't have that from
    a climatology endpoint — only 12 monthly means — so this substitutes
    the coefficient of variation of monthly means as a proxy for the true
    wind speed spread.

    IMPORTANT CAVEAT: monthly-mean variability only captures the seasonal
    cycle, not the much larger synoptic/diurnal variability that actually
    dominates a real Weibull k. This will UNDERSTATE the true spread in most
    climates (i.e. bias k slightly high / distribution slightly too narrow).
    It's a genuine improvement over a flat, location-blind k=2 default, but
    it is a heuristic proxy, not a rigorous derivation — treat it as better
    than nothing, not as good as real sub-daily data (which is what the GWA
    wind-rose path, used once a site centre is set, actually provides).
    """
    if not monthly_speeds or len(monthly_speeds) < 12:
        return 2.0
    mean = sum(monthly_speeds) / len(monthly_speeds)
    if mean <= 0:
        return 2.0
    variance = sum((v - mean) ** 2 for v in monthly_speeds) / len(monthly_speeds)
    stddev = math.sqrt(variance)
    cv = stddev / mean
    if cv <= 0:
        return 2.5
    k = cv ** (-1.086)
    return max(1.3, min(3.0, k))  # clamp to a physically plausible range


def isa_air_density(elevation_m: float) -> float:
    """
    Air density from elevation only, using the ICAO International Standard
    Atmosphere barometric formula (sea-level 15°C reference, 1.225 kg/m³).
    This ignores actual site temperature/pressure on the day, which a real
    IEC 61400-12 correction would use — elevation-only is the common
    simplified approximation for a screening tool.
    """
    return 1.225 * (1 - 2.25577e-5 * max(elevation_m, 0)) ** 5.25588


async def fetch_wind_climatology(lat: float, lon: float) -> tuple[float, float]:
    """
    Query NASA POWER for the long-term average wind speed at 50m above
    ground level for this location. Returns (annual_mean_ws, estimated_k).
    """
    params = {
        "parameters": "WS50M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "format": "JSON",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(NASA_POWER_URL, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="NASA POWER API request failed")
        data = resp.json()

    try:
        ws50m_block = data["properties"]["parameter"]["WS50M"]
        ws50m_annual = float(ws50m_block["ANN"])
    except (KeyError, TypeError):
        raise HTTPException(status_code=502, detail="Unexpected response shape from NASA POWER")

    month_keys = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    try:
        monthly = [float(ws50m_block[m]) for m in month_keys]
        k_estimate = estimate_k_from_monthly_variability(monthly)
    except (KeyError, TypeError, ValueError):
        k_estimate = 2.0  # fall back to the old assumption if monthly data is missing for any reason

    return ws50m_annual, k_estimate


@app.get("/api/wind-resource")
async def wind_resource(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hub_height: float = Query(80, ge=10, le=300, description="Turbine hub height in metres"),
    alpha: float = Query(0.14, ge=0.05, le=0.4, description="Wind shear exponent (power law). No site data means this is an assumption, not a measurement."),
):
    """
    Returns a hub-height wind resource estimate for a given point:
    mean wind speed, Weibull A/k parameters, and the data vintage/source.
    """
    key = _cache_key(lat, lon, hub_height, alpha)
    cached = _cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    v_50m, k_estimate = await fetch_wind_climatology(lat, lon)
    v_hub = power_law_extrapolate(v_50m, h_ref=50, h_target=hub_height, alpha=alpha)
    A, k = weibull_params_from_mean(v_hub, k=k_estimate)

    result = {
        "lat": lat,
        "lon": lon,
        "hub_height_m": hub_height,
        "shear_exponent_used": alpha,
        "mean_wind_speed_50m": round(v_50m, 2),
        "mean_wind_speed_hub": round(v_hub, 2),
        "weibull_A": round(A, 2),
        "weibull_k": round(k, 3),
        "weibull_k_note": "Estimated from seasonal (monthly) variability via the Justus-Mikhail relation — "
                          "a heuristic proxy, not derived from real sub-daily wind speed distribution. "
                          "See METHODOLOGY.md for the caveat on this.",
        "source": "NASA POWER climatology (MERRA-2 reanalysis, 1981-present, ~50km grid)",
        "note": "Screening-grade estimate. Not terrain-corrected — a real ridge or valley "
                "at this exact point could differ meaningfully from this grid-cell average.",
    }
    _cache_put(_cache, key, {"result": result, "fetched_at": time.time()})
    return result


_iav_cache: dict = {}


@app.get("/api/iav")
async def interannual_variability(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    """
    Interannual variability source data: year-by-year annual mean 50m wind
    speeds from NASA POWER's monthly time series (MERRA-2 derived, ~50 km
    grid). Returns the raw annual means; the frontend detrends the series
    and converts wind variability to energy variability through the active
    power-curve pipeline. IAV is a synoptic-scale (regional) quantity, so
    the cache key is deliberately coarse (0.25 deg), matching the
    wind-resource cache — one fetch typically covers a whole site.
    Years with any missing months are skipped rather than patched.
    """
    key = f"{round(lat * 4) / 4}_{round(lon * 4) / 4}"
    cached = _iav_cache.get(key)
    if cached and (time.time() - cached["fetched_at"] < CACHE_TTL_SECONDS):
        return cached["result"]

    params = {
        "parameters": "WS50M",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": "1984",
        "end": "2024",
        "format": "JSON",
    }
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.get(NASA_POWER_MONTHLY_URL, params=params)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="NASA POWER monthly API request failed")
        data = resp.json()

    try:
        block = data["properties"]["parameter"]["WS50M"]
    except (KeyError, TypeError):
        raise HTTPException(status_code=502, detail="Unexpected response shape from NASA POWER monthly API")

    # Keys are YYYYMM strings; NASA POWER uses month "13" for the year's
    # annual value, which we ignore, computing annual means ourselves so a
    # year with any missing month (-999 fill value) can be skipped cleanly.
    by_year: dict = {}
    for k, v in block.items():
        if len(k) != 6:
            continue
        year, month = k[:4], k[4:6]
        if month == "13":
            continue
        by_year.setdefault(year, []).append(float(v))

    years, annual_means = [], []
    for year in sorted(by_year.keys()):
        months = by_year[year]
        if len(months) == 12 and all(m > -100 for m in months):
            years.append(int(year))
            annual_means.append(sum(months) / 12.0)

    if len(annual_means) < 10:
        raise HTTPException(status_code=502, detail="Insufficient complete years in NASA POWER monthly record")

    result = {"years": years, "annual_means_50m": [round(v, 4) for v in annual_means], "n_years": len(years)}
    _cache_put(_iav_cache, key, {"result": result, "fetched_at": time.time()})
    return result


# ---------------------------------------------------------------------------
# ERA5 (Open-Meteo) long-term hourly resource + OSM external-wake neighbours
# ---------------------------------------------------------------------------
# ERA5 is ~31 km -- COARSER than GWA's 250 m microscale. It does NOT replace
# GWA. The division of labour is the one industry actually uses: reanalysis
# for the long-term temporal climate (rose shape, Weibull fitted to real
# hourly data, IAV, hysteresis), microscale model for spatial downscaling.
# Each does what it is good at.

def weibull_mle(speeds, calm_threshold=0.5, tol=1e-7, max_iter=200):
    """
    Maximum-likelihood Weibull fit to a wind speed sample.

    Solves the standard MLE equation for k by Newton iteration:
        1/k = (sum v^k ln v)/(sum v^k) - mean(ln v)
    then A = (mean(v^k))^(1/k).

    Calms below calm_threshold are dropped: ln(0) is undefined, and sub-0.5 m/s
    hours carry no energy and are below every cut-in speed in the library. This
    biases A/k slightly high versus a fit including calms -- documented, and the
    standard practice when fitting Weibull to reanalysis for energy purposes.
    """
    v = [s for s in speeds if s is not None and s > calm_threshold]
    n = len(v)
    if n < 100:
        return None
    ln_v = [math.log(x) for x in v]
    mean_ln = sum(ln_v) / n
    k = 2.0
    for _ in range(max_iter):
        vk = [x ** k for x in v]
        s_vk = sum(vk)
        if s_vk <= 0:
            return None
        s_vk_ln = sum(vk[i] * ln_v[i] for i in range(n))
        s_vk_ln2 = sum(vk[i] * ln_v[i] * ln_v[i] for i in range(n))
        f = s_vk_ln / s_vk - 1.0 / k - mean_ln
        df = (s_vk_ln2 * s_vk - s_vk_ln ** 2) / (s_vk ** 2) + 1.0 / (k * k)
        if abs(df) < 1e-12:
            break
        k_new = k - f / df
        if k_new <= 0.1:
            k_new = 0.1
        if abs(k_new - k) < tol:
            k = k_new
            break
        k = k_new
    A = (sum(x ** k for x in v) / n) ** (1.0 / k)
    return A, k


def sector_index(direction_deg, n_sectors=12):
    """Bin a meteorological direction into a sector centred on N, NNE... ."""
    width = 360.0 / n_sectors
    return int(math.floor((direction_deg % 360.0) / width + 0.5)) % n_sectors


def build_rose(speeds, directions, n_sectors=12):
    """
    12-sector rose fitted to real hourly data: per-sector frequency and
    MLE Weibull A/k. Sectors with too few hours for a stable fit fall back
    to the all-directions fit, flagged in `sectors_fitted`.
    """
    bins = [[] for _ in range(n_sectors)]
    total = 0
    for s, d in zip(speeds, directions):
        if s is None or d is None:
            continue
        bins[sector_index(d, n_sectors)].append(s)
        total += 1
    if total == 0:
        return None
    overall = weibull_mle(speeds)
    freqs, As, ks, fitted = [], [], [], []
    for b in bins:
        freqs.append(100.0 * len(b) / total)
        fit = weibull_mle(b)
        if fit is None:
            fit = overall
            fitted.append(False)
        else:
            fitted.append(True)
        As.append(fit[0] if fit else 0.0)
        ks.append(fit[1] if fit else 2.0)
    return {
        "frequency_pct": freqs,
        "weibull_A": As,
        "weibull_k": ks,
        "sectors_fitted": fitted,
        "n_hours": total,
    }


def _icing_counts(temps_2m, rh, dT):
    """
    IEC / IEA Task 19 screening criterion, chunk-safe half: returns
    (n_valid, n_hits) for hours with hub-height T < 0 C AND RH > 96%
    simultaneously, so a 20-year series can be reduced incrementally rather
    than held in memory. Trips at 1% of the year (see _icing_result).

    ERA5 gives 2 m temperature; hub-height T is estimated with a standard
    environmental lapse rate (dT, passed in). At 100 m that is only ~0.64 K,
    but it is in the right direction and cheap. RH is not lapse-corrected --
    RH generally rises with height in the boundary layer, so using 2 m RH is
    conservative (under-counts icing hours). Documented as such.
    """
    n = 0
    hits = 0
    for t, h in zip(temps_2m, rh):
        if t is None or h is None:
            continue
        n += 1
        if (t - dT) < 0.0 and h > 96.0:
            hits += 1
    return n, hits


def _icing_result(n, hits, dT):
    """Builds the icing result dict from accumulated counts (see _icing_counts)."""
    if n == 0:
        return None
    pct = 100.0 * hits / n
    return {
        "icing_hours_pct": round(pct, 3),
        "n_hours": n,
        "threshold_pct": 1.0,
        "criterion_tripped": pct > 1.0,
        "hub_temp_offset_K": round(dT, 2),
    }




def hysteresis_loss(speeds, cut_out, restart, hub_height=None):
    """
    High-wind hysteresis (IEC 61400-15 category 4b).

    The power curve already zeroes production above cut-out, so that energy is
    not double-counted. Hysteresis is the ADDITIONAL loss from the restart
    deadband: after a cut-out trip, the turbine stays down until wind falls to
    `restart` (typically ~cut_out - 3..5 m/s), so hours in [restart, cut_out)
    that FOLLOW a trip produce nothing despite being in-limits.

    This needs the chronological series -- it cannot be done from a Weibull fit,
    which is exactly why hourly ERA5 unlocks it. Returns lost in-limits hours as
    a fraction of in-limits hours; the frontend weights this by the energy those
    hours would have produced.
    """
    down = False
    inlimits = 0
    lost = 0
    for s in speeds:
        if s is None:
            continue
        if s >= cut_out:
            down = True
            continue
        inlimits += 1
        if down:
            # Restart occurs when wind falls TO the restart threshold, not
            # strictly below it. Using `<` here counted the boundary hour as
            # lost, overstating hysteresis by one hour per trip.
            if s <= restart:
                down = False
            else:
                lost += 1
    if inlimits == 0:
        return None
    return {
        "lost_hours": lost,
        "in_limits_hours": inlimits,
        "hysteresis_hours_pct": round(100.0 * lost / inlimits, 4),
        "cut_out_ms": cut_out,
        "restart_ms": restart,
    }


_era5_cache: dict = {}
_neighbours_cache: dict = {}


async def _fetch_era5_chunk(client, lat, lon, start, end):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "hourly": "wind_speed_100m,wind_direction_100m,temperature_2m,relative_humidity_2m",
        "wind_speed_unit": "ms",
        "timezone": "UTC",
    }
    resp = await client.get(OPEN_METEO_ARCHIVE_URL, params=params)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Open-Meteo ERA5 request failed ({resp.status_code})")
    return resp.json().get("hourly", {})


@app.get("/api/era5-resource")
async def era5_resource(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hub_height: float = Query(100, ge=10, le=300),
    start_year: int = Query(2005, ge=1950, le=2024),
    end_year: int = Query(2024, ge=1950, le=2025),
    cut_out: float = Query(25.0, ge=10, le=40),
    restart: float = Query(22.0, ge=5, le=40),
):
    """
    Long-term hourly ERA5 statistics: IAV, icing criterion, hysteresis.

    Returns SUMMARY STATISTICS ONLY -- the raw hourly series (~175k points for
    20 years) is reduced server-side and never sent to the browser.

    VALIDATION NOTE (July 2026): ERA5 was tested as a full resource baseline
    against London Array (+9% on A) and Braes of Doune (-16% on A). The ~31 km
    cell smooths terrain, making it unusable for onshore resource -- GWA
    remains the resource path. ERA5 is used here only for quantities where
    synoptic-scale data is appropriate: interannual variability, the IEC
    icing screening criterion, and (offshore only) high-wind hysteresis.
    The rose/Weibull fields are retained for diagnostics and comparison,
    NOT as the production resource input.
    """
    _rate_check(request, "era5")
    key = f"{round(lat*100)/100}_{round(lon*100)/100}_{hub_height}_{start_year}_{end_year}_{cut_out}_{restart}"
    cached = _era5_cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    # Chunked in 5-year blocks: a single 20-year hourly request is a large
    # payload and Open-Meteo has been known to reject or time out on them.
    # Chunking also means one bad block does not lose the whole fetch.
    #
    # MEMORY (Aug 2026): holding five 175k-element lists for 20 years, plus the
    # copies made downstream, peaked near the 512 MB instance limit and caused
    # an OOM restart under concurrent use. Everything that does not need the
    # whole series -- per-year and per-month means, the icing counters, the
    # mean wind speed -- is now reduced chunk by chunk, so temperature,
    # humidity and timestamps are never held for the full record. Only speed
    # and direction are accumulated, because the rose fit, the overall Weibull
    # MLE and the chronological hysteresis pass genuinely need the full series.
    # The arithmetic is unchanged: running sums give identical means, and the
    # rose/Weibull/hysteresis inputs are byte-for-byte what they were before.
    speeds, dirs = [], []
    year_sum: dict = {}
    year_n: dict = {}
    month_sum: dict = {}
    month_n: dict = {}
    icing_n = 0
    icing_hits = 0
    dT = 6.5 * (hub_height - 2.0) / 1000.0
    sum_valid = 0.0
    n_valid = 0

    async with _ERA5_CONCURRENCY, httpx.AsyncClient(timeout=90.0) as client:
        for y0 in range(start_year, end_year + 1, 5):
            y1 = min(y0 + 4, end_year)
            h = await _fetch_era5_chunk(client, lat, lon, f"{y0}-01-01", f"{y1}-12-31")
            c_speeds = h.get("wind_speed_100m", []) or []
            c_dirs = h.get("wind_direction_100m", []) or []
            c_temps = h.get("temperature_2m", []) or []
            c_rhs = h.get("relative_humidity_2m", []) or []
            c_times = h.get("time", []) or []

            # Mean wind: same population as the old `valid` list comprehension.
            for s in c_speeds:
                if s is not None:
                    sum_valid += s
                    n_valid += 1

            # Per-year and per-month running sums (guards match the originals).
            for ts, s in zip(c_times, c_speeds):
                if s is None or not ts:
                    continue
                y = ts[:4]
                year_sum[y] = year_sum.get(y, 0.0) + s
                year_n[y] = year_n.get(y, 0) + 1
                if len(ts) < 7:
                    continue
                m = ts[5:7]
                month_sum[m] = month_sum.get(m, 0.0) + s
                month_n[m] = month_n.get(m, 0) + 1

            n_i, h_i = _icing_counts(c_temps, c_rhs, dT)
            icing_n += n_i
            icing_hits += h_i

            speeds += c_speeds
            dirs += c_dirs
            # Release the chunk payload before requesting the next one.
            del h, c_speeds, c_dirs, c_temps, c_rhs, c_times

    if len(speeds) < 8760:
        raise HTTPException(status_code=502, detail="ERA5 returned insufficient hourly data")

    rose = build_rose(speeds, dirs)
    if rose is None:
        raise HTTPException(status_code=502, detail="Could not fit wind rose to ERA5 data")
    overall = weibull_mle(speeds)

    # Annual means for IAV, straight from the hourly series -- a real
    # measured-structure IAV rather than one inferred from monthly means.
    years, ann = [], []
    for y in sorted(year_sum):
        if year_n[y] > 8000:  # tolerate a few gaps, reject part-years
            years.append(int(y))
            ann.append(year_sum[y] / year_n[y])

    # Monthly climatology from the same hourly series: mean 100 m wind per
    # calendar month across all years. Consumers get the SEASONAL SHAPE only —
    # the frontend scales it to the GWA-based annual energy, since ERA5's
    # absolute level is not the resource baseline (see role_note).
    monthly = []
    for mm in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
        cnt = month_n.get(mm, 0)
        monthly.append(round(month_sum[mm] / cnt, 3) if cnt else None)

    result = {
        "lat": lat, "lon": lon, "hub_height_m": hub_height,
        "monthly_mean_100m": monthly,
        "sectors_deg": [i * 30 for i in range(12)],
        "frequency_pct": [round(f, 4) for f in rose["frequency_pct"]],
        "weibull_A": [round(a, 3) for a in rose["weibull_A"]],
        "weibull_k": [round(k, 3) for k in rose["weibull_k"]],
        "sectors_fitted": rose["sectors_fitted"],
        "mean_wind_speed_100m": round(sum_valid / n_valid, 3) if n_valid else None,
        "weibull_A_overall": round(overall[0], 3) if overall else None,
        "weibull_k_overall": round(overall[1], 3) if overall else None,
        "n_hours": rose["n_hours"],
        "years": years,
        "annual_means_100m": [round(v, 4) for v in ann],
        "n_years": len(years),
        "icing": _icing_result(icing_n, icing_hits, dT),
        "hysteresis": hysteresis_loss(speeds, cut_out, restart),
        "hysteresis_note": "Validated for offshore/flat terrain only. ERA5's ~31 km cell "
                           "smooths terrain-accelerated gusts: at Braes of Doune (400 m upland "
                           "ridge) it reports zero hours above cut-out, which is a resolution "
                           "artefact, not a site property. Onshore complex-terrain sites should "
                           "treat 0.1-0.5% as the IEC-typical band and enter 4b manually.",
        "role_note": "Diagnostic/temporal data source. NOT the resource baseline: "
                     "validation against two UK farms showed ERA5 terrain smoothing of "
                     "-16% on wind speed at an upland site. GWA remains the resource path.",
        "source": "ERA5 reanalysis (~31 km, hourly) via Open-Meteo archive API",
        "caveat": "ERA5 is coarser than GWA (250 m) and is NOT terrain-corrected. Use it for "
                  "long-term temporal structure; use GWA for microscale spatial detail.",
        "weibull_note": "Weibull A/k fitted by maximum likelihood to real hourly data. Calms "
                        "below 0.5 m/s excluded (ln(0) undefined; sub-cut-in anyway), which "
                        "biases A high by ~0.5% and k high by ~1.5% at k=2, more at lower k.",
    }
    _cache_put(_era5_cache, key, {"result": result, "fetched_at": time.time()})
    return result


@app.get("/api/nearby-turbines")
async def nearby_turbines(
    request: Request,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: int = Query(20000, ge=1000, le=50000),
):
    """
    Neighbouring turbines from OpenStreetMap for external wake (IEC 4.1b).

    HONEST LIMITATION: OSM completeness varies by country. UK/DE/DK are well
    mapped; elsewhere is patchy. An unmapped neighbour returns silently as no
    loss -- an underestimate the user cannot see. The mitigation is visibility:
    the frontend renders these on the map and lets the user add or delete them,
    so a human validates the data rather than trusting it blind.
    """
    _rate_check(request, "nearby")
    # Snap to ~1 km. The wake loop ignores pairs beyond 25 km anyway, so a 1 km
    # shift in the query centre cannot change any turbine that actually
    # contributes -- and the coarser key means far more requests are served
    # from cache instead of hitting Overpass, which is the least reliable
    # dependency in the stack.
    key = f"{round(lat*100)/100}_{round(lon*100)/100}_{radius_m}"
    cached = _neighbours_cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    q = f"""[out:json][timeout:22];
(
  node(around:{radius_m},{lat},{lon})["generator:source"="wind"];
  way(around:{radius_m},{lat},{lon})["generator:source"="wind"];
);
out center tags;"""
    data = None
    last_err = "no response"
    async with httpx.AsyncClient(headers=OVERPASS_HEADERS) as client:
        for url, t_mirror in OVERPASS_MIRRORS:
            try:
                resp = await client.post(url, data={"data": q}, timeout=t_mirror)
                if resp.status_code == 200:
                    data = resp.json()
                    break
                last_err = f"HTTP {resp.status_code} from {url.split('/')[2]}"
            except Exception as exc:
                last_err = f"{type(exc).__name__} from {url.split('/')[2]}"
    if data is None:
        # Overpass is rate-limited and periodically flaky. Degrade to "found
        # nothing, say so" rather than failing the whole analysis. This must
        # NOT raise: an external-wake lookup failure is a data gap, not an
        # analysis failure.
        return {"turbines": [], "count": 0, "available": False,
                "note": f"OpenStreetMap lookup unavailable ({last_err}). "
                        "Add neighbouring turbines manually if external wake matters here."}

    out = []
    for el in data.get("elements", []):
        c = el.get("center") or el
        la, lo = c.get("lat"), c.get("lon")
        if la is None or lo is None:
            continue
        tags = el.get("tags", {}) or {}

        def _num(*keys):
            for k in keys:
                v = tags.get(k)
                if v is None:
                    continue
                try:
                    return float(str(v).split()[0].replace(",", "."))
                except ValueError:
                    continue
            return None

        rotor = _num("rotor:diameter", "generator:rotor:diameter")
        hub = _num("height:hub", "generator:height:hub")
        out.append({
            "lat": la, "lon": lo,
            "rotor_diameter_m": rotor,
            "hub_height_m": hub,
            "assumed": rotor is None or hub is None,
            "name": tags.get("name"),
            "operator": tags.get("operator"),
            "osm_id": el.get("id"),
        })

    n_assumed = sum(1 for t in out if t["assumed"])
    result = {
        "turbines": out, "count": len(out), "available": True,
        "radius_m": radius_m,
        "n_assumed_geometry": n_assumed,
        "source": "OpenStreetMap via Overpass API (power=generator, generator:source=wind)",
        "completeness_warning": "OSM coverage is not guaranteed complete. Turbines that exist "
                                "but are unmapped will not appear and will not be counted as "
                                "external wake. Verify against the map before relying on this.",
    }
    _cache_put(_neighbours_cache, key, {"result": result, "fetched_at": time.time()})
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Site elevation + terrain speed-up: air density, terrain complexity flag,
# and a genuine (if simplified) orographic correction
# ---------------------------------------------------------------------------
# Uses Open-Elevation (free, keyless, https://open-elevation.com) to sample
# the turbine's own elevation plus a ring of 8 points ~2km away, in a single
# batched request. This gives us three things:
#
#   1. Air density at the turbine (ISA barometric formula) — corrects the
#      power curve for elevation, since manufacturer curves are certified at
#      standard sea-level density.
#
#   2. A terrain complexity flag — if elevation varies a lot nearby, real
#      orographic effects are likely present that the correction below only
#      partially captures.
#
#   3. A terrain speed-up/slowdown correction — this is the real fix for the
#      biggest known gap in this tool (see METHODOLOGY.md §8/§11). It is NOT
#      a WAsP-equivalent orographic flow model (that requires a full linearized
#      spectral solution over an actual terrain grid, a genuine commercial
#      product's worth of engineering). It IS a simplified version of the same
#      underlying physics WAsP's own orographic module is built on:
#
#      Jackson & Hunt (1975) linear flow theory gives the fractional wind
#      speed-up over an isolated 2D hill, at the surface near the crest, as
#      approximately:
#
#          ΔS/S ≈ 2H/L
#
#      where H is the hill height above its surroundings and L is the
#      hill's characteristic half-length. This is the standard textbook
#      approximation (see e.g. Troen & Petersen 1989, the original WAsP
#      reference) for how much a hill or ridge speeds up the wind at its
#      crest relative to the regional/undisturbed wind speed.
#
#      We approximate H as the turbine's own elevation minus the mean
#      elevation of an 8-point ring ~2km around it ("local prominence" — how
#      much higher this exact point sits than its surroundings), and L as
#      the 2km ring radius. A turbine sitting on a local high point gets a
#      positive speed-up; one sitting in a local dip gets a slowdown. The
#      result is clamped to ±25% to avoid over-claiming precision this
#      simplified, isotropic (direction-blind) approximation doesn't have.
#
#      KNOWN LIMITATIONS of this correction (documented in METHODOLOGY.md):
#      - Isotropic: applied equally regardless of wind direction, when real
#        speed-up is highly direction-dependent (only the upwind slope
#        matters for a given wind direction).
#      - Uses only 8 sampled points, not a full terrain grid — a real ridge
#        or valley shape narrower than the 2km sampling radius won't be
#        captured accurately.
#      - No roughness-change or displacement-height modelling, no
#        atmospheric stability dependence — WAsP's full model includes both.
#      - This is a genuine improvement over doing nothing, not a substitute
#        for a real WAsP/CFD terrain assessment before any investment decision.
OPEN_ELEVATION_URL = "https://api.open-elevation.com/api/v1/lookup"
_elevation_cache: dict[str, dict] = {}

TERRAIN_RING_RADIUS_M = 2000
MAX_SPEEDUP_FRACTION = 0.25  # clamp — this simplified model shouldn't claim more precision than this


def _offset_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> dict:
    """Offset a lat/lon point by a given bearing and distance (small-distance approximation)."""
    dlat = (distance_m * math.cos(math.radians(bearing_deg))) / 111320.0
    dlon = (distance_m * math.sin(math.radians(bearing_deg))) / (111320.0 * max(math.cos(math.radians(lat)), 0.01))
    return {"latitude": lat + dlat, "longitude": lon + dlon}


@app.get("/api/site-elevation")
async def site_elevation(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
):
    # Cache at 4 decimal places (~11 m). The previous 2-dp key (~0.7-1.1 km)
    # caused turbines within ~1 km of each other to SHARE one cache slot: whichever
    # turbine's request completed first had its terrain/elevation result served to
    # its neighbours, race-dependent across reloads. Found via a user bug report
    # where two turbines 537 m apart swapped elevation values between sessions,
    # moving gross yield ~2%. Elevation is point data — unlike the wind-resource
    # cache (deliberately coarse to match NASA POWER's ~50 km grid), it must not
    # be shared between distinct turbine positions.
    key = f"{round(lat * 10000) / 10000}_{round(lon * 10000) / 10000}"
    cached = _elevation_cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    # Centre point + an 8-point ring at ~2km for the terrain speed-up estimate,
    # plus a closer 4-point ring at ~1km (kept from before) for the complexity flag.
    ring_2km = [_offset_point(lat, lon, bearing, TERRAIN_RING_RADIUS_M) for bearing in range(0, 360, 45)]
    ring_1km = [_offset_point(lat, lon, bearing, 1000) for bearing in [0, 90, 180, 270]]
    points = [{"latitude": lat, "longitude": lon}] + ring_2km + ring_1km

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(OPEN_ELEVATION_URL, json={"locations": points})
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail="Open-Elevation API request failed")
            data = resp.json()
        elevations = [r["elevation"] for r in data["results"]]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Elevation lookup failed: {exc}")

    centre_elevation = elevations[0]
    ring_2km_elevations = elevations[1:9]     # the 8-point 2km ring, in order
    ring_1km_elevations = elevations[9:13]    # the 4-point 1km ring

    # Terrain complexity: based on actual SLOPE (rise/run), not raw elevation range.
    # A 40m elevation change over 1km is only a 4% grade — completely ordinary rolling
    # terrain across most of the UK, not something that breaks the linear flow theory
    # this tool's terrain correction relies on. Real "complex terrain" in the sense
    # that matters (where linear theory like Jackson & Hunt starts to break down, and
    # where GWA's own RIX metric would flag caution) means genuinely steep slopes —
    # RIX itself is defined as the fraction of surrounding terrain exceeding 30% slope.
    # We use a somewhat more cautious 18% threshold here, checked across all 12 sampled
    # points (both rings) against their known distance from the turbine.
    TERRAIN_SLOPE_THRESHOLD = 0.18
    max_slope = 0.0
    for elev, dist in [(e, 2000) for e in ring_2km_elevations] + [(e, 1000) for e in ring_1km_elevations]:
        slope = abs(elev - centre_elevation) / dist
        if slope > max_slope:
            max_slope = slope
    elevation_range = max(ring_1km_elevations + [centre_elevation]) - min(ring_1km_elevations + [centre_elevation])

    # Terrain speed-up: local prominence (this point vs. its 2km surroundings),
    # via the simplified Jackson & Hunt 2H/L relation — see the module docstring
    # above for the full explanation and caveats.
    mean_surrounding_elevation = sum(ring_2km_elevations) / len(ring_2km_elevations)
    prominence_m = centre_elevation - mean_surrounding_elevation
    raw_speedup_fraction = (2 * prominence_m) / TERRAIN_RING_RADIUS_M
    speedup_fraction = max(-MAX_SPEEDUP_FRACTION, min(MAX_SPEEDUP_FRACTION, raw_speedup_fraction))

    air_density = isa_air_density(centre_elevation)

    result = {
        "lat": lat,
        "lon": lon,
        "elevation_m": round(centre_elevation, 1),
        "elevation_range_1km_m": round(elevation_range, 1),
        "max_slope_pct": round(max_slope * 100, 1),
        "terrain_complex": max_slope > TERRAIN_SLOPE_THRESHOLD,
        "prominence_m": round(prominence_m, 1),
        "terrain_speedup_fraction": round(speedup_fraction, 4),
        "terrain_speedup_clamped": abs(raw_speedup_fraction) > MAX_SPEEDUP_FRACTION,
        "air_density_kgm3": round(air_density, 4),
        "air_density_ratio": round(air_density / 1.225, 4),
        "source": "Open-Elevation (SRTM/derived DEM); terrain speed-up via simplified Jackson & Hunt (1975) linear flow theory",
        "note": "Terrain speed-up is a simplified, isotropic (direction-blind) approximation of the same "
                "physics WAsP's orographic model uses — a real improvement over no terrain correction at all, "
                "but not equivalent to a full WAsP/CFD terrain assessment. See METHODOLOGY.md §11.",
    }
    _cache_put(_elevation_cache, key, {"result": result, "fetched_at": time.time()})
    return result


# ---------------------------------------------------------------------------
# Wind rose: real Global Wind Atlas generalized wind climate (GWC) data
# ---------------------------------------------------------------------------
# GWA's GWC files contain sector-wise frequency of occurrence plus Weibull A/k
# per sector, at 5 reference roughness lengths (0.0, 0.03, 0.1, 0.4, 1.5m) and
# 5 heights (10/50/100/150/200m). This is the real methodology used by WAsP
# and every serious wind resource tool — NASA POWER's climatology endpoint
# (used elsewhere in this backend) can't produce this, since it only gives a
# single annual mean, not a directional distribution.
#
# The wind-stats library (pip install wind-stats) wraps GWA's API and returns
# this as an xarray Dataset. We interpolate to the requested hub height and a
# roughness length, since we don't have local land-cover data to pick one
# precisely — 0.03m ("open agricultural land, few buildings") is a common
# default for open onshore terrain.
try:
    from wind_stats import get_gwc_data
    WIND_STATS_AVAILABLE = True
except ImportError:
    WIND_STATS_AVAILABLE = False

_rose_cache: dict[str, dict] = {}


@app.get("/api/wind-rose")
def wind_rose(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    hub_height: float = Query(80, ge=10, le=200),
    roughness: float = Query(0.03, description="Surface roughness length in metres. 0.0=water, 0.03=open farmland, 0.1=scattered obstacles, 0.4=many obstacles/forest, 1.5=urban/city centre."),
):
    """
    Returns a 12-sector wind rose (frequency %, Weibull A, Weibull k per
    sector) from Global Wind Atlas GWC data, interpolated to hub height
    and the given roughness length. Note this is still not terrain-corrected
    at your exact point — GWC data represents a generalized regional climate,
    which is the input to microscale modelling (e.g. WAsP), not the output.
    """
    if not WIND_STATS_AVAILABLE:
        raise HTTPException(status_code=501, detail="wind-stats package not installed on this backend")

    # 4-dp key (~11 m): GWA GWC data has ~250 m effective resolution, so the
    # previous 0.25-degree (~28 km) rounding could serve one site centre's rose
    # to a different site tens of km away.
    key = f"{round(lat * 10000) / 10000}_{round(lon * 10000) / 10000}_{hub_height}_{roughness}"
    cached = _rose_cache.get(key)
    if cached and (time.time() - cached["fetched_at"]) < CACHE_TTL_SECONDS:
        return cached["result"]

    try:
        ds = get_gwc_data(lat, lon)
        interpolated = ds.interp(height=hub_height, roughness=roughness)
        sectors = ds["sector"].values.tolist()
        frequency = interpolated["frequency"].values.tolist()
        weibull_A = interpolated["A"].values.tolist()
        weibull_k = interpolated["k"].values.tolist()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Global Wind Atlas lookup failed: {exc}")

    result = {
        "lat": lat,
        "lon": lon,
        "hub_height_m": hub_height,
        "roughness_m": roughness,
        "sectors_deg": sectors,
        "frequency_pct": frequency,
        "weibull_A": weibull_A,
        "weibull_k": weibull_k,
        "source": "Global Wind Atlas v3 GWC (generalized wind climate, not terrain-corrected at this exact point)",
    }
    _cache_put(_rose_cache, key, {"result": result, "fetched_at": time.time()})
    return result
