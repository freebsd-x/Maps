// Maps.xapp — Tile-based map for FreeBSD-X
//
// Layout:
//   Top toolbar (48px): search bar + Search button | + / - zoom | 📍 location pin
//   Map area (flex):    GestureDetectorWidget wrapping a composited ImageWidget
//
// Features:
//   • OSM slippy tiles fetched via libcurl (imaps → https://tile.openstreetmap.org)
//   • Disk cache at ~/.freebsdx-maps-cache/{z}/{x}/{y}.png
//   • In-memory tile cache (up to kMemCacheMax tiles)
//   • Composite RGBA buffer: blit tiles → single ImageWidget (PDFViewer pattern)
//   • Pan via GestureDetectorWidget onPan(dx,dy)
//   • Pinch-to-zoom via GestureDetectorWidget onPinch(scale,dx,dy)
//   • Zoom buttons (integer snap) + keyboard +/-
//   • Nominatim geocoding (OSM) for text search → fly-to animation
//   • POI markers rendered as filled circles in the composite buffer
//   • GPS location pin placeholder (gray until com.freebsdx.GPS service exists)
//   • User-Agent header required by OSM tile policy

#include "xkit/widget.h"
#include "xkit/application.h"
#include "xkit/xbus_client.h"
#include "shared/protocol.h"
#include <curl/curl.h>
#include <png.h>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <list>

using namespace fbx::xkit;
using namespace fbx::shared;
namespace fs = std::filesystem;

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int    kTileSize      = 256;
static constexpr int    kMapW          = 1100;
static constexpr int    kMapH          = 640;
static constexpr int    kZoomMin       = 0;
static constexpr int    kZoomMax       = 18;
static constexpr int    kMemCacheMax   = 256;   // max decoded tiles in memory
static constexpr double kDefaultLat    = 48.8566;  // Paris
static constexpr double kDefaultLon    = 2.3522;
static constexpr int    kDefaultZoom   = 12;
static constexpr double kAnimSteps     = 20.0;  // fly-to animation frames

static const char* kTileUrlFmt     = "https://tile.openstreetmap.org/%d/%d/%d.png";
static const char* kNominatimUrlFmt =
    "https://nominatim.openstreetmap.org/search?q=%s&format=json&limit=1&"
    "accept-language=en";
static const char* kUserAgent = "FreeBSD-X Maps/1.0 (com.freebsdx.maps; libcurl)";
// IP-based geolocation fallback (free, no key, HTTP-only on the free tier). Only
// used when the GPS service has no fix AND the user has opted in — see
// ipGeoEnabled(). Sending the client IP to a third party is the inherent cost of
// IP geolocation, so it is off by default.
static const char* kIpApiUrl = "http://ip-api.com/json";

// ── Tile key ──────────────────────────────────────────────────────────────────

struct TileKey {
    int z, x, y;
    bool operator==(const TileKey& o) const noexcept {
        return z == o.z && x == o.x && y == o.y;
    }
};
struct TileKeyHash {
    size_t operator()(const TileKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.z);
        h ^= static_cast<size_t>(k.x) * 2654435761ULL + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= static_cast<size_t>(k.y) * 2246822519ULL + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ── POI ───────────────────────────────────────────────────────────────────────

struct Poi {
    double      lat, lon;
    std::string name;
    uint8_t     r{220}, g{50}, b{50};  // marker colour
};

// ── Map state ─────────────────────────────────────────────────────────────────

static double             g_lat      = kDefaultLat;
static double             g_lon      = kDefaultLon;
static int                g_zoom     = kDefaultZoom;
static std::vector<Poi>   g_pois;

// Memory tile cache (LRU eviction via doubly-linked list + map)
static std::mutex         g_cache_mutex;
using TilePixels = std::vector<uint8_t>;  // RGBA, 256×256×4
struct CacheEntry {
    TileKey    key;
    TilePixels pixels;
};
static std::list<CacheEntry>                                      g_lru_list;
static std::unordered_map<TileKey, std::list<CacheEntry>::iterator,
                          TileKeyHash>                            g_cache_map;

// Set of tiles currently being fetched (to avoid duplicate requests)
static std::mutex              g_pending_mutex;
static std::unordered_map<TileKey, bool, TileKeyHash> g_pending;

// Rebuild hook (set in main after widget creation)
static std::function<void()>   g_redraw;

// XBus client, set inside the Run() init lambda so button handlers created before
// Run() (e.g. the GPS button) can reach the bus. Null until the app is running.
static XBusClient*             g_xbus = nullptr;

// ── Helpers: coordinate math ──────────────────────────────────────────────────

static double lon2tx(double lon, int z) noexcept {
    return (lon + 180.0) / 360.0 * (1 << z);
}
static double lat2ty(double lat, int z) noexcept {
    double lr = lat * M_PI / 180.0;
    return (1.0 - std::log(std::tan(lr) + 1.0 / std::cos(lr)) / M_PI) / 2.0 * (1 << z);
}
static double tx2lon(double x, int z) noexcept {
    return x / (1 << z) * 360.0 - 180.0;
}
static double ty2lat(double y, int z) noexcept {
    double n = M_PI - 2.0 * M_PI * y / (1 << z);
    return 180.0 / M_PI * std::atan(0.5 * (std::exp(n) - std::exp(-n)));
}

// Pixel coordinates of a lat/lon in the composite canvas
static void latlonToPixel(double lat, double lon, int z,
                           double cx, double cy,
                           int& px, int& py) noexcept {
    double tx = lon2tx(lon, z);
    double ty = lat2ty(lat, z);
    // canvas center corresponds to (cx, cy) in tile-space
    double origin_x = kMapW / 2.0 - cx * kTileSize;
    double origin_y = kMapH / 2.0 - cy * kTileSize;
    px = static_cast<int>(origin_x + tx * kTileSize);
    py = static_cast<int>(origin_y + ty * kTileSize);
}

// ── Disk cache ────────────────────────────────────────────────────────────────

static fs::path cacheRoot() {
    const char* home = std::getenv("HOME");
    return fs::path(home ? home : "/tmp") / ".freebsdx-maps-cache";
}

static fs::path tilePath(int z, int x, int y) {
    return cacheRoot() / std::to_string(z)
                       / std::to_string(x)
                       / (std::to_string(y) + ".png");
}

static bool tileOnDisk(int z, int x, int y) {
    std::error_code ec;   // no throwing: the sandbox can't even stat the cache
    return fs::is_regular_file(tilePath(z, x, y), ec);
}

// ── PNG decode ────────────────────────────────────────────────────────────────

// Decode a PNG file to RGBA (256×256×4). Returns empty on failure.
// Decodes from an open stream and closes it.
static TilePixels decodePngStream(FILE* fp) {
    if (!fp) return {};

    uint8_t sig[8];
    if (std::fread(sig, 1, 8, fp) != 8 || png_sig_cmp(sig, 0, 8)) {
        std::fclose(fp);
        return {};
    }

    png_structp png = png_create_read_struct(
        PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return {}; }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(fp);
        return {};
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        return {};
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int w   = static_cast<int>(png_get_image_width(png, info));
    int h   = static_cast<int>(png_get_image_height(png, info));
    int ct  = png_get_color_type(png, info);
    int bd  = png_get_bit_depth(png, info);

    // Normalise to 8-bit RGBA
    if (bd == 16) png_set_strip_16(png);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_GRAY && bd < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (ct == PNG_COLOR_TYPE_RGB  || ct == PNG_COLOR_TYPE_GRAY ||
        ct == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    png_read_update_info(png, info);

    TilePixels pixels(static_cast<size_t>(w) * h * 4, 0xFF);
    std::vector<png_bytep> rows(h);
    for (int i = 0; i < h; ++i)
        rows[i] = pixels.data() + static_cast<ptrdiff_t>(i) * w * 4;
    png_read_image(png, rows.data());
    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);

    // If tile isn't exactly 256×256, scale trivially (shouldn't normally happen)
    if (w != kTileSize || h != kTileSize) {
        TilePixels scaled(kTileSize * kTileSize * 4, 0x88);
        int sw = std::min(w, kTileSize), sh = std::min(h, kTileSize);
        for (int r = 0; r < sh; ++r)
            std::memcpy(scaled.data() + r * kTileSize * 4,
                        pixels.data() + r * w * 4,
                        static_cast<size_t>(sw) * 4);
        return scaled;
    }
    return pixels;
}

static TilePixels decodePng(const std::string& path) {
    return decodePngStream(std::fopen(path.c_str(), "rb"));
}

// A tile straight from the downloaded bytes (when it can't be cached on disk).
static TilePixels decodePngMemory(std::string& body) {
    return decodePngStream(::fmemopen(body.data(), body.size(), "rb"));
}

// ── Memory cache ──────────────────────────────────────────────────────────────

// Returns nullptr if not in memory cache.
static const TilePixels* lookupCache(const TileKey& key) {
    // Caller must hold g_cache_mutex
    auto it = g_cache_map.find(key);
    if (it == g_cache_map.end()) return nullptr;
    // Move to front (most-recently-used)
    g_lru_list.splice(g_lru_list.begin(), g_lru_list, it->second);
    return &it->second->pixels;
}

static void insertCache(TileKey key, TilePixels pixels) {
    // Caller must hold g_cache_mutex
    auto it = g_cache_map.find(key);
    if (it != g_cache_map.end()) return;  // already present

    if ((int)g_lru_list.size() >= kMemCacheMax) {
        // Evict least-recently-used
        TileKey evict = g_lru_list.back().key;
        g_cache_map.erase(evict);
        g_lru_list.pop_back();
    }
    g_lru_list.push_front({key, std::move(pixels)});
    g_cache_map[key] = g_lru_list.begin();
}

// ── libcurl helpers ───────────────────────────────────────────────────────────

static size_t curlWrite(char* ptr, size_t sz, size_t nmemb, std::string* s) {
    s->append(ptr, sz * nmemb);
    return sz * nmemb;
}

// Fetch a URL into a byte buffer, returns false on error.
static bool httpGet(const std::string& url, std::string& body) {
    CURL* curl = curl_easy_init();
    if (!curl) return false;
    curl_easy_setopt(curl, CURLOPT_URL,           url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERAGENT,     kUserAgent);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWrite);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &body);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT,        15L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    CURLcode rc = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return rc == CURLE_OK;
}

// ── Tile fetching ─────────────────────────────────────────────────────────────

// Fetch tile from OSM, save to disk cache, decode, insert into memory cache.
// Runs in a background thread; calls g_redraw when done.
static void fetchTile(TileKey key) {
    // Build OSM URL
    char url[128];
    std::snprintf(url, sizeof(url), kTileUrlFmt, key.z, key.x, key.y);

    std::string body;
    if (!httpGet(url, body) || body.size() < 8) {
        std::lock_guard<std::mutex> lk(g_pending_mutex);
        g_pending.erase(key);
        return;
    }

    // Save to the disk cache if we can — inside the Capsicum sandbox the
    // cache directory can't be created; the throwing create_directories()
    // used to abort the whole app from this thread. Decode from memory then.
    auto path = tilePath(key.z, key.x, key.y);
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    bool cached = false;
    if (!ec) {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(body.data(), static_cast<std::streamsize>(body.size()));
        cached = static_cast<bool>(f);
    }

    // Decode PNG
    auto pixels = cached ? decodePng(path.string()) : decodePngMemory(body);

    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        if (!pixels.empty()) insertCache(key, std::move(pixels));
    }
    {
        std::lock_guard<std::mutex> lk(g_pending_mutex);
        g_pending.erase(key);
    }

    if (g_redraw) g_redraw();
}

// Get tile pixels (from memory → disk → network). Returns nullptr if unavailable.
// If fetching needed, spawns background thread and returns nullptr immediately.
static const TilePixels* getTile(const TileKey& key) {
    // 1. Memory cache
    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        if (auto* p = lookupCache(key)) return p;
    }

    // 2. Disk cache
    if (tileOnDisk(key.z, key.x, key.y)) {
        auto pixels = decodePng(tilePath(key.z, key.x, key.y).string());
        if (!pixels.empty()) {
            std::lock_guard<std::mutex> lk(g_cache_mutex);
            insertCache(key, pixels);
            return lookupCache(key);
        }
    }

    // 3. Schedule network fetch (once per tile)
    {
        std::lock_guard<std::mutex> lk(g_pending_mutex);
        if (g_pending.count(key)) return nullptr;
        g_pending[key] = true;
    }
    TileKey k = key;
    std::thread([k]() { fetchTile(k); }).detach();
    return nullptr;
}

// ── Composite rendering ───────────────────────────────────────────────────────

// Blit a decoded tile (RGBA 256×256) into the canvas buffer at pixel (px,py).
static void blitTile(std::vector<uint8_t>& canvas, int cw, int ch,
                     const TilePixels& tile, int px, int py) {
    int src_x0 = std::max(0, -px);
    int src_y0 = std::max(0, -py);
    int src_x1 = std::min(kTileSize, cw - px);
    int src_y1 = std::min(kTileSize, ch - py);
    if (src_x1 <= src_x0 || src_y1 <= src_y0) return;

    for (int r = src_y0; r < src_y1; ++r) {
        int dst_y = py + r;
        if (dst_y < 0 || dst_y >= ch) continue;
        std::memcpy(canvas.data() + (dst_y * cw + px + src_x0) * 4,
                    tile.data()  + (r * kTileSize + src_x0) * 4,
                    static_cast<size_t>(src_x1 - src_x0) * 4);
    }
}

// Draw a placeholder dark square for tiles that are still loading.
static void blitPlaceholder(std::vector<uint8_t>& canvas, int cw, int ch,
                             int px, int py) {
    int x0 = std::max(0, px);
    int y0 = std::max(0, py);
    int x1 = std::min(cw, px + kTileSize);
    int y1 = std::min(ch, py + kTileSize);
    for (int r = y0; r < y1; ++r) {
        for (int c = x0; c < x1; ++c) {
            uint8_t* p = canvas.data() + (r * cw + c) * 4;
            p[0] = 0x25; p[1] = 0x26; p[2] = 0x27; p[3] = 0xFF;
        }
    }
    // 1-pixel border
    for (int c = x0; c < x1; ++c) {
        uint8_t* t = canvas.data() + (y0 * cw + c) * 4;
        t[0] = t[1] = t[2] = 0x44; t[3] = 0xFF;
        if (y1 - 1 >= 0) {
            uint8_t* b = canvas.data() + ((y1 - 1) * cw + c) * 4;
            b[0] = b[1] = b[2] = 0x44; b[3] = 0xFF;
        }
    }
}

// Draw a filled circle (anti-aliased edge approximation) at (cx, cy) radius r.
static void drawCircle(std::vector<uint8_t>& canvas, int cw, int ch,
                       int cx, int cy, int r,
                       uint8_t cr, uint8_t cg, uint8_t cb, uint8_t ca = 255) {
    for (int y = cy - r - 1; y <= cy + r + 1; ++y) {
        for (int x = cx - r - 1; x <= cx + r + 1; ++x) {
            if (x < 0 || x >= cw || y < 0 || y >= ch) continue;
            double dist = std::sqrt((x - cx) * (x - cx) + (y - cy) * (y - cy));
            double alpha = std::clamp(static_cast<float>(r + 0.5 - dist), 0.0f, 1.0f);
            if (alpha <= 0.0) continue;
            uint8_t* p = canvas.data() + (y * cw + x) * 4;
            float a = alpha * ca / 255.0f;
            p[0] = static_cast<uint8_t>(p[0] * (1 - a) + cr * a);
            p[1] = static_cast<uint8_t>(p[1] * (1 - a) + cg * a);
            p[2] = static_cast<uint8_t>(p[2] * (1 - a) + cb * a);
            p[3] = 0xFF;
        }
    }
}

// Draw a teardrop-shaped location pin at (px, py) (pointing down).
static void drawPin(std::vector<uint8_t>& canvas, int cw, int ch,
                    int px, int py,
                    uint8_t r = 30, uint8_t g_c = 100, uint8_t b = 220) {
    // Circle body
    drawCircle(canvas, cw, ch, px, py - 10, 9, r, g_c, b);
    // White centre dot
    drawCircle(canvas, cw, ch, px, py - 10, 3, 255, 255, 255);
    // Triangle "tail" pointing down
    for (int dy = 0; dy <= 10; ++dy) {
        int half = static_cast<int>((10 - dy) * 0.5f + 0.5f);
        for (int dx = -half; dx <= half; ++dx) {
            int x = px + dx, y = py - dy;
            if (x < 0 || x >= cw || y < 0 || y >= ch) continue;
            uint8_t* p = canvas.data() + (y * cw + x) * 4;
            p[0] = r; p[1] = g_c; p[2] = b; p[3] = 0xFF;
        }
    }
}

// Draw the full composite map canvas.
static std::vector<uint8_t> renderMap(double lat, double lon, int zoom) {
    std::vector<uint8_t> canvas(kMapW * kMapH * 4);
    // Fill with a neutral dark-blue background (ocean colour)
    for (int i = 0; i < kMapW * kMapH; ++i) {
        canvas[i * 4 + 0] = 0x17;
        canvas[i * 4 + 1] = 0x2B;
        canvas[i * 4 + 2] = 0x3E;
        canvas[i * 4 + 3] = 0xFF;
    }

    double cx = lon2tx(lon, zoom);
    double cy = lat2ty(lat, zoom);
    double origin_x = kMapW / 2.0 - cx * kTileSize;
    double origin_y = kMapH / 2.0 - cy * kTileSize;

    int nTiles = 1 << zoom;
    int tx0 = static_cast<int>(std::floor(-origin_x / kTileSize));
    int ty0 = static_cast<int>(std::floor(-origin_y / kTileSize));
    int tx1 = static_cast<int>(std::ceil((kMapW - origin_x) / kTileSize));
    int ty1 = static_cast<int>(std::ceil((kMapH - origin_y) / kTileSize));

    for (int ty = ty0; ty <= ty1; ++ty) {
        for (int tx = tx0; tx <= tx1; ++tx) {
            int tx_w = ((tx % nTiles) + nTiles) % nTiles;
            int ty_c = std::max(0, std::min(nTiles - 1, ty));

            int px = static_cast<int>(origin_x + tx * kTileSize);
            int py = static_cast<int>(origin_y + ty * kTileSize);

            TileKey key{zoom, tx_w, ty_c};
            const TilePixels* tile;
            {
                std::lock_guard<std::mutex> lk(g_cache_mutex);
                tile = lookupCache(key);
            }
            if (!tile) {
                // Schedule fetch (getTile also checks disk)
                tile = getTile(key);
            }
            if (tile)
                blitTile(canvas, kMapW, kMapH, *tile, px, py);
            else
                blitPlaceholder(canvas, kMapW, kMapH, px, py);
        }
    }

    // Draw POI markers
    for (const auto& poi : g_pois) {
        int px, py;
        latlonToPixel(poi.lat, poi.lon, zoom, cx, cy, px, py);
        if (px >= -10 && px < kMapW + 10 && py >= -10 && py < kMapH + 10)
            drawPin(canvas, kMapW, kMapH, px, py, poi.r, poi.g, poi.b);
    }

    // Draw crosshair at map centre
    int mx = kMapW / 2, my = kMapH / 2;
    for (int d = -6; d <= 6; ++d) {
        if (d == 0) continue;
        auto set_pix = [&](int x, int y) {
            if (x < 0 || x >= kMapW || y < 0 || y >= kMapH) return;
            uint8_t* p = canvas.data() + (y * kMapW + x) * 4;
            p[0] = p[1] = p[2] = 255; p[3] = 180;
        };
        set_pix(mx + d, my);
        set_pix(mx, my + d);
    }

    return canvas;
}

// ── Nominatim geocoding ───────────────────────────────────────────────────────

// Very lightweight JSON extraction — parses first lat/lon from Nominatim response.
// Example input: [{"lat":"48.8566","lon":"2.3522","display_name":"Paris, France"...}]
// IP geolocation is opt-in: enabled when $FREEBSDX_MAPS_IP_GEO is set to a
// non-empty value other than "0"/"false". Default off (no surprise external call).
static bool ipGeoEnabled() {
    const char* e = std::getenv("FREEBSDX_MAPS_IP_GEO");
    return e && e[0] && std::strcmp(e, "0") != 0 && std::strcmp(e, "false") != 0;
}

// Parse ip-api.com/json: {"status":"success",...,"lat":48.85,"lon":2.35,"city":"…"}.
// Note values are JSON numbers (unquoted), unlike Nominatim's quoted strings.
static bool parseIpApi(const std::string& json,
                       double& out_lat, double& out_lon, std::string& out_city) {
    if (json.find("\"success\"") == std::string::npos) return false;   // status != success
    auto extractNum = [&](const char* key, double& out) -> bool {
        std::string k = std::string("\"") + key + "\":";
        auto pos = json.find(k);
        if (pos == std::string::npos) return false;
        pos += k.size();
        while (pos < json.size() && json[pos] == ' ') ++pos;
        size_t end = pos;
        while (end < json.size()) {
            char c = json[end];
            if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' ||
                c == 'e' || c == 'E') ++end;
            else break;
        }
        if (end == pos) return false;
        try { out = std::stod(json.substr(pos, end - pos)); } catch (...) { return false; }
        return true;
    };
    if (!extractNum("lat", out_lat) || !extractNum("lon", out_lon)) return false;
    // City is a quoted string; best-effort.
    std::string k = "\"city\":\"";
    if (auto pos = json.find(k); pos != std::string::npos) {
        pos += k.size();
        if (auto e = json.find('"', pos); e != std::string::npos)
            out_city = json.substr(pos, e - pos);
    }
    return true;
}

static bool parseNominatim(const std::string& json,
                            double& out_lat, double& out_lon,
                            std::string& out_name) {
    // Find first "lat":"..."
    auto extract = [&](const char* key) -> std::string {
        std::string k = std::string("\"") + key + "\":\"";
        auto pos = json.find(k);
        if (pos == std::string::npos) return {};
        pos += k.size();
        auto end = json.find('"', pos);
        if (end == std::string::npos) return {};
        return json.substr(pos, end - pos);
    };
    std::string lat_s = extract("lat");
    std::string lon_s = extract("lon");
    std::string name  = extract("display_name");
    if (lat_s.empty() || lon_s.empty()) return false;
    try {
        out_lat = std::stod(lat_s);
        out_lon = std::stod(lon_s);
    } catch (...) { return false; }
    // Truncate display_name at first comma
    auto comma = name.find(',');
    out_name = (comma != std::string::npos) ? name.substr(0, comma) : name;
    return true;
}

// URL-encode a query string (encode everything except unreserved chars).
static std::string urlEncode(const std::string& s) {
    std::string result;
    result.reserve(s.size() * 3);
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            result += static_cast<char>(c);
        } else {
            char hex[4];
            std::snprintf(hex, sizeof(hex), "%%%02X", c);
            result += hex;
        }
    }
    return result;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    curl_global_init(CURL_GLOBAL_DEFAULT);

    // ── State ─────────────────────────────────────────────────────────────────
    auto cur_lat     = std::make_shared<double>(kDefaultLat);
    auto cur_lon     = std::make_shared<double>(kDefaultLon);
    auto cur_zoom    = std::make_shared<int>(kDefaultZoom);
    auto is_syncing  = std::make_shared<std::atomic<bool>>(false);
    auto status_text = std::make_shared<std::string>("");
    // GPS: placeholder coordinates (0 = no fix)
    auto gps_lat     = std::make_shared<double>(0.0);
    auto gps_lon     = std::make_shared<double>(0.0);
    auto gps_fix     = std::make_shared<bool>(false);

    // ── Root layout ───────────────────────────────────────────────────────────
    //   FlexColumn
    //   ├── toolbar (FlexRow, fixed 48px)
    //   └── map area (flex, GestureDetector → map_container → ImageWidget)

    // Map container (replaced on each redraw)
    auto map_container = std::make_shared<FlexColumn>(
        std::vector<std::shared_ptr<Widget>>{});
    map_container->setFlexGrow(1.0f);
    map_container->setBackgroundColor(Color{0.09f, 0.17f, 0.24f, 1.0f});

    // Zoom label (e.g. "z13") shown in toolbar
    auto zoom_label = std::make_shared<TextWidget>(
        "z" + std::to_string(kDefaultZoom), 11.0f);
    zoom_label->setColor(Color{0.6f, 0.6f, 0.6f, 1.0f});
    zoom_label->setFixedSize(28.0f, 20.0f);

    // Status label (geocoding results / errors)
    auto status_lbl = std::make_shared<TextWidget>("", 11.0f);
    status_lbl->setColor(Color{0.65f, 0.65f, 0.55f, 1.0f});
    status_lbl->setFlexGrow(0.0f);

    // ── Redraw lambda ─────────────────────────────────────────────────────────
    g_redraw = [=]() {
        auto pixels = renderMap(*cur_lat, *cur_lon, *cur_zoom);
        auto img = std::make_shared<ImageWidget>(std::move(pixels), kMapW, kMapH);
        map_container->clearChildren();
        map_container->addChild(img);
    };

    // ── Gesture callbacks ─────────────────────────────────────────────────────
    auto on_pan = [=](float dx, float dy) {
        // Convert pixel delta to lon/lat delta
        int z   = *cur_zoom;
        double s = 1.0 / (kTileSize * (1 << z));
        *cur_lon -= dx * s * 360.0;
        *cur_lat  = ty2lat(lat2ty(*cur_lat, z) + dy * s * (1 << z), z);
        *cur_lat  = std::clamp(*cur_lat, -85.0511, 85.0511);
        *cur_lon  = std::fmod(*cur_lon + 180.0, 360.0) - 180.0;
        g_redraw();
    };

    auto on_pinch = [=](float scale, float cx, float cy) {
        if (std::abs(scale - 1.0f) < 0.01f) return;
        // Zoom toward pinch centre
        double factor = std::log2(static_cast<double>(scale));
        int    new_z  = std::clamp(static_cast<int>(std::round(*cur_zoom + factor)),
                                    kZoomMin, kZoomMax);
        if (new_z == *cur_zoom) return;

        // Keep the pinch centre lat/lon fixed
        int oz = *cur_zoom;
        double tx = lon2tx(*cur_lon, oz) + (cx - kMapW / 2.0) / kTileSize;
        double ty = lat2ty(*cur_lat, oz) + (cy - kMapH / 2.0) / kTileSize;
        double ratio = static_cast<double>(1 << new_z) / (1 << oz);
        *cur_lon = tx2lon(lon2tx(*cur_lon, oz) + (tx - lon2tx(*cur_lon, oz)) * (1 - ratio), oz);
        *cur_lat = ty2lat(lat2ty(*cur_lat, oz) + (ty - lat2ty(*cur_lat, oz)) * (1 - ratio), oz);

        *cur_zoom = new_z;
        zoom_label->setText("z" + std::to_string(new_z));
        g_redraw();
    };

    auto on_tap = [=](float tx, float ty) {
        // Tap: drop a temporary POI marker at the tapped location
        int    z  = *cur_zoom;
        double cx = lon2tx(*cur_lon, z);
        double cy = lat2ty(*cur_lat, z);
        double ox = kMapW / 2.0 - cx * kTileSize;
        double oy = kMapH / 2.0 - cy * kTileSize;
        double tile_x = (tx - ox) / kTileSize;
        double tile_y = (ty - oy) / kTileSize;
        double tap_lon = tx2lon(tile_x, z);
        double tap_lat = ty2lat(tile_y, z);
        tap_lat = std::clamp(tap_lat, -85.0511, 85.0511);
        // Drop a yellow POI at the tap position
        Poi poi;
        poi.lat = tap_lat;
        poi.lon = tap_lon;
        poi.name = std::to_string(tap_lat).substr(0, 7) + ", " +
                   std::to_string(tap_lon).substr(0, 7);
        poi.r = 240; poi.g = 180; poi.b = 40;
        g_pois.push_back(poi);
        status_lbl->setText("Pin: " + poi.name);
        g_redraw();
    };

    // Gesture detector wraps the map container
    auto gesture = std::make_shared<GestureDetectorWidget>(
        map_container, on_tap, on_pan, on_pinch);
    gesture->setFlexGrow(1.0f);

    // ── Toolbar ───────────────────────────────────────────────────────────────
    auto search_input = std::make_shared<TextInputWidget>();
    search_input->setPlaceholder("Search place…");
    search_input->setFontSize(13.0f);
    search_input->setFixedSize(0.0f, 30.0f);
    search_input->setFlexGrow(1.0f);

    auto search_btn = std::make_shared<ButtonWidget>(
        std::make_shared<TextWidget>("Search", 12.0f),
        [=]() {
            std::string q = search_input->getText();
            if (q.empty()) return;
            status_lbl->setText("Searching…");
            std::thread([=]() {
                char url[512];
                std::snprintf(url, sizeof(url), kNominatimUrlFmt,
                              urlEncode(q).c_str());
                std::string body;
                if (!httpGet(url, body)) {
                    try { status_lbl->setText("Network error"); } catch (...) {}
                    return;
                }
                double rlat, rlon;
                std::string rname;
                if (!parseNominatim(body, rlat, rlon, rname)) {
                    try { status_lbl->setText("Not found: " + q); } catch (...) {}
                    return;
                }
                // Fly-to: animate in kAnimSteps increments
                double from_lat = *cur_lat, from_lon = *cur_lon;
                for (int i = 1; i <= static_cast<int>(kAnimSteps); ++i) {
                    double t = i / kAnimSteps;
                    // Ease-in-out cubic
                    double ease = t < 0.5 ? 4 * t * t * t
                                          : 1 - std::pow(-2 * t + 2, 3) / 2;
                    *cur_lat = from_lat + (rlat - from_lat) * ease;
                    *cur_lon = from_lon + (rlon - from_lon) * ease;
                    try { g_redraw(); } catch (...) {}
                    std::this_thread::sleep_for(std::chrono::milliseconds(30));
                }
                // Drop a POI at the result
                Poi poi;
                poi.lat = rlat; poi.lon = rlon; poi.name = rname;
                poi.r = 80; poi.g = 160; poi.b = 220;
                g_pois.push_back(poi);
                try {
                    status_lbl->setText(rname);
                    zoom_label->setText("z" + std::to_string(*cur_zoom));
                    g_redraw();
                } catch (...) {}
            }).detach();
        });
    search_btn->setFixedSize(72.0f, 30.0f);
    search_btn->setBackgroundColor(Color{0.18f, 0.48f, 0.88f, 1.0f});

    // Zoom + button
    auto zoom_in = std::make_shared<ButtonWidget>(
        std::make_shared<TextWidget>("+", 18.0f),
        [=]() {
            if (*cur_zoom >= kZoomMax) return;
            ++*cur_zoom;
            zoom_label->setText("z" + std::to_string(*cur_zoom));
            g_redraw();
        });
    zoom_in->setFixedSize(34.0f, 30.0f);
    zoom_in->setBackgroundColor(Color{0.20f, 0.22f, 0.26f, 1.0f});

    // Zoom - button
    auto zoom_out = std::make_shared<ButtonWidget>(
        std::make_shared<TextWidget>("−", 18.0f),
        [=]() {
            if (*cur_zoom <= kZoomMin) return;
            --*cur_zoom;
            zoom_label->setText("z" + std::to_string(*cur_zoom));
            g_redraw();
        });
    zoom_out->setFixedSize(34.0f, 30.0f);
    zoom_out->setBackgroundColor(Color{0.20f, 0.22f, 0.26f, 1.0f});

    // Clear pins button
    auto clear_btn = std::make_shared<ButtonWidget>(
        std::make_shared<TextWidget>("✕ Pins", 11.0f),
        [=]() {
            g_pois.clear();
            status_lbl->setText("");
            g_redraw();
        });
    clear_btn->setFixedSize(60.0f, 30.0f);
    clear_btn->setBackgroundColor(Color{0.25f, 0.15f, 0.15f, 1.0f});

    // GPS / My Location button. Asks com.freebsdx.GPS for a fix; the service
    // replies and broadcasts a LocationUpdate (opcode 0x01) which the handler
    // below turns into a blue pin. If a fix is already cached, recentre now.
    auto gps_btn = std::make_shared<ButtonWidget>(
        std::make_shared<TextWidget>("📍", 16.0f),
        [=]() {
            if (*gps_fix) {
                *cur_lat = *gps_lat;
                *cur_lon = *gps_lon;
                status_lbl->setText("My location");
                g_redraw();
            }
            if (!g_xbus) return;
            status_lbl->setText("Locating…");
            // RequestLocation (0x02); see core/xserver/include/xserver/gps_service.h.
            g_xbus->Call("com.freebsdx.GPS", 0x02u, {}, [=](const XBusMessage& resp) {
                // Reply payload: GpsFixReply { u8 has_fix; u8 source; double lat,lon; ... }
                // A hardware/manual fix also arrives via the LocationUpdate
                // broadcast, which drops the pin and recentres — nothing to do here.
                if (resp.payload_[0]) return;

                // No GPS/manual fix. Fall back to IP geolocation if opted in — the
                // explicit tap is the consent; still gated so it never fires silently.
                if (!ipGeoEnabled()) {
                    status_lbl->setText("No GPS fix (set gps.lat/gps.lon, attach a "
                                        "receiver, or enable FREEBSDX_MAPS_IP_GEO)");
                    return;
                }
                status_lbl->setText("Locating via IP…");
                std::thread([=]() {
                    std::string body;
                    double lat, lon; std::string city;
                    if (httpGet(kIpApiUrl, body) && parseIpApi(body, lat, lon, city)) {
                        *gps_lat = lat; *gps_lon = lon; *gps_fix = true;
                        bool pinned = false;
                        for (auto& p : g_pois)
                            if (p.name == "GPS") { p.lat = lat; p.lon = lon; pinned = true; break; }
                        if (!pinned) g_pois.push_back({lat, lon, "GPS", 60, 200, 80});
                        *cur_lat = lat; *cur_lon = lon;   // recentre
                        try {
                            status_lbl->setText(city.empty() ? std::string("My location (IP)")
                                                             : ("My location: " + city));
                            g_redraw();
                        } catch (...) {}
                    } else {
                        try { status_lbl->setText("IP location unavailable"); } catch (...) {}
                    }
                }).detach();
            });
        });
    gps_btn->setFixedSize(36.0f, 30.0f);
    gps_btn->setBackgroundColor(Color{0.16f, 0.18f, 0.20f, 0.75f});

    auto toolbar = std::make_shared<FlexRow>(std::vector<std::shared_ptr<Widget>>{
        search_input,
        std::make_shared<Spacer>(6, 0),
        search_btn,
        std::make_shared<Spacer>(10, 0),
        zoom_out,
        std::make_shared<Spacer>(4, 0),
        zoom_label,
        std::make_shared<Spacer>(4, 0),
        zoom_in,
        std::make_shared<Spacer>(10, 0),
        gps_btn,
        std::make_shared<Spacer>(10, 0),
        clear_btn,
        std::make_shared<Spacer>(10, 0),
        status_lbl,
    });
    toolbar->setFixedSize(0.0f, 48.0f);
    toolbar->setPadding(9.0f);
    toolbar->setGap(0.0f);
    toolbar->setCrossAxisAlignment(CrossAxisAlignment::Center);
    toolbar->setBackgroundColor(Color{0.11f, 0.12f, 0.14f, 1.0f});

    // ── Root ─────────────────────────────────────────────────────────────────
    auto root = std::make_shared<FlexColumn>(std::vector<std::shared_ptr<Widget>>{
        toolbar,
        gesture,
    });
    root->setGap(0.0f);
    root->setBackgroundColor(Color{0.0f, 0.0f, 0.0f, 1.0f});

    // ── Pre-load from argv[1]: treat as "lat,lon" or "place name" ─────────────
    std::string preload_query;
    if (argc >= 2) preload_query = argv[1];

    const int rc = XApplication::Run(root, "/tmp/freebsdx.sock", /*sandboxed=*/false,
                             /*is_shell=*/false,
        [=](XBusClient& xbus) {
            XApplication::setTitle("Maps");
            g_xbus = &xbus;   // let pre-Run button handlers (GPS) reach the bus

            // Initial render (uses disk cache + schedules missing tile fetches)
            g_redraw();

            // com.freebsdx.GPS::LocationUpdate — payload is two little-endian
            // doubles (lat, lon). Drop/refresh a blue "GPS" pin and recentre the
            // map on the fix (this is what a "My Location" tap should do).
            xbus.OnBroadcast([=](const XBusMessage& msg) {
                if (std::strcmp(msg.service_name_, "com.freebsdx.GPS") != 0) return;
                if (msg.opcode_ != 0x01) return;   // LocationUpdate
                double lat, lon;
                std::memcpy(&lat, msg.payload_,     sizeof(lat));
                std::memcpy(&lon, msg.payload_ + 8, sizeof(lon));
                *gps_lat = lat;
                *gps_lon = lon;
                *gps_fix = true;
                // Add or move the blue GPS pin
                bool already_pinned = false;
                for (auto& p : g_pois) {
                    if (p.name == "GPS") { p.lat = lat; p.lon = lon;
                                          already_pinned = true; break; }
                }
                if (!already_pinned)
                    g_pois.push_back({lat, lon, "GPS", 60, 200, 80});
                *cur_lat = lat;   // recentre on the fix
                *cur_lon = lon;
                try { status_lbl->setText("My location"); } catch (...) {}
                try { g_redraw(); } catch (...) {}
            });

            // If launched with a query (e.g. "Maps.xapp 'Eiffel Tower'"), geocode it
            if (!preload_query.empty()) {
                // Simulate a search
                search_input->setText(preload_query);
                // Trigger the search button callback
                std::thread([=]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                    char url[512];
                    std::snprintf(url, sizeof(url), kNominatimUrlFmt,
                                  urlEncode(preload_query).c_str());
                    std::string body;
                    if (!httpGet(url, body)) return;
                    double rlat, rlon;
                    std::string rname;
                    if (!parseNominatim(body, rlat, rlon, rname)) return;
                    *cur_lat = rlat;
                    *cur_lon = rlon;
                    g_pois.push_back({rlat, rlon, rname, 80, 160, 220});
                    try {
                        status_lbl->setText(rname);
                        zoom_label->setText("z" + std::to_string(*cur_zoom));
                        g_redraw();
                    } catch (...) {}
                }).detach();
            }
        });

    // Leave without exit-time teardown: detached tile / search threads may be
    // mid-request inside libcurl/OpenSSL, and running their atexit cleanup
    // under them crashed the app as its window closed. Nothing here needs it.
    std::_Exit(rc);
}
