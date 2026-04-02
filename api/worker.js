// Yubimoji Dojo API Worker
// Stores/serves calibration data via Cloudflare KV
//
// GET  /calibration          — read calibration data (public)
// PUT  /calibration          — write calibration data (admin only, requires ?key=ADMIN_KEY)
// GET  /calibration/:char    — read single char
// PUT  /calibration/:char    — write single char (admin only)
// OPTIONS *                  — CORS preflight

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, PUT, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
};

function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json", ...CORS_HEADERS },
  });
}

function isAdmin(request, env) {
  const url = new URL(request.url);
  return url.searchParams.get("key") === env.ADMIN_KEY;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // GET /calibration — return all calibration data
    if (path === "/calibration" && request.method === "GET") {
      const data = await env.CALIBRATION.get("all", { type: "json" });
      return json(data || {});
    }

    // PUT /calibration — replace all calibration data (admin)
    if (path === "/calibration" && request.method === "PUT") {
      if (!isAdmin(request, env)) {
        return json({ error: "unauthorized" }, 403);
      }
      const body = await request.json();
      await env.CALIBRATION.put("all", JSON.stringify(body));
      return json({ ok: true, signs: Object.keys(body).length });
    }

    // GET /calibration/:char — single char
    const charMatch = path.match(/^\/calibration\/(.+)$/);
    if (charMatch && request.method === "GET") {
      const char = decodeURIComponent(charMatch[1]);
      const data = await env.CALIBRATION.get("all", { type: "json" });
      if (data && data[char]) {
        return json(data[char]);
      }
      return json({ error: "not found" }, 404);
    }

    // PUT /calibration/:char — update single char (admin)
    if (charMatch && request.method === "PUT") {
      if (!isAdmin(request, env)) {
        return json({ error: "unauthorized" }, 403);
      }
      const char = decodeURIComponent(charMatch[1]);
      const body = await request.json();
      const data = (await env.CALIBRATION.get("all", { type: "json" })) || {};
      data[char] = body;
      await env.CALIBRATION.put("all", JSON.stringify(data));
      return json({ ok: true, char });
    }

    return json({ error: "not found" }, 404);
  },
};
