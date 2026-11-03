const CACHE_NAME = "aura-cache-v2";
const APP_SHELL = ["/", "/static/manifest.json", "/static/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys.filter((key) => key.startsWith("aura-cache-") && key !== CACHE_NAME)
          .map((key) => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  // Never cache chat/API responses or other dynamic endpoints.
  if (url.pathname.startsWith("/api/")) return;

  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request).then((response) => response)
        .catch(async () => (await caches.match("/")) || Response.error())
    );
    return;
  }

  // Only cache the app shell and static assets; do not cache arbitrary GETs.
  if (url.pathname !== "/" && !url.pathname.startsWith("/static/")) return;
  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(request);
    if (cached) return cached;
    try {
      const response = await fetch(request);
      if (response.ok && response.type === "basic") {
        await cache.put(request, response.clone());
      }
      return response;
    } catch {
      return (await caches.match("/")) || Response.error();
    }
  })());
});
