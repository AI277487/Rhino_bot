/*
 * RhinoBot service worker — intentionally minimal.
 *
 * Its ONLY job is to exist, so the browser treats the site as an installable
 * PWA (home-screen icon, standalone window). It deliberately does NOT cache
 * HTML, JS, or API responses: every request goes straight to the network.
 *
 * Why no caching: RhinoBot needs the server for every answer (retrieval +
 * model), so offline caching buys nothing — and an over-eager cache is the
 * classic PWA footgun where users keep seeing an OLD version after you deploy.
 * During active development that would be a constant headache. Network-only
 * keeps every load fresh.
 */

self.addEventListener("install", (event) => {
  self.skipWaiting();               // activate this SW immediately
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  // pass through to network, no cache. If offline, the request simply fails
  // (correct behaviour for a tool that can't work without the server).
  return;
});
