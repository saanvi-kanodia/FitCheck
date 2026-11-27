// FitCheck service worker: makes the site installable, quick to open and usable offline.
// Served from /sw.js (scope "/") by app.py, which fills in __VERSION__ from the page and this file,
// so every deploy installs a fresh copy and clears the old saved files.
const VERSION = '__VERSION__';
const SHELL = `fitcheck-shell-${VERSION}`;   // the page, icons and site images (replaced on each deploy)
const MODELS = 'fitcheck-models-v1';         // MediaPipe code, wasm and the pose model (large, kept across deploys)
const FONTS = 'fitcheck-fonts-v1';

const PRECACHE = [
  '/', '/manifest.webmanifest',
  '/static/favicon.svg', '/static/icon-192.png', '/static/icon-512.png', '/static/hero-curlups-poster.webp',
  ...['pushup', 'squats', 'curlups', 'plank', 'jumpingjack', 'verticaljump', 'shuttlerun', 'sitreach', 'backstretch']
    .map(k => `/static/sil-${k}.png`),
];

self.addEventListener('install', event => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL);
    // One missing file must not stop the app from installing.
    await Promise.allSettled(PRECACHE.map(url => cache.add(new Request(url, { cache: 'reload' }))));
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    const keep = [SHELL, MODELS, FONTS];
    for (const key of await caches.keys()) if (key.startsWith('fitcheck-') && !keep.includes(key)) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', event => {
  const req = event.request;
  // Uploads and other writes, and video seeking (range requests), always go straight to the network.
  if (req.method !== 'GET' || req.headers.has('range')) return;
  const url = new URL(req.url);

  if (url.origin === self.location.origin) {
    // Analysis progress, results, challenge boards and videos are live data: never served from a cache.
    if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/jobs/') || url.pathname.endsWith('.mp4') || url.pathname === '/sw.js') return;
    if (req.mode === 'navigate') { event.respondWith(pageNetworkFirst(req)); return; }
    if (url.pathname.startsWith('/static/') || url.pathname === '/manifest.webmanifest') { event.respondWith(staleWhileRevalidate(req, SHELL)); return; }
    return;
  }
  // Live mode's pose tracking: cached after first use, so it starts fast next time and works offline.
  if (url.hostname === 'cdn.jsdelivr.net' || url.hostname === 'storage.googleapis.com') { event.respondWith(cacheFirst(req, MODELS)); return; }
  if (url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com') { event.respondWith(staleWhileRevalidate(req, FONTS)); return; }
});

// The page: fresh from the network when online (so updates show straight away), the saved copy when offline.
async function pageNetworkFirst(req) {
  const cache = await caches.open(SHELL);
  try {
    const res = await fetch(req);
    if (res.ok) cache.put('/', res.clone());
    return res;
  } catch (e) {
    return (await cache.match('/')) || new Response(
      '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FitCheck</title>' +
      '<body style="margin:0;display:grid;place-items:center;min-height:100vh;background:#1e1f1e;color:#eef2ec;font:16px system-ui;text-align:center">' +
      '<div><p style="font-size:22px;font-weight:700">You\'re offline</p><p>Connect to the internet and try again.</p></div>',
      { headers: { 'Content-Type': 'text/html; charset=utf-8' } });
  }
}

async function staleWhileRevalidate(req, name) {
  const cache = await caches.open(name);
  const cached = await cache.match(req);
  const network = fetch(req).then(res => {
    if (res.ok || res.type === 'opaque') cache.put(req, res.clone());
    return res;
  }).catch(() => cached);
  return cached || network;
}

async function cacheFirst(req, name) {
  const cache = await caches.open(name);
  const cached = await cache.match(req);
  if (cached) return cached;
  const res = await fetch(req);
  if (res.ok) cache.put(req, res.clone());
  return res;
}
