// ── Powerball Analytics Hub — Service Worker ──────────────────
const CACHE_NAME    = 'pb-analytics-v1';
const STATIC_ASSETS = [
  '/',
  '/generated_numbers_history',
  '/check-my-numbers',
  '/hot_cold_numbers',
  '/frequency_analysis',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  // External fonts & libs (cached on first load)
  'https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap',
  'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css',
  'https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js'
];

// ── Install: cache core assets ─────────────────────────────────
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Caching core assets');
      // Cache what we can — don't fail install if some external assets are unavailable
      return Promise.allSettled(
        STATIC_ASSETS.map(url =>
          cache.add(url).catch(err => console.warn('[SW] Could not cache:', url, err))
        )
      );
    }).then(() => self.skipWaiting())
  );
});

// ── Activate: clean up old caches ─────────────────────────────
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(key => key !== CACHE_NAME)
            .map(key => {
              console.log('[SW] Deleting old cache:', key);
              return caches.delete(key);
            })
      )
    ).then(() => self.clients.claim())
  );
});

// ── Fetch: network-first for API, cache-first for static ───────
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Always go to network for API calls and Supabase
  if (
    url.pathname.startsWith('/api/') ||
    url.pathname.startsWith('/analyze') ||
    url.pathname.startsWith('/check-numbers') ||
    url.hostname.includes('supabase') ||
    event.request.method !== 'GET'
  ) {
    event.respondWith(
      fetch(event.request).catch(() => {
        // If offline and API call fails, return a friendly JSON error
        return new Response(
          JSON.stringify({ error: 'You are offline. Please check your connection.' }),
          { headers: { 'Content-Type': 'application/json' } }
        );
      })
    );
    return;
  }

  // Cache-first for static assets (fonts, CSS, JS, icons)
  if (
    url.pathname.startsWith('/static/') ||
    url.hostname.includes('googleapis.com') ||
    url.hostname.includes('jsdelivr.net') ||
    url.hostname.includes('gstatic.com')
  ) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        return cached || fetch(event.request).then(response => {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          return response;
        });
      })
    );
    return;
  }

  // Network-first for HTML pages — show cached version if offline
  event.respondWith(
    fetch(event.request)
      .then(response => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        return response;
      })
      .catch(() => {
        return caches.match(event.request).then(cached => {
          if (cached) return cached;
          // Fallback to home page if specific page not cached
          return caches.match('/');
        });
      })
  );
});
