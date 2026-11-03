# filmcrab.in

FilmCrab marketing site: AI agents, custom software, production services and a producer network for film, starting with Malayalam cinema.

A single static page (`index.html`) with hash routing. No build step.

## Status: preview build

- `index.html` carries `noindex` and `robots.txt` disallows all crawling, so search engines ignore it.
- The page still shows its draft-notes panel and `[CONFIRM]` placeholders. These are intentional until the items below are settled.
- The demo form builds a copyable request only. It sends nothing yet.

## Before launch

1. Confirm who delivers the script doctor and production unit support, the areas, and the prices.
2. Get legal advice on investor introductions before offering or advertising them.
3. Supply the demo email (and WhatsApp number if wanted), then connect the form.
4. Remove the draft-notes panel and any remaining `[CONFIRM]` text.
5. Remove the `noindex` meta tag in `index.html`, delete `robots.txt` (or allow crawling), and add real page URLs to the schema.

## Deploy

Pushes to `main` deploy through Hostinger's Git integration.
