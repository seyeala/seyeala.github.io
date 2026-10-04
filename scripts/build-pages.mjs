import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pages = JSON.parse(readFileSync(path.join(root, 'content/pages.json'), 'utf8'));
const primary = [['about.html', 'Home'], ['research.html', 'Research'], ['laboratory.html', 'Laboratory'], ['publications.html', 'Publications'], ['team.html', 'Team']];
const secondary = [['service.html', 'Service'], ['opportunities.html', 'Opportunities'], ['calendar.html', 'Calendar'], ['outreach.html', 'Outreach'], ['chatbot.html', 'Chatbot'], ['News.html', 'News']];
const escape = value => value.replaceAll('&', '&amp;').replaceAll('"', '&quot;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const link = ([url, label], current) => `<a href="${url}"${url === current ? ' aria-current="page"' : ''}>${label}</a>`;

function header(current) {
  return `<a class="skip-link" href="#main-content">Skip to content</a>
  <header class="site-header">
    <div class="container header-inner">
      <a class="brand" href="about.html"><span class="brand-mark" aria-hidden="true">SA</span><span class="brand-name">Seyedhamidreza Alaie</span></a>
      <button class="nav-toggle" type="button" aria-expanded="false" aria-controls="primary-navigation" hidden>Menu</button>
      <nav class="site-nav" id="primary-navigation" aria-label="Primary navigation">
        ${primary.map(item => link(item, current)).join('\n        ')}
        <details class="more-nav"${secondary.some(([url]) => url === current) ? ' data-current="true"' : ''}><summary>More</summary><div class="more-links">${secondary.map(item => link(item, current)).join('')}</div></details>
      </nav>
    </div>
  </header>`;
}

function footer() {
  return `<footer class="site-footer"><div class="container">
    <div class="footer-inner"><div class="footer-name">Seyedhamidreza Alaie</div><ul class="footer-links"><li><a href="http://www.nmsu.edu/">NMSU Website</a></li><li><a href="https://mae.nmsu.edu">ME &amp; Aero Home</a></li><li><a href="mailto:alaie.unm@gmail.com">Contact the webmaster</a></li></ul></div>
    <div class="footer-secondary"><p>All copyright reserved.</p></div>
  </div></footer>`;
}

for (const page of pages) {
  const heading = page.home ? '' : `<div class="page-heading"><p class="eyebrow">${escape(page.eyebrow || 'Alaie Research')}</p><h1>${escape(page.title)}</h1>${page.subtitle ? `<p class="lead">${escape(page.subtitle)}</p>` : ''}</div>`;
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escape(page.title)} | Seyedhamidreza Alaie</title>
  <meta name="description" content="${escape(page.description)}">
  <link rel="canonical" href="https://seyeala.github.io/${page.path}">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
  <link rel="stylesheet" href="assets/css/site.css">
  <script src="assets/js/site.js" defer></script>
</head>
<body>
  ${header(page.path)}
  <main id="main-content" class="container" tabindex="-1">${heading}
    ${page.body}
  </main>
  ${footer()}
</body>
</html>
`;
  writeFileSync(path.join(root, page.path), html);
}
console.log(`Built ${pages.length} pages from shared templates; existing URLs preserved.`);
