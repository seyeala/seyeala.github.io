"""Dependency-free structural, link, and preservation checks (not browser QA)."""
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json
import re
import sys

ROOT = Path(__file__).resolve().parent.parent
VOID = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.elements = []
        self.text = []
        self.main_text = []
        self.stack = []
        self.errors = []
        self.ids = []
        self.main_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.elements.append((tag, attrs))
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        if tag == 'main':
            self.main_depth += 1
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == 'main':
            self.main_depth -= 1
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f'unbalanced closing tag: {tag}')
        else:
            self.stack.pop()

    def handle_data(self, data):
        self.text.append(data)
        if self.main_depth:
            self.main_text.append(data)


def plain(fragment):
    parser = Page()
    parser.feed(fragment)
    return ' '.join(parser.text)


def normalize(text):
    text = unescape(text).lower()
    text = re.sub(r'\(\s*web\s*\)', '', text)
    edits = {
        'simeltenous': 'simultaneous', 'piecwise': 'piecewise',
        'pusuing': 'pursuing', 'schemiatics': 'schematics', 'presetations': 'presentations',
        'undegraduate': 'undergraduate', 'undergraduatue': 'undergraduate',
        'confrence': 'conference', ' joint ': ' joined ',
        'is currently as systems engineer': 'is currently a systems engineer',
        'if received admission': 'if you have received admission',
        'dr. alaie research lab': "dr. alaie's research lab",
        'in github': 'on github', '(web)': '',
        'ph.d. student opening:': 'ph.d. student opening',
    }
    for before, after in edits.items():
        text = text.replace(before, after)
    return ''.join(ch for ch in text if ch.isalnum())


pages = json.loads((ROOT / 'content/pages.json').read_text())
baseline = json.loads((ROOT / 'scripts/content-baseline.json').read_text())
parsers = {}
failures = []
if (ROOT / 'CV.pdf').exists():
    failures.append('Privacy: CV.pdf must not be present in the website root')
for page in pages:
    path = page['path']
    html = (ROOT / path).read_text()
    parser = Page()
    parser.feed(html)
    parsers[path] = parser
    if not html.startswith('<!DOCTYPE html>'):
        failures.append(f'{path}: missing doctype')
    failures.extend(f'{path}: {error}' for error in parser.errors)
    if parser.stack:
        failures.append(f'{path}: unclosed tags {parser.stack}')
    if len(parser.ids) != len(set(parser.ids)):
        failures.append(f'{path}: duplicate IDs')
    for tag in ['header', 'nav', 'main', 'footer', 'h1', 'title']:
        if sum(t == tag for t, _ in parser.elements) != 1:
            failures.append(f'{path}: expected one {tag}')
    if not any(t == 'meta' and a.get('name') == 'viewport' for t, a in parser.elements):
        failures.append(f'{path}: missing viewport')
    if not any(t == 'meta' and a.get('name') == 'description' and a.get('content') for t, a in parser.elements):
        failures.append(f'{path}: missing description')
    # The original standalone Machine Learning page has no incoming menu link.
    if path not in ('404.html', 'ml.html') and sum(a.get('aria-current') == 'page' for _, a in parser.elements) != 1:
        failures.append(f'{path}: expected one active navigation item')
    nav = Page()
    nav.feed(re.search(r'<nav\b.*?</nav>', html, re.S).group())
    nav_links = [a.get('href') for t, a in nav.elements if t == 'a']
    if nav_links.count('chatbot.html') != 1:
        failures.append(f'{path}: original Chatbot navigation link missing or duplicated')
    if not (nav_links.index('outreach.html') < nav_links.index('chatbot.html') < nav_links.index('News.html')):
        failures.append(f'{path}: original Chatbot navigation order changed')
    footer = Page()
    footer.feed(re.search(r'<footer\b.*?</footer>', html, re.S).group())
    if any(a.get('href') in ('ml.html', 'chatbot.html', 'CV.pdf') for t, a in footer.elements if t == 'a'):
        failures.append(f'{path}: added secondary footer links remain')
    if 'All copyright reserved.' not in ' '.join(footer.text):
        failures.append(f'{path}: original copyright wording missing')
    if 'Seyedhamidreza Alaie' in ' '.join(footer.text):
        failures.append(f'{path}: repeated personal name remains in footer')
    footer_links = [a.get('href') for t, a in footer.elements if t == 'a']
    if footer_links.count('mailto:alaie@nmsu.edu') != 1:
        failures.append(f'{path}: webmaster must use the owner-approved academic email')
    if 'mailto:alaie.unm@gmail.com' in html:
        failures.append(f'{path}: old webmaster address remains')
    chatbot_scripts = sum(t == 'script' and a.get('src') == 'assets/js/chatbot.js' for t, a in parser.elements)
    if chatbot_scripts != (1 if path == 'chatbot.html' else 0):
        failures.append(f'{path}: chatbot script must be isolated to the Chatbot page')
    for tag, attrs in parser.elements:
        if tag == 'table':
            failures.append(f'{path}: layout table remains')
        if tag == 'img' and not attrs.get('alt'):
            failures.append(f'{path}: missing image alt text')
        if tag == 'iframe' and not attrs.get('title'):
            failures.append(f'{path}: missing iframe title')
        if tag == 'script' and attrs.get('src') == 'script.js':
            failures.append(f'{path}: legacy webcam script coupling')
        for key in ['href', 'src']:
            value = attrs.get(key)
            if not value:
                continue
            url = urlsplit(unescape(value))
            if unquote(url.path).lower().endswith('/cv.pdf') or unquote(url.path).lower() == 'cv.pdf':
                failures.append(f'{path}: CV link or embedded file remains')
            if url.scheme in ('http', 'https', 'data', 'tel'):
                continue
            if url.scheme == 'mailto':
                if not re.fullmatch(r'[^<>\s]+@[^<>\s]+\.[^<>\s]+', url.path):
                    failures.append(f'{path}: malformed email link')
                continue
            local = ROOT / unquote(url.path)
            if url.path and not local.exists() and url.path not in ('DSC01028.jpg', 'Lab_Layout_V02.jpg'):
                failures.append(f'{path}: missing local target {value}')
            if not url.path and url.fragment and url.fragment not in parser.ids:
                failures.append(f'{path}: missing fragment {value}')

preserved = 0
owner_replaced = 0
for page in baseline['pages']:
    parser = parsers[page['path']]
    actual = normalize(' '.join(parser.main_text))
    for original in page['original_paragraphs'] + page['original_list_items']:
        # Owner explicitly requested a chat frontend in place of this placeholder.
        if page['path'] == 'chatbot.html' and plain(original).strip() == 'To be updated':
            owner_replaced += 1
            continue
        expected = normalize(plain(original))
        if expected and expected not in actual:
            failures.append(f"{page['path']}: original material absent: {plain(original)[:130]}")
        else:
            preserved += 1
    actual_links = {a['href'] for tag, a in parser.elements if tag == 'a' and a.get('href')}
    for link in page['original_links']:
        if unescape(link) not in actual_links:
            failures.append(f"{page['path']}: original link missing: {link}")

pub = parsers['publications.html']
if sum(t == 'li' for t, a in pub.elements) - 3 != 12:
    failures.append('Publications: expected 12 citation placements plus 3 footer items')
if sum(t == 'article' for t, a in parsers['team.html'].elements) != 5:
    failures.append('Team: expected five members')
if sum(t == 'li' for t, a in parsers['laboratory.html'].elements) - 3 != 11:
    failures.append('Laboratory: expected 11 equipment entries')
if sum(t == 'time' for t, a in parsers['News.html'].elements) != 3:
    failures.append('News: expected three dated entries')
# Owner explicitly requested replacing the original personal Google sources.
calendar = parsers['calendar.html']
calendar_links = {a.get('href') for t, a in calendar.elements if t == 'a'}
for url in ('https://records.nmsu.edu/academic-calendar/', 'https://crimsonconnection.nmsu.edu/events'):
    if url not in calendar_links:
        failures.append(f'Calendar: official source missing: {url}')
if any(t == 'iframe' for t, a in calendar.elements):
    failures.append('Calendar: unverified embedded calendar remains')
if 'calendar.google.com' in (ROOT / 'calendar.html').read_text():
    failures.append('Calendar: personal Google calendar source remains')
chatbot = parsers['chatbot.html']
for required_id in ('chat-form', 'chat-input', 'chat-submit', 'chat-thread', 'chat-feedback', 'chat-reset'):
    if required_id not in chatbot.ids:
        failures.append(f'Chatbot: required frontend control missing: {required_id}')
if not any(t == 'textarea' and a.get('id') == 'chat-input' and a.get('maxlength') == '2000' for t, a in chatbot.elements):
    failures.append('Chatbot: expected limited, labeled message composer')
if not any(t == 'label' and a.get('for') == 'chat-input' for t, a in chatbot.elements):
    failures.append('Chatbot: composer label missing')
if 'Not connected' not in ' '.join(chatbot.text):
    failures.append('Chatbot: initial backend state must be transparent')
if not (ROOT / 'index.html').read_text().find('url=about.html') >= 0:
    failures.append('Index: original redirect changed')

if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(f'PASS: {len(pages)} pages; {preserved} preserved original content blocks; {owner_replaced} owner-approved placeholder replacement; links, counts, structure, metadata, and script isolation.')
print('Limits: source checks only; external link availability, layout, and interactive behavior need browser review.')
