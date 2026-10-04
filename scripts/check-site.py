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
    if path != '404.html' and sum(a.get('aria-current') == 'page' for _, a in parser.elements) != 1:
        failures.append(f'{path}: expected one active navigation item')
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
for page in baseline['pages']:
    parser = parsers[page['path']]
    actual = normalize(' '.join(parser.main_text))
    for original in page['original_paragraphs'] + page['original_list_items']:
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
calendar = next(a['src'] for t, a in parsers['calendar.html'].elements if t == 'iframe')
if len(re.findall(r'(?:\?|&)src=', calendar)) != 6:
    failures.append('Calendar: original six calendar sources not preserved')
if not (ROOT / 'index.html').read_text().find('url=about.html') >= 0:
    failures.append('Index: original redirect changed')

if failures:
    print('\n'.join(failures))
    sys.exit(1)
print(f'PASS: {len(pages)} pages; {preserved} original content blocks; links, counts, structure, metadata, and legacy-script isolation.')
print('Limits: source checks only; external link availability, layout, and interactive behavior need browser review.')
