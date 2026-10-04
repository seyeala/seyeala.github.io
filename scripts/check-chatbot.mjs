import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import test from 'node:test';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = readFileSync(path.join(root, 'assets/js/chatbot.js'), 'utf8');
const html = readFileSync(path.join(root, 'chatbot.html'), 'utf8');

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.listeners = new Map();
    this.dataset = {};
    this.attributes = {};
    this.children = [];
    this.value = '';
    this.textContent = '';
    this.hidden = false;
    this.disabled = false;
  }
  addEventListener(type, callback) { this.listeners.set(type, callback); }
  async emit(type, event = {}) {
    return this.listeners.get(type)?.({ preventDefault() {}, ...event });
  }
  setAttribute(key, value) { this.attributes[key] = value; }
  focus() { this.focused = true; }
  append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
  replaceChildren() { this.children = []; }
  remove() { this.parent.children = this.parent.children.filter(item => item !== this); }
  get firstElementChild() { return this.children[0]; }
  get scrollHeight() { return this.children.length; }
  requestSubmit() { this.submission = this.emit('submit'); }
}

function setup() {
  const ids = Object.fromEntries([...html.matchAll(/id="(chat-[^"]+)"/g)].map(match => [match[1], new Element()]));
  const prompts = [...html.matchAll(/data-chat-prompt="([^"]+)"/g)].map(match => {
    const button = new Element('button');
    button.dataset.chatPrompt = match[1];
    return button;
  });
  const created = [];
  const document = {
    querySelector: () => new Element(),
    querySelectorAll: () => prompts,
    getElementById: id => ids[id],
    createElement(tag) { created.push(tag); return new Element(tag); },
  };
  const window = {};
  vm.runInNewContext(source, { document, window, AbortController });
  return { ids, prompts, window, created };
}

test('offline mode is explicit, empty submission disabled, suggestions populate composer', async () => {
  const { ids, prompts } = setup();
  assert.equal(ids['chat-connection'].textContent, 'Not connected');
  assert.equal(ids['chat-submit'].disabled, true);
  await prompts[0].emit('click');
  assert.equal(ids['chat-input'].value, 'What are the main research areas?');
  assert.equal(ids['chat-submit'].disabled, false);
  assert.equal(ids['chat-input'].focused, true);
  await ids['chat-form'].emit('submit');
  assert.equal(ids['chat-thread'].children.length, 1);
  assert.equal(ids['chat-thread'].children[0].children[0].textContent, 'Your draft · not sent');
  assert.match(ids['chat-feedback'].textContent, /nothing was sent/);
});

test('visitor input remains literal text and no network/storage implementation is present', async () => {
  const { ids, created } = setup();
  const text = '<img src=x onerror="alert(1)">';
  ids['chat-input'].value = text;
  await ids['chat-input'].emit('input');
  await ids['chat-form'].emit('submit');
  assert.equal(ids['chat-thread'].children[0].children[1].textContent, text);
  assert.deepEqual(created, ['article', 'span', 'p']);
  assert.doesNotMatch(source, /\b(fetch|XMLHttpRequest|WebSocket|localStorage|sessionStorage)\b/);
});

test('new chat clears memory, restores empty state and focuses composer', async () => {
  const { ids } = setup();
  ids['chat-input'].value = 'Draft';
  await ids['chat-form'].emit('submit');
  await ids['chat-reset'].emit('click');
  assert.equal(ids['chat-thread'].children.length, 0);
  assert.equal(ids['chat-thread'].hidden, true);
  assert.equal(ids['chat-welcome'].hidden, false);
  assert.equal(ids['chat-input'].focused, true);
});

test('Enter stays multiline; Ctrl/Cmd+Enter submits but IME composition does not', async () => {
  const { ids } = setup();
  ids['chat-input'].value = 'Question';
  await ids['chat-input'].emit('input');
  await ids['chat-input'].emit('keydown', { key: 'Enter', ctrlKey: false, metaKey: false });
  assert.equal(ids['chat-thread'].children.length, 0);
  await ids['chat-input'].emit('keydown', { key: 'Enter', ctrlKey: true, isComposing: true });
  assert.equal(ids['chat-thread'].children.length, 0);
  await ids['chat-input'].emit('keydown', { key: 'Enter', metaKey: true });
  await ids['chat-form'].submission;
  assert.equal(ids['chat-thread'].children.length, 1);
});

test('overlength and blank questions are not added', async () => {
  const { ids } = setup();
  ids['chat-input'].value = ' '.repeat(8);
  await ids['chat-form'].emit('submit');
  ids['chat-input'].value = 'x'.repeat(2001);
  await ids['chat-input'].emit('input');
  assert.equal(ids['chat-submit'].disabled, true);
  await ids['chat-form'].emit('submit');
  assert.equal(ids['chat-thread'].children.length, 0);
  assert.match(ids['chat-feedback'].textContent, /2000/);
});

test('future connector clears preview drafts and renders only actual adapter replies', async () => {
  const { ids, window } = setup();
  ids['chat-input'].value = 'Private local preview';
  await ids['chat-form'].emit('submit');
  let request;
  window.AlaieChatSkin.connect({ async send(payload) { request = payload; return { text: 'Adapter reply' }; } });
  assert.equal(ids['chat-thread'].children.length, 0);
  assert.equal(ids['chat-connection'].textContent, 'Connected');
  ids['chat-input'].value = 'New question';
  await ids['chat-form'].emit('submit');
  assert.equal(request.message, 'New question');
  assert.equal(request.history.length, 1);
  assert.equal(request.history[0].content, 'New question');
  assert.equal(ids['chat-thread'].children[1].children[1].textContent, 'Adapter reply');
  window.AlaieChatSkin.disconnect();
  assert.equal(ids['chat-connection'].textContent, 'Not connected');
  assert.equal(ids['chat-thread'].children.length, 0);
});

test('clearing a pending request aborts it and ignores stale replies', async () => {
  const { ids, window } = setup();
  let finish, signal;
  window.AlaieChatSkin.connect({ send(payload) { signal = payload.signal; return new Promise(resolve => { finish = resolve; }); } });
  ids['chat-input'].value = 'Question';
  const pending = ids['chat-form'].emit('submit');
  assert.equal(ids['chat-input'].disabled, true);
  window.AlaieChatSkin.clear();
  assert.equal(signal.aborted, true);
  finish({ text: 'Old reply' });
  await pending;
  assert.equal(ids['chat-thread'].children.length, 0);
  assert.equal(ids['chat-input'].disabled, false);
});

test('adapter failures do not expose internal errors or fabricate replies', async () => {
  const { ids, window } = setup();
  window.AlaieChatSkin.connect({ async send() { throw new Error('secret backend details'); } });
  ids['chat-input'].value = 'Question';
  await ids['chat-form'].emit('submit');
  assert.equal(ids['chat-thread'].children.length, 1);
  assert.match(ids['chat-feedback'].textContent, /Unable to get a reply/);
  assert.doesNotMatch(ids['chat-feedback'].textContent, /secret/);
  assert.equal(ids['chat-form'].attributes['aria-busy'], 'false');
});
