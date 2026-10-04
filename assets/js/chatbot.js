'use strict';

(() => {
  if (!document.querySelector('[data-chat-app]')) return;
  const input = document.getElementById('chat-input');
  const form = document.getElementById('chat-form');
  const submit = document.getElementById('chat-submit');
  const counter = document.getElementById('chat-counter');
  const connection = document.getElementById('chat-connection');
  const feedback = document.getElementById('chat-feedback');
  const privacy = document.getElementById('chat-privacy');
  const welcome = document.getElementById('chat-welcome');
  const welcomeCopy = document.getElementById('chat-welcome-copy');
  const thread = document.getElementById('chat-thread');
  const resetButton = document.getElementById('chat-reset');
  let connector = null;
  let history = [];
  let busy = false;
  let generation = 0;
  let pending = null;
  const limit = 2000;

  function updateControls() {
    const length = input.value.length;
    counter.textContent = `${length} / ${limit} · Ctrl or Cmd + Enter`;
    submit.disabled = busy || !input.value.trim() || length > limit;
    submit.textContent = busy ? 'Waiting for reply…' : connector ? 'Send message' : 'Preview message';
    input.disabled = busy;
    form.setAttribute('aria-busy', String(busy));
  }

  function resetConversation(shouldFocus = true) {
    generation += 1;
    pending?.abort();
    pending = null;
    busy = false;
    history = [];
    thread.replaceChildren();
    thread.hidden = true;
    welcome.hidden = false;
    input.value = '';
    feedback.textContent = connector
      ? 'Assistant connected. New messages go to the configured backend.'
      : 'Interface preview only. No messages are sent or saved by this page.';
    updateControls();
    if (shouldFocus) input.focus();
  }

  function setConnection() {
    connection.textContent = connector ? 'Connected' : 'Not connected';
    connection.dataset.state = connector ? 'connected' : 'disconnected';
    welcomeCopy.textContent = connector
      ? 'Ask a question using the composer below. Answers come from the configured assistant and depend on its approved knowledge and capabilities.'
      : 'This front end is ready for the custom assistant being prepared. For now, you can preview the interface; no messages are sent and no AI replies are generated.';
    privacy.textContent = connector
      ? 'Do not include confidential or personal information. Messages are processed by the configured assistant.'
      : 'Drafts stay in temporary page memory and clear when you reload or start a new chat.';
  }

  function appendMessage(role, text) {
    welcome.hidden = true;
    thread.hidden = false;
    const item = document.createElement('article');
    item.className = `chat-message chat-message-${role}`;
    const label = document.createElement('span');
    label.className = 'chat-message-label';
    label.textContent = role === 'assistant' ? 'Assistant' : connector ? 'You' : 'Your draft · not sent';
    const content = document.createElement('p');
    content.textContent = text;
    item.append(label, content);
    thread.append(item);
    if (thread.children.length > 30) thread.firstElementChild.remove();
    thread.scrollTop = thread.scrollHeight;
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const message = input.value.trim();
    if (busy || !message) return;
    if (input.value.length > limit) {
      feedback.textContent = `Please keep your question within ${limit} characters.`;
      return;
    }
    appendMessage('user', message);
    input.value = '';
    if (!connector) {
      feedback.textContent = 'Draft previewed locally. The assistant is not connected, so nothing was sent and no reply was generated.';
      updateControls();
      return;
    }

    history.push({ role: 'user', content: message });
    history = history.slice(-30);
    const requestGeneration = generation;
    const controller = new AbortController();
    pending = controller;
    busy = true;
    feedback.textContent = 'Waiting for the connected assistant…';
    updateControls();
    try {
      const response = await connector.send({
        message,
        history: history.map(item => ({ ...item })),
        signal: controller.signal,
      });
      if (requestGeneration !== generation) return;
      if (!response || typeof response.text !== 'string' || !response.text.trim()) {
        throw new Error('Invalid assistant response');
      }
      appendMessage('assistant', response.text);
      history.push({ role: 'assistant', content: response.text });
      history = history.slice(-30);
      feedback.textContent = 'Reply received from the connected assistant.';
    } catch {
      if (requestGeneration === generation) {
        feedback.textContent = 'Unable to get a reply. You can try again or start a new chat.';
      }
    } finally {
      if (requestGeneration === generation) {
        busy = false;
        pending = null;
        updateControls();
      }
    }
  });

  input.addEventListener('input', updateControls);
  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey) && !event.isComposing) {
      event.preventDefault();
      if (!submit.disabled) form.requestSubmit();
    }
  });
  document.querySelectorAll('[data-chat-prompt]').forEach(button => {
    button.addEventListener('click', () => {
      if (busy) return;
      input.value = button.dataset.chatPrompt;
      updateControls();
      input.focus();
    });
  });
  resetButton.addEventListener('click', () => resetConversation());

  // No backend, credentials, network calls, or persistent storage are installed.
  // A later approved integration supplies only this browser-safe connector.
  window.AlaieChatSkin = Object.freeze({
    connect(adapter) {
      if (!adapter || typeof adapter.send !== 'function') throw new TypeError('A send function is required');
      connector = adapter;
      setConnection();
      resetConversation(false); // Never forward drafts from preview mode.
    },
    disconnect() {
      connector = null;
      setConnection();
      resetConversation(false);
    },
    clear() { resetConversation(); },
  });
  setConnection();
  updateControls();
})();
