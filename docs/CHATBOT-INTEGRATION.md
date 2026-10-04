# Chatbot front-end handoff

The Chatbot page is a dependency-free front end. It is deliberately disconnected: suggestions fill the composer, Preview message displays a local draft, and New chat clears memory. There are no network calls, analytics, uploads, browser storage, external scripts, or fabricated AI replies. No credentials are embedded. Messages are retained only in the current page's memory.

The owner will provide the customized assistant later. A ChatGPT GPT sharing URL is a link-out destination, not the backend contract below. For actual on-site answers, supply an approved server-side integration that recreates the desired instructions, approved knowledge, and tools. Review each skill before enabling it. Do not provide private calendar sources or the removed CV as bot knowledge.

## Connector contract

After `assets/js/chatbot.js` loads, a separately approved integration can call:

```js
window.AlaieChatSkin.connect({
  async send({ message, history, signal }) {
    // Call your approved backend here, never the OpenAI API with a browser key.
    // history is an array of { role, content } and includes the latest message.
    // Honor signal to cancel requests when the visitor starts a new chat.
    return { text: 'The actual response returned by your backend' };
  }
});
```

This is an interface example, not an installed backend. The UI switches from Preview message to Send message only after an adapter is explicitly registered. Connecting clears all unsent preview drafts. Error details are not shown to visitors; stale responses are ignored after reset. `window.AlaieChatSkin.disconnect()` removes the adapter and clears the conversation. `window.AlaieChatSkin.clear()` clears the current conversation without disconnecting.

Before enabling real messages: configure server-side credentials, permitted knowledge/tools, authentication as needed, rate and spend limits, and a data/privacy notice. GitHub Pages cannot run a server-side secret-bearing backend; use a separate service or an approved Sites server implementation. No backend or external account was created in this revision.

Relevant official guidance: [custom server-side ChatKit integration](https://developers.openai.com/api/docs/guides/custom-chatkit), [API skills](https://developers.openai.com/api/docs/guides/tools-skills).

## Checks

Run `node scripts/check-chatbot.mjs` for DOM-mock behavior tests and `python scripts/check-site.py` for structural/privacy checks. These are not rendered-browser or live-backend tests.
