# Read to Me

Paste markdown and hear it read aloud by Kokoro, an open source speech model that runs inside your browser. There's no account and no API, and your text never leaves your computer.

Live page: https://cgparker.github.io/read-to-me/

## Using it

- Open the page in Chrome or Edge on a computer. The first visit downloads the voice model (about 325 MB). Later visits load it from the browser's cache.
- Paste markdown or drop in a .md file. Click any paragraph to jump there. Space plays and pauses, and the arrow keys skip between paragraphs.
- Save as MP3 offers one track per section (a zip for the Music app) or one file with chapter markers.

## Updating

Replace `index.html` with the new version and commit. GitHub Pages republishes in a minute or two.

## Credits

Voice: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache 2.0) through [kokoro-js](https://www.npmjs.com/package/kokoro-js). Markdown: [marked](https://marked.js.org). Sanitizing: [DOMPurify](https://github.com/cure53/DOMPurify). MP3 encoding: [lamejs](https://github.com/zhuker/lamejs) (LGPL).
