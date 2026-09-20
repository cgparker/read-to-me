# md2audio

Turn a markdown file into spoken audio on your Mac, using the same Kokoro voice as the Read to Me page. Everything runs locally: no account, no API, no per-use cost.

The output lands next to the input file with the same name:

    md2audio ~/Documents/Weekly Plan.md   ->   ~/Documents/Weekly Plan.zip

The zip contains one MP3 per section, tagged as an album, ready for BookPlayer or the Music app.

## Setup (once)

```sh
node -v                      # needs Node 18 or newer; brew install node if missing
mkdir -p ~/bin/read-to-me    # put md2audio.mjs and package.json here
cd ~/bin/read-to-me
npm install
echo 'alias md2audio="node ~/bin/read-to-me/md2audio.mjs"' >> ~/.zshrc
```

Open a new terminal, or run `source ~/.zshrc`, and the `md2audio` command is available.

The first conversion downloads the voice model (about 92 MB) into `~/.cache/read-to-me`. Later runs use the cached copy and work offline.

## Use

```sh
md2audio "~/Library/Mobile Documents/com~apple~CloudDocs/Notes/Weekly Plan.md"
md2audio notes.md --single                 # one MP3 with chapter markers
md2audio notes.md --voice af_bella --speed 1.1
```

Options:

| Option | What it does |
|---|---|
| `--single` | One MP3 with chapter markers instead of a zip of section tracks |
| `--voice <id>` | Default `af_heart`. Also `af_bella`, `af_nicole`, `bf_emma`, `af_aoede`, `af_kore`, `af_sarah`, `am_michael`, `am_fenrir`, `am_puck`, `bm_george`, `bm_fable` |
| `--speed <n>` | 0.5 to 2, default 1 |
| `--read-code` | Read fenced code blocks aloud (skipped by default) |
| `--dtype fp32` | Higher precision model (325 MB) if `q8` sounds worse than the web page |
| `--out <path>` | Write somewhere other than next to the input |
| `--quiet` | Print only the output path |

Progress goes to the terminal; the final line is the output path, so it works in a pipeline.

## Sections

Sections come from your headings: the script uses the highest heading level that appears at least twice. In a document with one `#` title and several `##` headings, each `##` becomes a track, and anything before the first one becomes the opening track.

## iCloud

Save the markdown in iCloud Drive and the zip appears beside it, which syncs to your iPhone. If a run fails because iCloud hasn't downloaded the file locally, run `brctl download "path/to/file.md"` first.
