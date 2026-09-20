#!/usr/bin/env node
/**
 * md2audio: turn a markdown file into audio using the Kokoro voice model.
 *
 *   node md2audio.mjs notes.md                 -> notes.zip (one MP3 per section)
 *   node md2audio.mjs notes.md --single        -> notes.mp3 (one file with chapters)
 *   node md2audio.mjs notes.md --voice af_bella --speed 1.1
 *
 * Output lands next to the input file with the same name.
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import { marked } from "marked";
import { env } from "@huggingface/transformers";
import { KokoroTTS } from "kokoro-js";
import lameModule from "@breezystack/lamejs";

const lame = lameModule.default || lameModule;

const MODEL_ID = "onnx-community/Kokoro-82M-v1.0-ONNX";
const VOICE_IDS = [
  "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
  "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
  "am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_alice", "bf_emma",
  "bf_isabella", "bf_lily", "bm_daniel", "bm_fable", "bm_george", "bm_lewis"
];
const CHUNK_MAX = 260;
const MERGE_BELOW = 30;
const SEG_MAX = 320;
const BITRATE = 64;

/* Keep the downloaded model in one place no matter where the script runs. */
env.cacheDir = path.join(os.homedir(), ".cache", "read-to-me");

/* ---------- arguments ---------- */
function parseArgs(argv) {
  const opts = { voice: "af_heart", speed: 1, single: false, readCode: false, dtype: "q8", out: null, quiet: false };
  const rest = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--single") opts.single = true;
    else if (a === "--read-code") opts.readCode = true;
    else if (a === "--quiet" || a === "-q") opts.quiet = true;
    else if (a === "--voice") opts.voice = argv[++i];
    else if (a === "--speed") opts.speed = Number(argv[++i]) || 1;
    else if (a === "--dtype") opts.dtype = argv[++i];
    else if (a === "--out") opts.out = argv[++i];
    else if (a === "--help" || a === "-h") opts.help = true;
    else rest.push(a);
  }
  opts.input = rest[0];
  return opts;
}

const HELP = `md2audio: markdown to spoken audio, using Kokoro locally

Usage:
  md2audio <file.md> [options]

Options:
  --single        One MP3 with chapter markers instead of a zip of section tracks
  --voice <id>    Voice id (default af_heart). Others: af_bella, af_nicole,
                  bf_emma, af_aoede, af_kore, af_sarah, am_michael, am_fenrir,
                  am_puck, bm_george, bm_fable
  --speed <n>     Speaking speed, 0.5 to 2 (default 1)
  --read-code     Read fenced code blocks out loud (skipped by default)
  --dtype <d>     Model precision: q8 (default, smaller and faster) or fp32
  --out <path>    Write somewhere other than next to the input file
  --quiet         Only print the output path
`;

/* ---------- markdown to readable blocks ---------- */
function inlineText(tokens) {
  if (!tokens) return "";
  return tokens
    .map((t) => {
      switch (t.type) {
        case "text":
          return t.tokens ? inlineText(t.tokens) : t.text || "";
        case "escape":
        case "codespan":
          return t.text || "";
        case "strong":
        case "em":
        case "del":
          return inlineText(t.tokens);
        case "link":
          return inlineText(t.tokens) || t.text || "";
        case "image":
          return t.text ? "Image: " + t.text : "";
        case "br":
          return " ";
        case "html":
          return "";
        default:
          return t.tokens ? inlineText(t.tokens) : t.text || "";
      }
    })
    .join("");
}

function collectBlocks(tokens, out, readCode) {
  for (const t of tokens) {
    switch (t.type) {
      case "heading":
        out.push({ tag: "H" + t.depth, level: t.depth, text: inlineText(t.tokens) });
        break;
      case "paragraph":
        out.push({ tag: "P", text: inlineText(t.tokens) });
        break;
      case "blockquote":
        collectBlocks(t.tokens || [], out, readCode);
        break;
      case "list":
        for (const item of t.items || []) {
          for (const it of item.tokens || []) {
            if (it.type === "text") out.push({ tag: "LI", text: inlineText(it.tokens || [{ type: "text", text: it.text }]) });
            else collectBlocks([it], out, readCode);
          }
        }
        break;
      case "code":
        if (readCode) {
          const spoken = String(t.text || "").split("\n").map((l) => l.trim()).filter(Boolean).join(". ");
          if (spoken) out.push({ tag: "PRE", text: spoken });
        }
        break;
      case "table": {
        const row = (cells) => (cells || []).map((c) => inlineText(c.tokens)).filter(Boolean).join(", ");
        const head = row(t.header);
        if (head) out.push({ tag: "TR", text: head });
        for (const r of t.rows || []) {
          const line = row(r);
          if (line) out.push({ tag: "TR", text: line });
        }
        break;
      }
      case "space":
      case "hr":
      case "html":
        break;
      default:
        if (t.tokens) collectBlocks(t.tokens, out, readCode);
        else if (t.text) out.push({ tag: "P", text: String(t.text) });
    }
  }
}

function stripFrontMatter(md) {
  return md.replace(/^\uFEFF?---[ \t]*\r?\n[\s\S]*?\r?\n(?:---|\.\.\.)[ \t]*(?:\r?\n|$)/, "");
}

const PICTOGRAPHIC = /\p{Extended_Pictographic}\uFE0F?/gu;

function speakable(t) {
  return String(t || "")
    .replace(/https?:\/\/[^\s)]+/g, "link")
    .replace(PICTOGRAPHIC, " ")
    .replace(/[\u2190-\u21FF\u27A1\u2794]/g, " ")
    .replace(/\s[\u2014\u2013]\s|[\u2014]/g, ", ")
    .replace(/[[\]]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/* ---------- chunking and sections ---------- */
function chunkText(t, max) {
  if (!t) return [];
  if (t.length <= max) return [t];
  const parts = t.split(/(?<=[.!?;])\s+/);
  const chunks = [];
  let cur = "";
  for (let s of parts) {
    s = s.trim();
    if (!s) continue;
    if (s.length > max) {
      if (cur) { chunks.push(cur); cur = ""; }
      let piece = "";
      for (const p of s.split(/(?<=,)\s+/)) {
        if (p.length > max) {
          if (piece) { chunks.push(piece); piece = ""; }
          for (const w of p.split(" ")) {
            if (piece && (piece + " " + w).length > max) { chunks.push(piece); piece = w; }
            else piece = piece ? piece + " " + w : w;
          }
          continue;
        }
        if (piece && (piece + " " + p).length > max) { chunks.push(piece); piece = p; }
        else piece = piece ? piece + " " + p : p;
      }
      cur = piece;
      continue;
    }
    if (cur && (cur + " " + s).length > max) { chunks.push(cur); cur = s; }
    else cur = cur ? cur + " " + s : s;
  }
  if (cur) chunks.push(cur);
  return chunks;
}

function joinText(a, b) {
  return /[.!?:;,]$/.test(a) ? a + " " + b : a + ". " + b;
}

function buildSegments(blocks) {
  const pieces = [];
  blocks.forEach((b, i) => {
    if (!b.text) return;
    chunkText(b.text, CHUNK_MAX).forEach((text, k) => pieces.push({ text, block: i, first: k === 0 }));
  });
  const segs = [];
  let cur = null;
  for (const p of pieces) {
    const isHeading = /^H[1-6]$/.test(blocks[p.block].tag);
    const canMerge = !isHeading && cur && cur.mergeable && cur.text.length < MERGE_BELOW &&
      cur.text.length + p.text.length + 2 <= SEG_MAX;
    if (canMerge) {
      cur.text = joinText(cur.text, p.text);
      if (!cur.blocks.includes(p.block)) cur.blocks.push(p.block);
      cur.mergeable = p.first && cur.mergeable;
    } else {
      cur = { text: p.text, blocks: [p.block], mergeable: p.first };
      segs.push(cur);
    }
  }
  const blockSeg = new Map();
  segs.forEach((s, si) => s.blocks.forEach((b) => { if (!blockSeg.has(b)) blockSeg.set(b, si); }));
  return { segs, blockSeg };
}

function chapterPlan(blocks, segs, blockSeg, albumFallback) {
  const heads = [];
  blocks.forEach((b, i) => { if (/^H[1-6]$/.test(b.tag) && blockSeg.has(i)) heads.push(i); });
  const counts = {};
  heads.forEach((i) => { const lv = blocks[i].level; counts[lv] = (counts[lv] || 0) + 1; });
  let level = 0;
  for (let lv = 1; lv <= 6; lv++) { if ((counts[lv] || 0) >= 2) { level = lv; break; } }
  const starts = [];
  if (level) heads.forEach((i) => { if (blocks[i].level === level) starts.push({ seg: blockSeg.get(i), title: blocks[i].text }); });
  const chapters = [];
  const firstStart = starts.length ? starts[0].seg : segs.length;
  if (firstStart > 0) {
    let title = null;
    for (let s = 0; s < firstStart && title === null; s++) {
      for (const bi of segs[s].blocks) {
        if (title === null && /^H[1-6]$/.test(blocks[bi].tag)) title = blocks[bi].text;
      }
    }
    chapters.push({ seg: 0, title: title || (starts.length ? "Introduction" : albumFallback) });
  }
  for (const c of starts) {
    if (!chapters.length || c.seg > chapters[chapters.length - 1].seg) chapters.push(c);
  }
  return chapters.slice(0, 255);
}

/* ---------- ID3v2.3 ---------- */
function u32(n) { return Buffer.from([(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255]); }
function latin1z(s) { return Buffer.concat([Buffer.from(s, "latin1"), Buffer.from([0])]); }
function utf16bom(s) {
  const body = Buffer.from(s, "utf16le");
  return Buffer.concat([Buffer.from([0xff, 0xfe]), body]);
}
function id3Frame(id, payload) {
  return Buffer.concat([Buffer.from(id, "latin1"), u32(payload.length), Buffer.from([0, 0]), payload]);
}
function id3Text(id, value) {
  return id3Frame(id, Buffer.concat([Buffer.from([1]), utf16bom(String(value))]));
}
function id3Tag(fields, chapters) {
  const frames = [];
  for (const [k, v] of Object.entries(fields)) if (v) frames.push(id3Text(k, v));
  if (chapters && chapters.length) {
    const toc = [latin1z("toc"), Buffer.from([0x03, chapters.length])];
    chapters.forEach((c, i) => toc.push(latin1z("ch" + i)));
    frames.push(id3Frame("CTOC", Buffer.concat(toc)));
    chapters.forEach((c, i) => {
      frames.push(id3Frame("CHAP", Buffer.concat([
        latin1z("ch" + i), u32(c.startMs), u32(c.endMs), u32(0xffffffff), u32(0xffffffff),
        id3Text("TIT2", c.title)
      ])));
    });
  }
  const body = Buffer.concat(frames);
  const pad = 512;
  const size = body.length + pad;
  const head = Buffer.from([0x49, 0x44, 0x33, 3, 0, 0,
    (size >> 21) & 0x7f, (size >> 14) & 0x7f, (size >> 7) & 0x7f, size & 0x7f]);
  return Buffer.concat([head, body, Buffer.alloc(pad)]);
}

/* ---------- zip (stored) ---------- */
const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 255] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}
function makeZip(files) {
  const out = [];
  const central = [];
  let offset = 0;
  const d = new Date();
  const dosTime = (d.getHours() << 11) | (d.getMinutes() << 5) | Math.floor(d.getSeconds() / 2);
  const dosDate = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  for (const f of files) {
    const name = Buffer.from(f.name, "utf8");
    const crc = crc32(f.data);
    const lh = Buffer.alloc(30);
    lh.writeUInt32LE(0x04034b50, 0);
    lh.writeUInt16LE(20, 4);
    lh.writeUInt16LE(0x0800, 6);
    lh.writeUInt16LE(dosTime, 10);
    lh.writeUInt16LE(dosDate, 12);
    lh.writeUInt32LE(crc, 14);
    lh.writeUInt32LE(f.data.length, 18);
    lh.writeUInt32LE(f.data.length, 22);
    lh.writeUInt16LE(name.length, 26);
    out.push(lh, name, f.data);
    const ch = Buffer.alloc(46);
    ch.writeUInt32LE(0x02014b50, 0);
    ch.writeUInt16LE(20, 4);
    ch.writeUInt16LE(20, 6);
    ch.writeUInt16LE(0x0800, 8);
    ch.writeUInt16LE(dosTime, 12);
    ch.writeUInt16LE(dosDate, 14);
    ch.writeUInt32LE(crc, 16);
    ch.writeUInt32LE(f.data.length, 20);
    ch.writeUInt32LE(f.data.length, 24);
    ch.writeUInt16LE(name.length, 28);
    ch.writeUInt32LE(offset, 42);
    central.push(ch, name);
    offset += 30 + name.length + f.data.length;
  }
  const cd = Buffer.concat(central);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(files.length, 8);
  end.writeUInt16LE(files.length, 10);
  end.writeUInt32LE(cd.length, 12);
  end.writeUInt32LE(offset, 16);
  return Buffer.concat([...out, cd, end]);
}

/* ---------- audio ---------- */
function encodeInto(enc, float32, out) {
  const i16 = new Int16Array(float32.length);
  for (let k = 0; k < float32.length; k++) {
    let v = float32[k];
    v = v < -1 ? -1 : v > 1 ? 1 : v;
    i16[k] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  const BLOCK = 1152 * 16;
  for (let off = 0; off < i16.length; off += BLOCK) {
    const mp3 = enc.encodeBuffer(i16.subarray(off, off + BLOCK));
    if (mp3.length) out.push(Buffer.from(mp3));
  }
}

function cleanName(t, fallback) {
  const s = String(t || "")
    .replace(/[\\/:*?"<>|\u0000-\u001f]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return (s || fallback).slice(0, 80).trim();
}

/* ---------- main ---------- */
async function main() {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help || !opts.input) {
    process.stdout.write(HELP);
    process.exit(opts.input ? 0 : 1);
  }
  const inputPath = path.resolve(opts.input);
  if (!fs.existsSync(inputPath)) {
    console.error("Not found: " + inputPath);
    process.exit(1);
  }
  const log = (m) => { if (!opts.quiet) process.stderr.write(m + "\n"); };

  if (!VOICE_IDS.includes(opts.voice)) {
    console.error(`Unknown voice "${opts.voice}". Available: ${VOICE_IDS.join(", ")}`);
    process.exit(1);
  }

  const md = stripFrontMatter(fs.readFileSync(inputPath, "utf8"));
  const blocks = [];
  collectBlocks(marked.lexer(md), blocks, opts.readCode);
  for (const b of blocks) b.text = speakable(b.text);
  const readable = blocks.filter((b) => b.text);
  if (!readable.length) {
    console.error("Nothing to read in " + inputPath);
    process.exit(1);
  }
  const base = path.basename(inputPath).replace(/\.(md|markdown|mdown|txt)$/i, "");
  const firstHeading = readable.find((b) => /^H[1-6]$/.test(b.tag));
  const album = cleanName(firstHeading ? firstHeading.text : base, base);

  const { segs, blockSeg } = buildSegments(readable);
  const chapters = chapterPlan(readable, segs, blockSeg, album);
  const startsAt = new Map();
  chapters.forEach((c, ci) => startsAt.set(c.seg, ci));

  const outPath = opts.out
    ? path.resolve(opts.out)
    : path.join(path.dirname(inputPath), base + (opts.single ? ".mp3" : ".zip"));

  log(`Reading ${path.basename(inputPath)}: ${segs.length} parts, ${chapters.length} section${chapters.length === 1 ? "" : "s"}, voice ${opts.voice}`);
  log("Loading the voice model (first run downloads it, then it is cached)");
  const tts = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: opts.dtype, device: "cpu" });

  const started = Date.now();
  const files = [];
  const pad = String(chapters.length).length < 2 ? 2 : String(chapters.length).length;
  let enc = null;
  let chunks = [];
  let sr = 24000;
  let samples = 0;
  let audioSeconds = 0;

  function closeTrack(ci) {
    const tail = enc.flush();
    if (tail.length) chunks.push(Buffer.from(tail));
    const tag = id3Tag({
      TIT2: chapters[ci].title,
      TPE1: "Read to Me",
      TPE2: "Read to Me",
      TALB: album,
      TRCK: `${ci + 1}/${chapters.length}`,
      TCON: "Spoken Word"
    });
    const n = String(ci + 1).padStart(pad, "0");
    files.push({
      name: `${album}/${n} ${cleanName(chapters[ci].title, "Section " + (ci + 1))}.mp3`,
      data: Buffer.concat([tag, ...chunks])
    });
    enc = null;
    chunks = [];
  }

  for (let i = 0; i < segs.length; i++) {
    const audio = await tts.generate(segs[i].text, { voice: opts.voice, speed: opts.speed });
    sr = audio.sampling_rate;
    const ci = startsAt.get(i);
    if (ci !== undefined) {
      if (opts.single) chapters[ci].startMs = Math.round((samples / sr) * 1000);
      else if (enc) closeTrack(ci - 1);
    }
    if (!enc) enc = new lame.Mp3Encoder(1, sr, BITRATE);
    const nextIsChapter = startsAt.has(i + 1);
    const gapSec = !opts.single && nextIsChapter ? 0.8 : (i + 1 < segs.length && blockSeg.get(segs[i + 1].blocks[0]) === i + 1 ? 0.45 : 0.12);
    encodeInto(enc, audio.audio, chunks);
    const silence = new Float32Array(Math.round(sr * gapSec));
    encodeInto(enc, silence, chunks);
    samples += audio.audio.length + silence.length;
    audioSeconds += audio.audio.length / sr;
    const pct = Math.round(((i + 1) / segs.length) * 100);
    const perPart = (Date.now() - started) / 1000 / (i + 1);
    const left = Math.round(perPart * (segs.length - i - 1));
    log(`  ${pct}%  part ${i + 1}/${segs.length}${left > 0 ? `, about ${left < 60 ? left + "s" : Math.round(left / 60) + " min"} left` : ""}`);
  }

  let output;
  if (opts.single) {
    const tail = enc.flush();
    if (tail.length) chunks.push(Buffer.from(tail));
    const totalMs = Math.round((samples / sr) * 1000);
    const chaps = chapters.map((c, ci) => ({
      title: c.title,
      startMs: c.startMs || 0,
      endMs: chapters[ci + 1] ? chapters[ci + 1].startMs : totalMs
    }));
    const tag = id3Tag({ TIT2: album, TPE1: "Read to Me", TALB: album, TCON: "Spoken Word" }, chaps);
    output = Buffer.concat([tag, ...chunks]);
  } else {
    closeTrack(chapters.length - 1);
    output = makeZip(files);
  }

  fs.writeFileSync(outPath, output);
  const mb = (output.length / 1048576).toFixed(1);
  const mins = Math.round(audioSeconds / 60);
  const took = Math.round((Date.now() - started) / 1000);
  log(`Done in ${took < 60 ? took + "s" : Math.round(took / 60) + " min"}: ${mins} min of audio, ${mb} MB, ${opts.single ? chapters.length + " chapters" : files.length + " tracks"}`);
  process.stdout.write(outPath + "\n");
}

main().catch((err) => {
  console.error("Failed: " + (err && err.message ? err.message : err));
  process.exit(1);
});
