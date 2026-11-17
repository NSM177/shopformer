import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const distDir = path.join(__dirname, 'dist');
const assetsDir = path.join(distDir, 'assets');
const targetHtmlPath = path.resolve(__dirname, '../src/shoplift_iot/dashboard/templates/index.html');
const targetStaticDir = path.resolve(__dirname, '../src/shoplift_iot/dashboard/static/assets');

fs.mkdirSync(targetStaticDir, { recursive: true });

let html = fs.readFileSync(path.join(distDir, 'index.html'), 'utf-8');

// Copy assets to dashboard/static/assets as well
if (fs.existsSync(assetsDir)) {
  for (const file of fs.readdirSync(assetsDir)) {
    fs.copyFileSync(path.join(assetsDir, file), path.join(targetStaticDir, file));
  }
}

// Inline CSS and JS into templates/index.html so FastAPI GET / serves the complete React SPA self-contained
const cssPath = path.join(assetsDir, 'style.css');
if (fs.existsSync(cssPath)) {
  const cssContent = fs.readFileSync(cssPath, 'utf-8');
  html = html.replace(/<link[^>]+href="\/assets\/style\.css"[^>]*>/g, () => `<style>${cssContent}</style>`);
}

const jsPath = path.join(assetsDir, 'app.js');
if (fs.existsSync(jsPath)) {
  const jsContent = fs.readFileSync(jsPath, 'utf-8');
  html = html.replace(
    /<script[^>]+src="\/assets\/app\.js"[^>]*><\/script>/g,
    () => `<script type="module">${jsContent}</script>`
  );
}

fs.writeFileSync(targetHtmlPath, html, 'utf-8');
console.log(`Built React bundle into ${targetHtmlPath}`);

