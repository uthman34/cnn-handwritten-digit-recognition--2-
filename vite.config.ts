import fs from 'fs';
import path from 'path';
import { defineConfig } from 'vite';

export default defineConfig(() => {
  const publicRoot = path.resolve(__dirname, 'public');
  const distDir = path.resolve(__dirname, 'dist');

  return {
    root: publicRoot,
    publicDir: false as const,
    build: {
      outDir: distDir,
      emptyOutDir: true,
    },
    plugins: [
      // Copy static runtime assets (TF.js bundle, CSS, and converted Python model) into dist/
      {
        name: 'copy-static-tfjs-assets',
        apply: 'build',
        closeBundle(error?: Error): void {
          if (error || !fs.existsSync(distDir)) {
            return;
          }
          for (const dirName of ['tfjs_model', 'emnist_model', 'vendor']) {
            const srcDir = path.join(publicRoot, dirName);
            const destDir = path.join(distDir, dirName);
            if (fs.existsSync(srcDir)) {
              fs.cpSync(srcDir, destDir, { recursive: true, force: true });
            }
          }
          const cssSrc = path.join(publicRoot, 'styles.css');
          if (fs.existsSync(cssSrc)) {
            fs.copyFileSync(cssSrc, path.join(distDir, 'styles.css'));
          }
          const faviconSrc = path.join(publicRoot, 'favicon.svg');
          if (fs.existsSync(faviconSrc)) {
            fs.copyFileSync(faviconSrc, path.join(distDir, 'favicon.svg'));
          }
        },
      },
    ],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      hmr: process.env.DISABLE_HMR !== 'true',
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
