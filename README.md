# ML Notes Book

An interactive, connected book of machine-learning notes (geometry → regression → classification → ensembles → latent variables).

## Run / publish
- **Locally:** `python3 -m http.server` in this folder, open http://localhost:8000 (needed because chapters are fetched).
- **GitHub Pages:** push, then Settings → Pages → Deploy from branch → `main` / root.

## Add a chapter
1. Create `chapters/NN-name.md` (see chapter 1 for the format).
2. Add an entry in `chapters.json` (`connects` draws the arrows on the home-page concept map).
3. Figures: `:::figure F1.1 | Title | description` is a placeholder; `:::widget name | Title` mounts a function from `assets/widgets.js`.

Math is KaTeX (`$...$`, `$$...$$`). See `ROADMAP.md` for the plan and extra topics.
