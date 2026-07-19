# Publishing to GitHub

This folder is a self-contained code release. To publish it as its own public
repository (kept separate from the private working repo):

```bash
# from a fresh copy of this folder (outside the private repo):
cp -r release/hermes-hecktor2026 ~/hermes-hecktor2026 && cd ~/hermes-hecktor2026
git init && git add -A && git commit -m "HERMES (HECKTOR 2026) code release"
# create an empty public repo on GitHub, then:
git remote add origin git@github.com:wangkaiwan/hermes-hecktor2026.git
git branch -M main && git push -u origin main
```

After creating the repository, fill the two placeholders:
- `README.md` / paper §7 — the repository URL.
- `MODELS.md` — the checkpoint download link (GitHub Release asset, Zenodo DOI, or
  Grand Challenge Model).
