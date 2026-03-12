# Space-Time Covariance Explorer

Interactive visualisation of six space-time covariance functions.

**Live demo →** https://pulongma.github.io/spacetime-explorer/

## Models

| Model | Reference |
|---|---|
| Separable | Product of Matérn marginals |
| Gneiting | Gneiting (2002) JASA 97(458) |
| Cressie-Huang | Cressie & Huang (1999) |
| Stein | Stein (2005) JASA 100(469) |
| Lagrangian Matérn | Ma (2025) arXiv:2511.07959 |
| Lagrangian CH | Ma (2025) arXiv:2511.07959 |

## Compute engines (used in order)

1. **Local Python server** — exact `scipy.special.kv` and `scipy.special.hyperu`  
   Works from `file://` and any origin.  
   ```bash
   pip install numpy scipy
   python serve.py --open
   ```
2. **Pyodide** — same scipy functions, runs in-browser via WebAssembly.  
   Loads automatically on `https://` (GitHub Pages). No server needed.

## Deploy to GitHub Pages

```bash
git init
git add .
git commit -m "initial commit"
gh repo create <repo-name> --public --source=. --remote=origin --push
# then: Settings → Pages → Deploy from branch main / root
```

Or push to an existing repo and enable Pages in Settings.
