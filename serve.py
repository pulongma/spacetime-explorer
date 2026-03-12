#!/usr/bin/env python3
"""
gpbayeskit — Space-Time Covariance Explorer server
Serves the HTML and a /compute endpoint using exact scipy special functions.

Usage:
    python serve.py [port]        # default port 8765
    python serve.py --open        # also opens browser automatically

Then open:  http://localhost:8765
"""
import json
import sys
import os
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import scipy.special as sc


# ── Kernels ──────────────────────────────────────────────────────────────────

def _matern(r, phi, nu):
    """Matern C(r; phi, nu),  z = r/phi  (no sqrt(2*nu) pre-factor)."""
    z  = np.maximum(np.asarray(r, float), 0.0) / float(phi)
    nu = float(nu)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        if   nu == 0.5: out = np.exp(-z)
        elif nu == 1.5: out = (1.0 + z) * np.exp(-z)
        elif nu == 2.5: out = (1.0 + z + z*z/3.0) * np.exp(-z)
        else:
            c   = (2.0 ** (1.0 - nu)) / sc.gamma(nu)
            out = c * np.power(z, nu) * sc.kv(nu, z)
        out = np.where(z < 1e-10, 1.0, out)
        out = np.where(np.isfinite(out), out, 0.0)
    return out


def _ch(r, phi, nu, alpha):
    """CH(r; phi, nu, alpha) = Gamma(nu+alpha)/Gamma(nu) * U(alpha, 1-nu, (r/phi)^2)."""
    z  = np.maximum(np.asarray(r, float), 0.0) / float(phi)
    z2 = z * z
    c  = np.exp(sc.gammaln(float(nu) + float(alpha)) - sc.gammaln(float(nu)))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = c * sc.hyperu(float(alpha), float(1.0 - nu), z2)
        out = np.where(z < 1e-10, 1.0, out)
        out = np.where(np.isfinite(out) & (out >= 0), out, 0.0)
    return out


def compute_grid(model, h, u, p):
    """
    Evaluate C(h, u) on a meshgrid.
    h, u : 1-D float arrays
    p    : dict of model parameters
    Returns: 2-D float array, shape (len(u), len(h))
    """
    H, U = np.meshgrid(h, u)

    if model == 'sep':
        Cs = _matern(np.abs(H), p['phi_s'], p['nu_s'])
        Ct = _matern(np.abs(U), p['phi_t'], p['nu_t'])
        C  = p['sigma2'] * Cs * Ct

    elif model == 'gne':
        absU = np.abs(U)
        psi  = np.where(absU < 1e-10, 1.0,
                        p['phi_t'] * absU ** (2 * p['alpha']) + 1.0)
        r    = np.abs(H) / psi ** (p['beta'] / 2.0)
        C    = p['sigma2'] * psi ** (-p['delta']) * _matern(r, p['phi_s'], p['nu_s'])

    elif model == 'ch':
        d    = 2
        absU = np.abs(U)
        psi  = np.where(absU < 1e-10, 1.0,
                        p['phi_t'] * absU ** (2 * p['alpha']) + 1.0)
        C    = p['sigma2'] * psi ** (-d / 2.0) \
                           * np.exp(-H ** 2 / (p['phi_s'] ** 2 * psi))

    elif model == 'stein':
        C = np.zeros_like(H)
        for iy, u_val in enumerate(u):
            nu_u  = max(0.5, p['nu'] + p['zeta'] * abs(float(u_val)))
            r_row = np.abs(H[iy] - p['eps'] * u_val)
            C[iy] = p['sigma2'] * _matern(r_row, p['phi'], nu_u)

    elif model == 'lmat':
        d     = 2
        denom = 1.0 + p['lam'] * U ** 2
        scale = denom ** (-d / 2.0)
        r     = np.abs(H - p['v'] * U) / np.sqrt(denom)
        C     = p['sigma2'] * scale * _matern(r, p['phi'], p['nu'])

    elif model == 'lch':
        d     = 2
        denom = 1.0 + p['lam'] * U ** 2
        scale = denom ** (-d / 2.0)
        r     = np.abs(H - p['v'] * U) / np.sqrt(denom)
        C     = p['sigma2'] * scale * _ch(r, p['phi'], p['nu'], p['alpha'])

    else:
        C = np.zeros_like(H)

    return C


# ── HTTP handler ──────────────────────────────────────────────────────────────

HTML_FILE = os.path.join(os.path.dirname(__file__), 'st_covariance.html')


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # suppress request log noise

    def _send(self, code, ctype, body):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html', '/st_covariance.html'):
            try:
                with open(HTML_FILE, 'rb') as f:
                    self._send(200, 'text/html; charset=utf-8', f.read())
            except FileNotFoundError:
                self._send(404, 'text/plain', 'HTML file not found')
        elif path == '/ping':
            self._send(200, 'application/json', '{"status":"ok","engine":"scipy"}')
        else:
            self._send(404, 'text/plain', 'Not found')

    def do_POST(self):
        if urlparse(self.path).path != '/compute':
            self._send(404, 'text/plain', 'Not found')
            return

        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length)

        try:
            req   = json.loads(body)
            model = req['model']
            NX    = int(req['NX']);    NY    = int(req['NY'])
            hMin  = float(req['hMin']); hMax = float(req['hMax'])
            uMin  = float(req['uMin']); uMax = float(req['uMax'])
            p     = req['params']

            h = np.linspace(hMin, hMax, NX)
            u = np.linspace(uMin, uMax, NY)
            C = compute_grid(model, h, u, p)

            resp = json.dumps({'grid': C.flatten().tolist(),
                               'NX': NX, 'NY': NY})
            self._send(200, 'application/json', resp)

        except Exception as e:
            self._send(500, 'application/json',
                       json.dumps({'error': str(e)}))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args  = sys.argv[1:]
    port  = 8765
    do_open = '--open' in args
    for a in args:
        if a.isdigit():
            port = int(a)

    if not os.path.exists(HTML_FILE):
        # Try same directory as this script
        alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'st_covariance.html')
        if os.path.exists(alt):
            global HTML_FILE
            HTML_FILE = alt
        else:
            print(f"Warning: st_covariance.html not found next to serve.py")

    server = HTTPServer(('localhost', port), Handler)
    url    = f'http://localhost:{port}'
    print(f"gpbayeskit covariance explorer")
    print(f"  serving : {url}")
    print(f"  engine  : Python {sys.version.split()[0]} + scipy {sc.__version__}")
    print(f"  press Ctrl-C to stop\n")

    if do_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopped.')


if __name__ == '__main__':
    main()
