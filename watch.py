"""Live self-play viewer.

Runs a background thread where the latest checkpoint plays against itself,
and serves a web page that shows the board live. The model is reloaded
whenever a new checkpoint appears, so you can watch it improve while
train.py is running.

Usage: python watch.py [--port 8000] [--delay 0.7] [--temperature 0.5]
"""

import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess
import chess.svg
import torch

from chessrl.mcts import search, select_move
from chessrl.model import load_model
from chessrl.selfplay import pick_move

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "model.pt")
STATS = os.path.join(os.path.dirname(CKPT), "stats.json")

state_lock = threading.Lock()
state = {
    "svg": chess.svg.board(chess.Board(), size=560),
    "fen": chess.STARTING_FEN,
    "moves": [],
    "game_no": 0,
    "result": None,
    "value_white": 0.0,
    "model_iter": 0,
    "history": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
    "training": None,
}


def latest_training_stats():
    try:
        with open(STATS) as f:
            history = json.load(f)
        return history[-1] if history else None
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def game_loop(delay, temperature, max_plies, sims):
    torch.set_num_threads(2)
    model, model_iter = load_model(CKPT)
    ckpt_mtime = os.path.getmtime(CKPT) if os.path.exists(CKPT) else 0
    game_no = 0
    while True:
        # Pick up a newer checkpoint between games.
        mtime = os.path.getmtime(CKPT) if os.path.exists(CKPT) else 0
        if mtime != ckpt_mtime:
            model, model_iter = load_model(CKPT)
            ckpt_mtime = mtime
        game_no += 1
        board = chess.Board()
        moves_san = []
        with state_lock:
            state.update(game_no=game_no, moves=[], result=None,
                         model_iter=model_iter, fen=board.fen(),
                         svg=chess.svg.board(board, size=560),
                         training=latest_training_stats())
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if sims > 0:
                visits, moves, value = search(model, board, sims, add_noise=False)
                move, _ = select_move(visits, moves, temperature)
            else:
                move, _, value = pick_move(model, board, temperature)
            value_white = value if board.turn == chess.WHITE else -value
            moves_san.append(board.san(move))
            board.push(move)
            check_sq = board.king(board.turn) if board.is_check() else None
            svg = chess.svg.board(board, size=560, lastmove=move, check=check_sq)
            with state_lock:
                state.update(svg=svg, fen=board.fen(), moves=list(moves_san),
                             value_white=round(value_white, 3))
            time.sleep(delay)
        result = board.result(claim_draw=True)
        if result == "*":
            result = "1/2-1/2 (truncated)"
        with state_lock:
            key = result.split(" ")[0]
            if key in state["history"]:
                state["history"][key] += 1
            state["result"] = result
        time.sleep(3.0)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Chess RL — self-play</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { background: #14171c; color: #dde3ea; font: 15px/1.5 system-ui, sans-serif;
         display: flex; justify-content: center; padding: 28px 16px; }
  .wrap { display: flex; gap: 28px; flex-wrap: wrap; max-width: 1000px; }
  .board { width: min(560px, 92vw); }
  .board svg { width: 100%; height: auto; border-radius: 6px; }
  .side { width: 330px; display: flex; flex-direction: column; gap: 14px; }
  h1 { font-size: 19px; font-weight: 650; }
  .card { background: #1d222a; border: 1px solid #2a313c; border-radius: 8px; padding: 12px 14px; }
  .muted { color: #8b96a5; font-size: 13px; }
  .row { display: flex; justify-content: space-between; padding: 2px 0; }
  .evalbar { height: 10px; background: #2a313c; border-radius: 5px; overflow: hidden; margin-top: 6px; }
  .evalbar div { height: 100%; background: #e8e8e8; transition: width .4s; }
  .moves { max-height: 300px; overflow-y: auto; font-family: ui-monospace, monospace;
           font-size: 13px; line-height: 1.7; }
  .num { color: #8b96a5; margin-right: 4px; }
  .mv { margin-right: 10px; }
  .result { font-weight: 650; color: #ffc86b; }
</style></head>
<body><div class="wrap">
  <div class="board" id="board"></div>
  <div class="side">
    <h1>♟ Chess RL — self-play</h1>
    <div class="card">
      <div class="row"><span class="muted">Game</span><span id="game"></span></div>
      <div class="row"><span class="muted">Model iteration</span><span id="iter"></span></div>
      <div class="row"><span class="muted">Session score W / D / B</span><span id="hist"></span></div>
      <div class="row"><span class="muted">Eval (white)</span><span id="val"></span></div>
      <div class="evalbar"><div id="bar" style="width:50%"></div></div>
    </div>
    <div class="card">
      <div class="muted" style="margin-bottom:6px">Moves <span class="result" id="result"></span></div>
      <div class="moves" id="moves"></div>
    </div>
    <div class="card" id="train-card">
      <div class="muted" style="margin-bottom:6px">Latest training iteration</div>
      <div id="train" class="muted">no stats yet</div>
    </div>
  </div>
</div>
<script>
async function tick() {
  try {
    const s = await (await fetch('/state')).json();
    document.getElementById('board').innerHTML = s.svg;
    document.getElementById('game').textContent = '#' + s.game_no;
    document.getElementById('iter').textContent = s.model_iter || 'untrained';
    const h = s.history;
    document.getElementById('hist').textContent = h['1-0'] + ' / ' + h['1/2-1/2'] + ' / ' + h['0-1'];
    document.getElementById('val').textContent = (s.value_white >= 0 ? '+' : '') + s.value_white.toFixed(2);
    document.getElementById('bar').style.width = ((s.value_white + 1) * 50) + '%';
    document.getElementById('result').textContent = s.result ? '· ' + s.result : '';
    let html = '';
    for (let i = 0; i < s.moves.length; i += 2) {
      html += '<span class="num">' + (i / 2 + 1) + '.</span><span class="mv">' + s.moves[i] + '</span>';
      if (s.moves[i + 1]) html += '<span class="mv">' + s.moves[i + 1] + '</span>';
    }
    const mv = document.getElementById('moves');
    mv.innerHTML = html;
    mv.scrollTop = mv.scrollHeight;
    if (s.training) {
      const t = s.training;
      document.getElementById('train').innerHTML =
        'iter ' + t.iter + ' — ' + t.games + ' games, avg ' + t.avg_plies + ' plies<br>' +
        'W/B/D/trunc: ' + t.white_wins + '/' + t.black_wins + '/' + t.draws + '/' + t.truncated + '<br>' +
        'policy loss ' + t.policy_loss + ', value loss ' + t.value_loss + ' (' + t.seconds + 's)';
    }
  } catch (e) {}
}
setInterval(tick, 500); tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif self.path == "/state":
            with state_lock:
                body = json.dumps(state).encode()
            ctype = "application/json"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--delay", type=float, default=0.7)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--sims", type=int, default=64,
                    help="MCTS simulations per move (0 = raw policy, no search)")
    args = ap.parse_args()

    threading.Thread(target=game_loop,
                     args=(args.delay, args.temperature, args.max_plies, args.sims),
                     daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"watching self-play at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
