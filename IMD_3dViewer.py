#!/usr/bin/env python3
"""
IMD_3dViewer — Interactive Memory Dimension Viewer
Real-time 3D visualization of ASMC memory space (STM + LTM)

9D coordinates mapped as:
  Position (x, y, z) → world position
  Scale    (d, e, f) → point size (phase 1)
  Rotation (a, b, c) → color tint (phase 1)

Colors:
  Blue   = STM (RAM)         Orange = LTM (LMDB)
  White  = semantic links    Green  = succession links
  Yellow = radial links

Controls:
  Left drag → orbit    Scroll → zoom    R → reset    Q/Esc → quit
"""

import json
import os
import time
import math
import pickle
import threading
import numpy as np
import glfw
import moderngl
from imgui_bundle import imgui
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer

STM_PATHS = [
    "MemoryStructures/STM/stm_cache_A.json",
    "MemoryStructures/STM/stm_cache_B.json",
]
LTM_PATH    = "MemoryStructures/LTM/ltm.lmdb"
POLL_INTERVAL = 2.0

COLOR_STM        = (0.3, 0.6, 1.0, 1.0)
COLOR_LTM        = (1.0, 0.55, 0.15, 1.0)
COLOR_SEMANTIC   = (1.0, 1.0, 1.0, 0.5)
COLOR_SUCCESSION = (0.3, 1.0, 0.4, 0.6)
COLOR_RADIAL     = (1.0, 0.9, 0.2, 0.45)


# ─────────────────────────────────────────────────────────────────────────────
class DataReader:
    def __init__(self, base_path):
        self.base_path   = base_path
        self.lock        = threading.Lock()
        self.nodes          = {}   # coord_key → {pos, color, size, type}
        self.edges          = []   # [(ka, kb, color_arr)]
        self._stm_mtime     = 0
        self._stm_keys      = set()
        self.last_positions = None
        self.last_keys      = []
        self.selected_key   = None

    def _load_stm(self):
        best_data, best_mtime = None, 0
        for rel in STM_PATHS:
            path = os.path.join(self.base_path, rel)
            try:
                mt = os.path.getmtime(path)
                if mt > best_mtime:
                    best_mtime = mt
                    with open(path) as f:
                        best_data = json.load(f)
            except Exception:
                pass

        if best_data is None or best_mtime == self._stm_mtime:
            return
        self._stm_mtime = best_mtime

        entries     = best_data.get("stm_entries", {})
        new_keys    = set()

        for coord_key, entry in entries.items():
            c   = entry.get("coordinates", {})
            pos = np.array([c.get("x", 0), c.get("y", 0), c.get("z", 0)], dtype=np.float32)
            d, e, f = c.get("d", 0), c.get("e", 0), c.get("f", 0)
            size = max(5.0, 10.0 + (abs(d) + abs(e) + abs(f)) * 5.0)

            self.nodes[coord_key] = {
                "pos":   pos,
                "color": np.array(COLOR_STM, dtype=np.float32),
                "size":  size,
                "type":  "stm",
                "label": entry.get("semantic_summary", "")[:50],
                "abc":   np.array([c.get("a",0), c.get("b",0), c.get("c",0)], dtype=np.float32),
            }
            new_keys.add(coord_key)

            for linked in entry.get("semantic_links", []):
                self.edges.append((coord_key, linked,
                                   np.array(COLOR_SEMANTIC, dtype=np.float32)))

        # Remove promoted-away STM nodes
        for old in self._stm_keys - new_keys:
            if old in self.nodes and self.nodes[old]["type"] == "stm":
                del self.nodes[old]
        self._stm_keys = new_keys

    def _load_ltm(self):
        path = os.path.join(self.base_path, LTM_PATH)
        if not os.path.exists(path):
            return
        try:
            import lmdb
            env = lmdb.open(path, readonly=True, lock=False)
            with env.begin() as txn:
                for _, raw in txn.cursor():
                    try:
                        mem    = pickle.loads(raw)
                        coords = mem.get("coordinates", {})
                        if not coords:
                            continue
                        meta      = mem.get("metadata", {})
                        coord_key = meta.get("coordinate_key", "")
                        if not coord_key:
                            continue

                        pos  = np.array([coords.get("x", 0), coords.get("y", 0),
                                         coords.get("z", 0)], dtype=np.float32)
                        d, e, f = coords.get("d", 0), coords.get("e", 0), coords.get("f", 0)
                        size = max(4.0, 7.0 + (abs(d) + abs(e) + abs(f)) * 4.0)

                        # Don't overwrite an STM node still in RAM
                        if coord_key not in self.nodes:
                            self.nodes[coord_key] = {
                                "pos":   pos,
                                "color": np.array(COLOR_LTM, dtype=np.float32),
                                "size":  size,
                                "type":  "ltm",
                                "label": mem.get("semantic_summary", "")[:50],
                                "mem":   mem,
                                "abc":   np.array([coords.get("a",0), coords.get("b",0), coords.get("c",0)], dtype=np.float32),
                            }

                        links = mem.get("semantic_links", {}) or meta.get("semantic_links", {})
                        for lnk in links.get("succession_links", []):
                            tk = lnk.get("target_coordinate_key", "")
                            if tk:
                                self.edges.append((coord_key, tk,
                                    np.array(COLOR_SUCCESSION, dtype=np.float32)))
                        for lnk in links.get("radial_links", []):
                            tk = lnk.get("target_coordinate_key", "")
                            if tk:
                                self.edges.append((coord_key, tk,
                                    np.array(COLOR_RADIAL, dtype=np.float32)))
                        for lnk in links.get("semantic_links", []):
                            tk = lnk.get("target_coordinate_key", "")
                            if tk:
                                self.edges.append((coord_key, tk,
                                    np.array(COLOR_SEMANTIC, dtype=np.float32)))
                        for tk in mem.get("metadata", {}).get("metadata", {}).get("stm_semantic_links", []):
                            if tk:
                                self.edges.append((coord_key, tk,
                                    np.array(COLOR_SEMANTIC, dtype=np.float32)))
                    except Exception:
                        pass
            env.close()
        except Exception:
            pass

    def pick(self, cx, cy, mvp_mat, w, h, radius=15):
        if self.last_positions is None or len(self.last_keys) == 0:
            return None
        with self.lock:
            pos  = self.last_positions
            keys = self.last_keys
        ones = np.ones((len(pos), 1), dtype=np.float32)
        clip = (mvp_mat @ np.hstack([pos, ones]).T).T
        ndc  = clip[:, :3] / np.maximum(clip[:, 3:4], 1e-6)
        sx   = (ndc[:, 0] + 1.0) * 0.5 * w
        sy   = (1.0 - ndc[:, 1]) * 0.5 * h
        dists = np.hypot(sx - cx, sy - cy)
        idx   = int(np.argmin(dists))
        return keys[idx] if dists[idx] < radius else None

    def lookup_node(self, key):
        node = self.nodes.get(key)
        if node is None:
            return None
        if node["type"] == "stm":
            for rel in STM_PATHS:
                try:
                    with open(os.path.join(self.base_path, rel)) as f:
                        data = json.load(f)
                    entry = data.get("stm_entries", {}).get(key)
                    if entry:
                        return {"type": "stm", "key": key, **entry}
                except Exception:
                    pass
        else:
            mem = node.get("mem")
            if mem:
                return {"type": "ltm", "key": key, **mem}
        return None

    def refresh(self):
        with self.lock:
            self.edges = []
            self._load_stm()
            self._load_ltm()

    def get_gpu_arrays(self, show_stm_nodes=True, show_ltm_nodes=True,
                       show_stm_links=True, show_ltm_links=True,
                       show_scm=True, show_spikes=True):
        with self.lock:
            all_keys = list(self.nodes.keys())
            keys = [k for k in all_keys if
                    (self.nodes[k]["type"] == "stm" and show_stm_nodes) or
                    (self.nodes[k]["type"] == "ltm" and show_ltm_nodes)]
            if not keys:
                return None

            positions = np.array([self.nodes[k]["pos"]   for k in keys], dtype=np.float32)
            colors    = np.array([self.nodes[k]["color"] for k in keys], dtype=np.float32)
            sizes     = np.array([self.nodes[k]["size"]  for k in keys], dtype=np.float32)

            # Center on cluster centroid, then scale each axis independently
            centroid     = positions.mean(axis=0)
            centered     = positions - centroid
            axis_spread  = np.abs(centered).max(axis=0)
            axis_spread  = np.maximum(axis_spread, 1e-6)
            positions    = centered / axis_spread * 100.0
            self.last_positions = positions
            self.last_keys      = keys
            pos_by_key = {k: positions[i] for i, k in enumerate(keys)}

            # Selection dimming
            sel = self.selected_key
            connected = set()
            if sel and sel in pos_by_key:
                connected.add(sel)
                for (ka, kb, _) in self.edges:
                    if ka == sel: connected.add(kb)
                    if kb == sel: connected.add(ka)
                for i, k in enumerate(keys):
                    if k not in connected:
                        colors[i, 3] *= 0.08

            lpos, lcol, seen = [], [], set()
            for (ka, kb, col) in self.edges:
                if ka in pos_by_key and kb in pos_by_key:
                    ek = (min(ka, kb), max(ka, kb))
                    if ek not in seen:
                        seen.add(ek)
                        # Identify edge type by node types and color
                        ta = self.nodes.get(ka, {}).get("type", "ltm")
                        tb = self.nodes.get(kb, {}).get("type", "ltm")
                        is_scm = col[0] < 0.5 and col[1] > 0.8  # green = succession
                        is_stm_edge = ta == "stm" and tb == "stm"
                        if is_scm:
                            if not show_scm: continue
                        else:
                            if is_stm_edge and not show_stm_links: continue
                            if not is_stm_edge and not show_ltm_links: continue
                        dim = 1.0 if (not sel or ka in connected or kb in connected) else 0.08
                        c2  = np.array([col[0], col[1], col[2], col[3] * dim], dtype=np.float32)
                        lpos += [pos_by_key[ka], pos_by_key[kb]]
                        lcol += [c2, c2]

            # Direction spikes (a,b,c vector)
            SPIKE = 12.0; ARM = 4.0
            SCOL  = np.array([0.4, 0.8, 1.0, 0.4], dtype=np.float32)
            UP    = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            for k in (keys if show_spikes else []):
                abc = self.nodes[k].get("abc")
                if abc is None: continue
                mag = float(np.linalg.norm(abc))
                if mag < 1e-6: continue
                dim = 1.0 if (not sel or k in connected) else 0.08
                sc  = np.array([SCOL[0], SCOL[1], SCOL[2], SCOL[3] * dim], dtype=np.float32)
                n   = abc / mag
                p   = pos_by_key[k]; tip = p + n * SPIKE
                ax  = UP if abs(n[1]) < 0.9 else np.array([1.,0.,0.], dtype=np.float32)
                r   = np.cross(n, ax); r /= np.linalg.norm(r)
                u   = np.cross(r, n)
                lpos += [p, tip, tip - r*ARM, tip + r*ARM, tip - u*ARM, tip + u*ARM]
                lcol += [sc] * 6

            lpos_arr = np.array(lpos, dtype=np.float32) if lpos else np.zeros((0, 3), dtype=np.float32)
            lcol_arr = np.array(lcol, dtype=np.float32) if lcol else np.zeros((0, 4), dtype=np.float32)

            stm_count = sum(1 for k in keys if self.nodes[k]["type"] == "stm")
            ltm_count = len(keys) - stm_count

            return positions, colors, sizes, lpos_arr, lcol_arr, stm_count, ltm_count


# ─────────────────────────────────────────────────────────────────────────────
class OrbitCamera:
    def __init__(self):
        self.az   = 0.3
        self.el   = 0.4
        self.dist = 200.0
        self.target = np.zeros(3, dtype=np.float32)
        self._drag        = False
        self._last_xy     = (0, 0)
        self._press_xy    = (0, 0)
        self.pending_click = None

    def on_mouse_button(self, win, btn, action):
        if btn == glfw.MOUSE_BUTTON_LEFT:
            cx, cy = glfw.get_cursor_pos(win)
            if action == glfw.PRESS:
                self._drag     = True
                self._press_xy = (cx, cy)
                self._last_xy  = (cx, cy)
            else:
                dx, dy = cx - self._press_xy[0], cy - self._press_xy[1]
                if dx * dx + dy * dy < 25:
                    self.pending_click = (cx, cy)
                self._drag = False

    def on_cursor(self, x, y):
        if self._drag:
            dx, dy = x - self._last_xy[0], y - self._last_xy[1]
            self.az -= dx * 0.005
            self.el  = max(-1.55, min(1.55, self.el + dy * 0.005))
        self._last_xy = (x, y)

    def on_scroll(self, dy):
        self.dist = max(1.0, self.dist * (0.9 if dy > 0 else 1.1))

    def reset(self):
        self.az, self.el, self.dist = 0.3, 0.4, 200.0

    def view(self):
        ce, se = math.cos(self.el), math.sin(self.el)
        ca, sa = math.cos(self.az), math.sin(self.az)
        eye = self.target + self.dist * np.array([ce * sa, se, ce * ca], dtype=np.float32)
        fwd = self.target - eye;  fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0, 1, 0], dtype=np.float32))
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1, 0, 0], dtype=np.float32)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        m = np.eye(4, dtype=np.float32)
        m[0, :3] = right;  m[1, :3] = up;  m[2, :3] = -fwd
        m[:3, 3] = -m[:3, :3] @ eye
        return m

    def proj(self, w, h):
        f  = 1.0 / math.tan(math.radians(60) / 2)
        ar = w / max(h, 1)
        n, fa = 0.5, 5000.0
        p = np.zeros((4, 4), dtype=np.float32)
        p[0, 0] = f / ar;  p[1, 1] = f
        p[2, 2] = (fa + n) / (n - fa);  p[2, 3] = 2 * fa * n / (n - fa)
        p[3, 2] = -1.0
        return p


# ─────────────────────────────────────────────────────────────────────────────
POINT_VERT = """
#version 330 core
in vec3 in_position;
in vec4 in_color;
in float in_size;
uniform mat4 mvp;
out vec4 v_color;
void main() {
    gl_Position  = mvp * vec4(in_position, 1.0);
    gl_PointSize = clamp(in_size * 200.0 / gl_Position.w, 2.0, 32.0);
    v_color = in_color;
}
"""
POINT_FRAG = """
#version 330 core
in vec4 v_color;
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = dot(c, c);
    if (r > 1.0) discard;
    float edge = 1.0 - smoothstep(0.6, 1.0, r);
    f_color = vec4(v_color.rgb, v_color.a * edge);
}
"""
LINE_VERT = """
#version 330 core
in vec3 in_position;
in vec4 in_color;
uniform mat4 mvp;
out vec4 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
"""
LINE_FRAG = """
#version 330 core
in vec4 v_color;
out vec4 f_color;
void main() { f_color = v_color; }
"""


# ─────────────────────────────────────────────────────────────────────────────
def _draw_info_panel(info):
    imgui.set_next_window_pos((10.0, 10.0), imgui.Cond_.once)
    imgui.set_next_window_size((460.0, 340.0), imgui.Cond_.once)
    imgui.push_style_color(imgui.Col_.window_bg, (0.07, 0.07, 0.12, 0.93))
    imgui.begin("Memory Node  (click empty space to clear)")
    imgui.pop_style_color()

    t = info.get("type", "?").upper()
    tint = (0.4, 0.8, 1.0, 1.0) if t == "STM" else (1.0, 0.6, 0.2, 1.0)
    imgui.text_colored(tint, f"[{t}]")
    imgui.same_line()
    imgui.text(info.get("key", "")[:55])
    imgui.separator()
    imgui.text_wrapped(info.get("semantic_summary", "") or "—")
    imgui.spacing()

    c = info.get("coordinates", {})
    imgui.text(f"x={c.get('x',0):.4f}  y={c.get('y',0):.4f}  z={c.get('z',0):.4f}")
    imgui.text(f"d={c.get('d',0):.4f}  e={c.get('e',0):.4f}  f={c.get('f',0):.4f}")
    imgui.separator()

    meta = info.get("metadata", {})
    tags = meta.get("tags") or meta.get("metadata", {}).get("tags", [])
    imgui.text(f"Tags:   {tags}")
    imgui.text(f"Intent: {meta.get('directive_intent', '') or meta.get('metadata', {}).get('directive_intent', '') or '—'}")

    if t == "STM":
        user_in = info.get("user_input", "")
        ai_out  = info.get("ai_response", "")
    else:
        nested  = meta.get("metadata", {})
        user_in = nested.get("user_input", "") or info.get("input_text", "") or info.get("input", "")
        ai_out  = nested.get("ai_response", "")
        links   = meta.get("semantic_links", {})
        succ    = len(links.get("succession_links", []))
        radial  = len(links.get("radial_links", []))
        stm_lnk = len(nested.get("stm_semantic_links", []))
        imgui.text(f"Links:  {succ} succession  {radial} radial  {stm_lnk} stm")

    imgui.separator()
    imgui.text_wrapped(f"User: {str(user_in)[:220]}")
    imgui.separator()
    imgui.text_wrapped(f"AI:   {str(ai_out)[:220]}")

    imgui.end()


def _print_node(info):
    if info is None:
        print("\n── No node at click ──────────────────────")
        return
    t = info.get("type", "?").upper()
    print(f"\n── {t} NODE ──────────────────────────────────────────")
    print(f"  Key:     {info.get('key', '')}")
    print(f"  Summary: {info.get('semantic_summary', '')}")
    c = info.get("coordinates", {})
    print(f"  Coords:  x={c.get('x',0):.4f}  y={c.get('y',0):.4f}  z={c.get('z',0):.4f}")
    print(f"           d={c.get('d',0):.4f}  e={c.get('e',0):.4f}  f={c.get('f',0):.4f}")
    meta = info.get("metadata", {})
    print(f"  Tags:    {meta.get('tags', [])}")
    print(f"  Intent:  {meta.get('directive_intent', '')}")
    if t == "STM":
        print(f"  User:    {str(info.get('user_input', ''))[:140]}")
        print(f"  AI:      {str(info.get('ai_response', ''))[:140]}")
    else:
        nested = meta.get("metadata", {})
        print(f"  User:    {str(nested.get('user_input', '') or info.get('input_text', ''))[:140]}")
        print(f"  AI:      {str(nested.get('ai_response', ''))[:140]}")
    print("──────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    reader = DataReader(base)
    cam    = OrbitCamera()

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.SAMPLES, 4)

    win = glfw.create_window(1400, 800, "IMD 3D Memory Viewer", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("Window creation failed")

    glfw.make_context_current(win)
    glfw.swap_interval(1)
    imgui.create_context()
    impl = GlfwRenderer(win, attach_callbacks=False)

    def _mouse_btn(w, b, a, m):
        impl.mouse_button_callback(w, b, a, m)
        if not imgui.get_io().want_capture_mouse:
            cam.on_mouse_button(w, b, a)
    def _scroll(w, dx, dy):
        impl.scroll_callback(w, dx, dy)
        if not imgui.get_io().want_capture_mouse:
            cam.on_scroll(dy)
    glfw.set_mouse_button_callback(win, _mouse_btn)
    def _cursor(w, x, y):
        impl.mouse_callback()
        cam.on_cursor(x, y)
    glfw.set_cursor_pos_callback(win, _cursor)
    glfw.set_scroll_callback(win,       _scroll)

    def on_key(w, key, sc, action, mods):
        if action == glfw.PRESS:
            if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
                glfw.set_window_should_close(w, True)
            if key == glfw.KEY_R:
                cam.reset()
    glfw.set_key_callback(win, on_key)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND | moderngl.DEPTH_TEST)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    pt_prog   = ctx.program(vertex_shader=POINT_VERT, fragment_shader=POINT_FRAG)
    line_prog = ctx.program(vertex_shader=LINE_VERT,  fragment_shader=LINE_FRAG)

    pt_vbo_pos  = ctx.buffer(reserve=12)
    pt_vbo_col  = ctx.buffer(reserve=16)
    pt_vbo_size = ctx.buffer(reserve=4)
    ln_vbo_pos  = ctx.buffer(reserve=12)
    ln_vbo_col  = ctx.buffer(reserve=16)

    pt_vao = ctx.vertex_array(pt_prog, [
        (pt_vbo_pos,  "3f", "in_position"),
        (pt_vbo_col,  "4f", "in_color"),
        (pt_vbo_size, "1f", "in_size"),
    ])
    ln_vao = ctx.vertex_array(line_prog, [
        (ln_vbo_pos, "3f", "in_position"),
        (ln_vbo_col, "4f", "in_color"),
    ])

    n_pts = n_lns = 0
    last_count    = -1
    selected_info = None

    def upload_to_gpu():
        nonlocal n_pts, n_lns
        result = reader.get_gpu_arrays(
            show_stm_nodes=vis["stm_nodes"], show_ltm_nodes=vis["ltm_nodes"],
            show_stm_links=vis["stm_links"], show_ltm_links=vis["ltm_links"],
            show_scm=vis["scm"],             show_spikes=vis["spikes"])
        if result is None:
            n_pts = n_lns = 0
            return 0, 0, 0
        pos, col, siz, lpos, lcol, stm_c, ltm_c = result
        n_pts = len(pos)
        n_lns = len(lpos) // 2

        if n_pts:
            pt_vbo_pos.orphan(pos.nbytes);   pt_vbo_pos.write(pos.tobytes())
            pt_vbo_col.orphan(col.nbytes);   pt_vbo_col.write(col.tobytes())
            pt_vbo_size.orphan(siz.nbytes);  pt_vbo_size.write(siz.tobytes())
        if n_lns:
            ln_vbo_pos.orphan(lpos.nbytes);  ln_vbo_pos.write(lpos.tobytes())
            ln_vbo_col.orphan(lcol.nbytes);  ln_vbo_col.write(lcol.tobytes())
        return stm_c, ltm_c, n_lns

    # Visibility toggles
    vis = {"stm_nodes": True, "ltm_nodes": True, "stm_links": True,
           "ltm_links": True, "scm": True, "spikes": True}

    # Initial load
    reader.refresh()
    stm_c, ltm_c, lnk_c = upload_to_gpu()

    # Background poll
    def poll_loop():
        while not glfw.window_should_close(win):
            time.sleep(POLL_INTERVAL)
            reader.refresh()
    threading.Thread(target=poll_loop, daemon=True).start()

    while not glfw.window_should_close(win):
        glfw.poll_events()

        if cam.pending_click:
            cx, cy = cam.pending_click
            cam.pending_click = None
            if not imgui.get_io().want_capture_mouse:
                fw, fh = glfw.get_framebuffer_size(win)
                hit = reader.pick(cx, cy, cam.proj(fw, fh) @ cam.view(), fw, fh)
                selected_info = reader.lookup_node(hit) if hit else None
                reader.selected_key = hit
                last_count = -1  # force re-upload with new dimming

        cur_count = len(reader.nodes)
        if cur_count != last_count:
            stm_c, ltm_c, lnk_c = upload_to_gpu()
            last_count = cur_count
            glfw.set_window_title(win,
                f"IMD 3D Memory Viewer  |  STM: {stm_c}  LTM: {ltm_c}  Links: {lnk_c}")

        w, h = glfw.get_framebuffer_size(win)
        ctx.viewport = (0, 0, w, h)
        ctx.clear(0.06, 0.06, 0.10)

        mvp = (cam.proj(w, h) @ cam.view()).flatten(order="F").tobytes()

        if n_lns:
            line_prog["mvp"].write(mvp)
            ln_vao.render(moderngl.LINES, vertices=n_lns * 2)

        if n_pts:
            pt_prog["mvp"].write(mvp)
            pt_vao.render(moderngl.POINTS, vertices=n_pts)

        impl.process_inputs()
        imgui.new_frame()

        PANEL_W = 170
        imgui.set_next_window_pos((w - PANEL_W - 10, 10), imgui.Cond_.always)
        imgui.set_next_window_size((PANEL_W, 175), imgui.Cond_.always)
        imgui.begin("Visibility", flags=imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move)
        ch_a, vis["stm_nodes"] = imgui.checkbox("STM Nodes",   vis["stm_nodes"])
        ch_b, vis["ltm_nodes"] = imgui.checkbox("LTM Nodes",   vis["ltm_nodes"])
        imgui.separator()
        ch_c, vis["stm_links"] = imgui.checkbox("STM Links",   vis["stm_links"])
        ch_d, vis["ltm_links"] = imgui.checkbox("LTM Links",   vis["ltm_links"])
        ch_e, vis["scm"]       = imgui.checkbox("SCM Chain",   vis["scm"])
        imgui.separator()
        ch_f, vis["spikes"]    = imgui.checkbox("Dir Spikes",  vis["spikes"])
        imgui.end()

        if ch_a or ch_b or ch_c or ch_d or ch_e or ch_f:
            last_count = -1  # triggers upload_to_gpu() before next render

        if selected_info is not None:
            _draw_info_panel(selected_info)
        imgui.render()
        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(win)

    impl.shutdown()
    glfw.terminate()


if __name__ == "__main__":
    main()
