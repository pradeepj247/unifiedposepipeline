# Emoji Legend and Color Mapping

Referenceable, numbered emoji and color mappings for logs and UI.
Tell the assistant e.g. "use emoji 3 for file-found messages".

Each entry: Number. Emoji — Short label — Hex color — ANSI (terminal) suggestion

1. ✅ — Success — #28A745 — `\033[92m`
2. 🚀 — Step / Start — #0D6EFD — `\033[94m`
3. 🔍 — Found / Present — #20C997 — `\033[96m`
4. ❌ — Missing / Fail — #DC3545 — `\033[91m`
5. ⬇️ — Downloading / In-progress — #FD7E14 — `\033[33m`
6. ⚠️ — Warning / Attention — #FFC107 — `\033[93m`
7. ⏱️ — Performance / Timing — #6C757D — `\033[90m`
8. 💡 — Tip / Note — #17A2B8 — `\033[96m`
9. 📁 — File / Saved — #0D6EFD — `\033[94m`
10. 🧾 — Checkpoint / Summary — #6610F2 — `\033[95m`
11. ✔️ — Completed / Done — #198754 — `\033[92m`
12. 🛠️ — Action / Execute — #6F42C1 — `\033[95m`
13. 👤 — Person / ReID / ID — #E83E8C — `\033[95m`
14. ⚡ — Speed / Fast — #FFC107 — `\033[93m`
15. 💥 — Error / Crash — #DC3545 — `\033[91m`
16. ❓ — Question / Prompt — #0DCFF1 — `\033[96m`
17. 📌 — Important / Pin — #6610F2 — `\033[95m`
18. 🔄 — Retry / Sync — #FD7E14 — `\033[33m`
19. 🌸 — Friendly Found (alt) — #20C997 — `\033[96m`
20. 📊 — Stats / Metrics — #6C757D — `\033[90m`

## Usage examples

- Step header: `2. 🚀 STEP 2: Download Model Files` (use emoji 2)
- Model present: `3. ✅ YOLOv8s already exists: /path (21.5 MB)` (use emoji 1 or 3)
- Model missing + download:
  - `4. ❌ OSNet x0.25 (ONNX) not found` (emoji 4)
  - `5. ⬇️ Downloading OSNet x0.25 (ONNX) (~2 MB)` (emoji 5)
- Progress/Perf: `7. ⏱️ Detection FPS: 59.5 (16.8ms/frame)` (emoji 7)
- Checkpoint summary: `10. 🧾 Checkpoint saved to current_context.md` (emoji 10)

## Terminal color hints

Wrap messages with the ANSI code then reset: e.g.
```
print(f"\033[92m✅ Success: operation completed\033[0m")
```

## How to refer
- Tell the assistant: "Use emoji 4 for missing-file errors" or "Use emoji 5 when downloading".

If you want additions or remapping, tell me the number(s) to change.
