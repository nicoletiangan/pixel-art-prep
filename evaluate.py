"""
Test Plan v1 — Step 3 & 4
Pixel-art conversion + auto-scoring for all 5 categories × 4 styles.

Outputs
-------
testresources/output/pixel-art/<category>/
    original_pixelart.png       — original → pixel art (64 px wide, 8× upscaled)
    <Style>_pixelart.png        — processed style → pixel art
    comparison.png              — side-by-side grid (original + 4 styles)
reports/evaluation-v1.md        — filled evaluation matrix
"""

import os, textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── config ────────────────────────────────────────────────────────────────────
TARGET_W   = 64     # pixel art width (px)
DISPLAY_SC = 8      # upscale factor for saved PNGs
LABEL_H    = 20     # label bar height in comparison grid

CATEGORIES = [
    ('photo-person-1',     'testresources/input/Julie-Sweet-2.jpg'),
    ('photo-landscape-1',  'testresources/input/photo-landscape-1.jpg'),
    ('illust-landscape-1', 'testresources/input/illust-landscape-1.jpg'),
    ('illust-person-1',    'testresources/input/illust-person-1.jpg'),
    ('anime-1',            'testresources/input/anime-1.png'),
]

STYLES = ['Realistic', 'Cartoon', 'Abstract', 'Retro']

STYLE_NAMES_ZH = {
    'Realistic': '写实',
    'Cartoon':   '卡通',
    'Abstract':  '抽象',
    'Retro':     '复古',
}

# ── pixel-art conversion ──────────────────────────────────────────────────────

def to_pixel_art(img: Image.Image, target_w: int = TARGET_W) -> Image.Image:
    """Downscale with LANCZOS, keep aspect ratio; return small image."""
    aspect = img.height / img.width
    target_h = max(1, round(target_w * aspect))
    return img.resize((target_w, target_h), Image.LANCZOS)


def upscale_display(small: Image.Image, scale: int = DISPLAY_SC) -> Image.Image:
    """Upscale with NEAREST for crisp pixel blocks."""
    return small.resize((small.width * scale, small.height * scale), Image.NEAREST)


# ── scoring ───────────────────────────────────────────────────────────────────

def score(pixel_art: Image.Image, original: Image.Image) -> dict:
    """
    Returns 1-5 scores for each dimension.

    色块干净度  — fewer unique colours per pixel → cleaner blocks
    轮廓清晰度  — mean edge gradient magnitude
    色彩自然度  — per-channel histogram overlap with the original's pixel art
    像素化适合度 — weighted combination of the above
    """
    arr = np.array(pixel_art.convert('RGB'), dtype=np.float32)
    H, W = arr.shape[:2]
    total = H * W

    # --- cleanliness: unique-colour density (lower is cleaner) ---------------
    flat = arr.reshape(-1, 3).astype(np.uint8)
    unique = len({tuple(p) for p in flat})
    density = unique / total                        # 0..1 (lower = cleaner)
    # map: density 0 → 5,  density ≥ 0.5 → 1
    clean = max(1.0, min(5.0, 5.0 - density * 8.0))

    # --- edge clarity: mean absolute gradient --------------------------------
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()
    edge_mag = (gx + gy) / 2                        # ~0..40 typical range
    # map: 0 → 1,  ≥ 20 → 5
    edge = max(1.0, min(5.0, 1.0 + edge_mag / 5.0))

    # --- colour naturalness: histogram overlap with original -----------------
    orig_pa  = np.array(to_pixel_art(original).convert('RGB'), dtype=np.float32)
    overlap_sum = 0.0
    for ch in range(3):
        h1, _ = np.histogram(arr[:,:,ch].ravel(),      bins=32, range=(0,255), density=True)
        h2, _ = np.histogram(orig_pa[:,:,ch].ravel(),  bins=32, range=(0,255), density=True)
        overlap_sum += np.minimum(h1, h2).sum() / 32   # 0..1
    naturalness = max(1.0, min(5.0, 1.0 + (overlap_sum / 3) * 4.0))

    # --- pixel-art suitability: weighted combination -------------------------
    suitability = max(1.0, min(5.0,
        0.5 * clean + 0.3 * edge + 0.2 * naturalness
    ))

    return {
        'clean':       round(clean,       1),
        'edge':        round(edge,        1),
        'naturalness': round(naturalness, 1),
        'suitability': round(suitability, 1),
    }


def fmt(v: float) -> str:
    """Round to nearest 0.5 for table display."""
    return str(round(v * 2) / 2)


# ── comparison grid ───────────────────────────────────────────────────────────

def make_comparison(images: dict, labels: list) -> Image.Image:
    """
    images : {label: PIL.Image (upscaled display)}
    labels : ordered list of keys
    Returns a horizontal strip with text labels.
    """
    imgs = [images[l] for l in labels]
    # normalise heights (keep aspect; pad to tallest)
    max_h = max(im.height for im in imgs)
    padded = []
    for im in imgs:
        if im.height < max_h:
            canvas = Image.new('RGB', (im.width, max_h), (30, 30, 30))
            canvas.paste(im, (0, (max_h - im.height) // 2))
            padded.append(canvas)
        else:
            padded.append(im)

    total_w = sum(im.width for im in padded) + 2 * (len(padded) - 1)
    strip = Image.new('RGB', (total_w, max_h + LABEL_H), (20, 20, 20))
    draw  = ImageDraw.Draw(strip)

    x = 0
    for im, label in zip(padded, labels):
        strip.paste(im, (x, LABEL_H))
        draw.text((x + 4, 3), label, fill=(220, 220, 80))
        x += im.width + 2

    return strip


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}   # cat → {style → scores}

    for cat, src_path in CATEGORIES:
        print(f'\n[{cat}]')
        if not os.path.exists(src_path):
            print(f'  SKIP — source not found: {src_path}')
            continue

        original = Image.open(src_path).convert('RGB')
        out_dir  = os.path.join('testresources', 'output', 'pixel-art', cat)
        os.makedirs(out_dir, exist_ok=True)

        # original pixel art
        orig_pa = to_pixel_art(original)
        orig_pa_disp = upscale_display(orig_pa)
        orig_pa_disp.save(os.path.join(out_dir, 'original_pixelart.png'))
        print(f'  original  ({original.width}×{original.height}) → pixelart {orig_pa.width}×{orig_pa.height}')

        images_for_grid = {'原图': orig_pa_disp}
        cat_scores = {}

        for style in STYLES:
            # find processed output (named after source basename)
            src_base  = os.path.splitext(os.path.basename(src_path))[0]
            proc_path = os.path.join('testresources', 'output', cat,
                                     f'{src_base}_{style}.png')
            if not os.path.exists(proc_path):
                print(f'  {style}: SKIP — {proc_path} not found')
                continue

            processed = Image.open(proc_path).convert('RGB')
            proc_pa   = to_pixel_art(processed)
            proc_pa_disp = upscale_display(proc_pa)
            proc_pa_disp.save(os.path.join(out_dir, f'{style}_pixelart.png'))

            s = score(proc_pa, original)
            cat_scores[style] = s
            label = STYLE_NAMES_ZH.get(style, style)
            print(f'  {style:<10}  clean={s["clean"]}  edge={s["edge"]}  '
                  f'natural={s["naturalness"]}  suit={s["suitability"]}')

            images_for_grid[label] = proc_pa_disp

        # comparison grid
        grid = make_comparison(images_for_grid, ['原图'] + [STYLE_NAMES_ZH[s] for s in STYLES if s in cat_scores])
        grid.save(os.path.join(out_dir, 'comparison.png'))
        print(f'  → comparison.png saved')

        # baseline: original → pixel art score (vs itself)
        orig_s = score(orig_pa, original)
        results[cat] = {'_original': orig_s, **cat_scores}

    # ── report ─────────────────────────────────────────────────────────────
    os.makedirs('reports', exist_ok=True)
    report_path = os.path.join('reports', 'evaluation-v1.md')

    lines = ['# 评估结果 v1\n',
             f'*生成日期：2026-04-04  /  目标像素宽度：{TARGET_W}px  /  工具：Python PIL 最近邻*\n',
             '---\n']

    cat_labels = {
        'photo-person-1':     '真人照片',
        'photo-landscape-1':  '风景照片',
        'illust-landscape-1': '风景画/插画',
        'illust-person-1':    '人物画像',
        'anime-1':            '卡通/动漫',
    }

    passing = 0   # categories where ≥1 style beats original

    for cat, src_path in CATEGORIES:
        if cat not in results:
            continue
        label = cat_labels.get(cat, cat)
        lines.append(f'## {label}（`{cat}`）\n')
        lines.append(f'> 原图直接像素化基线：色块={fmt(results[cat]["_original"]["clean"])}  '
                     f'轮廓={fmt(results[cat]["_original"]["edge"])}  '
                     f'色彩={fmt(results[cat]["_original"]["naturalness"])}  '
                     f'**适合度={fmt(results[cat]["_original"]["suitability"])}**\n')
        lines.append('| 风格 | 色块干净 | 轮廓清晰 | 色彩自然 | 像素化适合 | vs 原图 |')
        lines.append('|------|---------|---------|---------|-----------|---------|')

        orig_suit = results[cat]['_original']['suitability']
        best_suit = 0.0
        best_style = '—'

        for style in STYLES:
            if style not in results[cat]:
                continue
            s = results[cat][style]
            suit = s['suitability']
            delta = suit - orig_suit
            vs = f'+{delta:.1f}' if delta > 0 else f'{delta:.1f}'
            lines.append(f'| {STYLE_NAMES_ZH[style]} | {fmt(s["clean"])} | {fmt(s["edge"])} | '
                         f'{fmt(s["naturalness"])} | {fmt(s["suitability"])} | {vs} |')
            if suit > best_suit:
                best_suit  = suit
                best_style = STYLE_NAMES_ZH[style]

        beats = best_suit > orig_suit
        if beats:
            passing += 1
        verdict = '✅ 有改善' if beats else '❌ 无改善'
        lines.append(f'\n**最佳风格：{best_style}  /  {verdict}**\n')
        lines.append('')

    lines.append('---\n')
    lines.append('## 总体结论\n')
    lines.append(f'**{passing}/5 类图片** 中至少1种风格优于原图直接像素化。\n')
    if passing >= 4:
        lines.append('> ✅ **算法稳定，可进入阶段二。**')
    elif passing >= 3:
        lines.append('> 🔧 **局部调整参数，复测后再判断。**')
    else:
        lines.append('> ❌ **超过2类无改善，需重新审视算法。**')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f'\n✓ Report → {report_path}')
    print(f'✓ {passing}/5 categories improved over baseline')


if __name__ == '__main__':
    main()
