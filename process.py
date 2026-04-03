"""
像素画前置处理工具 — CLI 批处理脚本
4种风格：写实 / 卡通 / 抽象 / 复古

用法：
  python process.py <输入图片> <输出目录> [--levels N]
"""

import sys, os, argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ─────────────────────────────────────────────
# 基础算法
# ─────────────────────────────────────────────

def posterize(arr, levels):
    levels = max(2, min(8, levels))
    step = 255 / (levels - 1)
    return np.round(np.round(arr.astype(float) / step) * step).clip(0, 255).astype(np.uint8)


def rgb_to_hsv(arr):
    f = arr.astype(np.float32) / 255.0
    r, g, b = f[:,:,0], f[:,:,1], f[:,:,2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc
    v = maxc
    s = np.where(maxc > 0, diff / maxc, 0.0)
    h = np.zeros_like(r)
    mask = diff > 0
    mr = mask & (maxc == r)
    mg = mask & (maxc == g)
    mb = mask & (maxc == b)
    h[mr] = (60 * (g[mr] - b[mr]) / diff[mr]) % 360
    h[mg] = 60 * (b[mg] - r[mg]) / diff[mg] + 120
    h[mb] = 60 * (r[mb] - g[mb]) / diff[mb] + 240
    return np.stack([h, s, v], axis=2)


def hsv_to_rgb(hsv):
    h, s, v = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    h6 = h / 60.0
    i  = np.floor(h6).astype(int) % 6
    f  = h6 - np.floor(h6)
    p  = v * (1 - s)
    q  = v * (1 - f * s)
    t  = v * (1 - (1 - f) * s)
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    for idx, (rv, gv, bv) in enumerate([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]):
        m = i == idx
        rgb[:,:,0][m] = rv[m]
        rgb[:,:,1][m] = gv[m]
        rgb[:,:,2][m] = bv[m]
    return (rgb * 255).clip(0, 255).astype(np.uint8)


def posterize_hsv(arr, h_levels=12, s_levels=4, v_levels=6):
    hsv = rgb_to_hsv(arr)
    if h_levels > 1:
        step = 360 / h_levels
        hsv[:,:,0] = (np.round(hsv[:,:,0] / step) * step) % 360
    for ch, lv in [(1, s_levels), (2, v_levels)]:
        if lv > 1:
            step = 1 / (lv - 1)
            hsv[:,:,ch] = np.round(hsv[:,:,ch] / step) * step
    return hsv_to_rgb(hsv)


def pixelate(arr, block_size):
    h, w = arr.shape[:2]
    out = arr.copy()
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            avg = block.mean(axis=(0, 1)).astype(np.uint8)
            out[y:y+block_size, x:x+block_size] = avg
    return out


def gaussian_blur(arr, radius):
    if radius < 1:
        return arr.copy()
    img = Image.fromarray(arr)
    for _ in range(3):
        img = img.filter(ImageFilter.BoxBlur(radius))
    return np.array(img)


def median_filter(arr, radius):
    if radius < 1:
        return arr.copy()
    img = Image.fromarray(arr)
    return np.array(img.filter(ImageFilter.MedianFilter(size=radius * 2 + 1)))


def bilateral_filter(arr, radius, sigma_color):
    if radius < 1:
        return arr.copy()
    h, w = arr.shape[:2]
    src = arr.astype(np.float32)
    sigma_space2 = 2 * radius ** 2
    sigma_color2 = 2 * sigma_color ** 2
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    spatial_w = np.exp(-(xx**2 + yy**2) / sigma_space2)
    pad = radius
    padded = np.pad(src, ((pad,pad),(pad,pad),(0,0)), mode='reflect')
    dst   = np.zeros_like(src)
    sum_w = np.zeros((h, w, 1), dtype=np.float32)
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            sw       = spatial_w[dy+radius, dx+radius]
            neighbor = padded[pad+dy:pad+dy+h, pad+dx:pad+dx+w]
            cd       = np.sum((neighbor - src)**2, axis=2, keepdims=True)
            w_       = sw * np.exp(-cd / sigma_color2)
            dst   += neighbor * w_
            sum_w += w_
    return (dst / sum_w).clip(0, 255).astype(np.uint8)


def adjust_saturation(arr, factor):
    img = Image.fromarray(arr)
    return np.array(ImageEnhance.Color(img).enhance(factor))


def adjust_contrast(arr, factor):
    img = Image.fromarray(arr)
    return np.array(ImageEnhance.Contrast(img).enhance(factor))


def warm_shift(arr):
    """轻微暖色偏移：红通道+5%，蓝通道-8%"""
    out = arr.astype(np.float32).copy()
    out[:,:,0] = np.clip(out[:,:,0] * 1.05, 0, 255)   # R up
    out[:,:,2] = np.clip(out[:,:,2] * 0.92, 0, 255)   # B down
    return out.astype(np.uint8)


# ─────────────────────────────────────────────
# 4种风格
# ─────────────────────────────────────────────

def style_realistic(arr, levels):
    """写实：轻双边去噪，保留色彩细节"""
    out = bilateral_filter(arr, radius=2, sigma_color=25)
    out = posterize(out, levels + 2)          # 多保留几级色阶
    return out


def style_cartoon(arr, levels):
    """卡通：自适应描边（边缘暗化0.35）+ HSV色阶保色相"""
    from scipy.ndimage import convolve, binary_dilation
    smoothed = bilateral_filter(arr, radius=3, sigma_color=50)
    colored  = posterize_hsv(smoothed, h_levels=12, s_levels=4, v_levels=6)
    gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.float32)
    kx   = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    gx   = convolve(gray, kx)
    gy   = convolve(gray, kx.T)
    edge = np.sqrt(gx**2 + gy**2) > 80
    edge = binary_dilation(edge, structure=np.ones((2,2), dtype=bool))
    out  = colored.astype(np.float32)
    out[edge] = out[edge] * 0.35
    return out.clip(0, 255).astype(np.uint8)


def style_abstract(arr, levels):
    """抽象：几何块（30px像素化）+ HSV色阶"""
    out = gaussian_blur(arr, radius=2)
    out = pixelate(out, block_size=30)
    out = posterize_hsv(out, h_levels=8, s_levels=3, v_levels=4)
    return out


def style_retro(arr, levels):
    """复古：暖色调，HSV色阶保轮廓清晰，低饱和度营造褪色感"""
    out = gaussian_blur(arr, radius=2)
    out = warm_shift(out)
    out = posterize_hsv(out, h_levels=12, s_levels=4, v_levels=6)
    out = adjust_saturation(out, 0.70)
    return out


STYLES = [
    ('Realistic', 'Realistic', style_realistic),
    ('Cartoon',   'Cartoon',   style_cartoon),
    ('Abstract',  'Abstract',  style_abstract),
    ('Retro',     'Retro',     style_retro),
]


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Pixel art pre-processor')
    parser.add_argument('input',  help='Input image path')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('--levels', type=int, default=4,
                        help='Base posterize levels 2-8 (default 4)')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f'Error: file not found: {args.input}')
        sys.exit(1)

    img = Image.open(args.input).convert('RGB')
    arr = np.array(img)
    print(f'Input : {args.input}  ({img.width} x {img.height}px)')
    print(f'Levels: {args.levels} (base)')
    print()

    os.makedirs(args.output, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    for label, tag, fn in STYLES:
        print(f'[{tag}] processing...', end=' ', flush=True)
        result   = fn(arr.copy(), args.levels)
        out_path = os.path.join(args.output, f'{base}_{tag}.png')
        Image.fromarray(result).save(out_path)
        print(f'-> {os.path.basename(out_path)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
