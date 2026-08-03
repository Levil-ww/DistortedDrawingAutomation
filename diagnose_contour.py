#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPS轮廓提取诊断脚本

用于排查"预览图只有直角矩形边框、没有圆角异形、JPG内容未显示"的问题。
运行后会在 output_diagnosis/ 目录下输出各步骤的中间图像供分析。
"""

import sys
import os
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(message)s')

from PIL import Image, ImageDraw
import numpy as np

from engines.pillow_engine import PillowEngine, get_eps_bbox
from services.design_service import DesignService
from processors.contour_extractor import ContourExtractor
from models import ProcessConfig


def diagnose(eps_file: str, jpg_file: str, out_dir: str = "output_diagnosis"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"EPS轮廓诊断 - 结果输出至: {out.resolve()}")
    print(f"{'='*60}\n")

    # 0. 基础信息
    eps_path = Path(eps_file)
    jpg_path = Path(jpg_file)
    if not eps_path.exists():
        print(f"✗ EPS文件不存在: {eps_path}")
        return
    if not jpg_path.exists():
        print(f"✗ JPG文件不存在: {jpg_path}")
        return

    bbox = get_eps_bbox(eps_path)
    print(f"[信息] EPS BoundingBox: {bbox} cm")
    print(f"[信息] EPS 文件大小: {eps_path.stat().st_size} 字节")

    # 1. 栅格化EPS (使用150 DPI用于预览诊断)
    width_cm, height_cm = bbox if bbox else (140.0, 80.0)
    print(f"\n[步骤1] EPS栅格化 ({width_cm}x{height_cm}cm @ 150dpi)...")
    engine = PillowEngine()
    eps_img = engine.open_eps(eps_path, dpi=150, width_cm=width_cm, height_cm=height_cm)
    eps_img.save(out / "step1_eps_rasterized.png")
    print(f"  ✓ 保存: step1_eps_rasterized.png  ({eps_img.width}x{eps_img.height})")

    # 可视化：统计各通道极值
    arr = np.array(eps_img.convert('L'))
    print(f"  灰度值统计: min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}")
    print(f"  深色像素(<120)比例: {(arr < 120).sum() / arr.size * 100:.2f}%")
    print(f"  白色像素(>250)比例: {(arr > 250).sum() / arr.size * 100:.2f}%")

    # 2. 轮廓提取 - 每一步可视化
    print(f"\n[步骤2] 详细轮廓提取过程可视化...")
    try:
        import cv2

        rgb = np.array(eps_img.convert('RGB'))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        Image.fromarray(binary).save(out / "step2a_threshold_binary.png")
        print(f"  ✓ step2a_threshold_binary.png (背景=255白, 线条=0黑)")

        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=3)
        Image.fromarray(closed).save(out / "step2b_morph_close.png")
        print(f"  ✓ step2b_morph_close.png (闭运算，试图弥合线条缺口)")

        inv = cv2.bitwise_not(closed)
        Image.fromarray(inv).save(out / "step2c_invert.png")
        print(f"  ✓ step2c_invert.png (线条=255白, 背景=0黑)")

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        inv2 = cv2.dilate(inv, kernel_dilate, iterations=2)
        inv2 = cv2.erode(inv2, kernel_dilate, iterations=1)
        Image.fromarray(inv2).save(out / "step2d_dilate_erode.png")
        print(f"  ✓ step2d_dilate_erode.png (线条加粗)")

        contours, hierarchy = cv2.findContours(inv2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  检测到 {len(contours)} 个轮廓 (RETR_CCOMP 模式)")

        # 绘制每个轮廓并保存
        vis = rgb.copy()
        contours_sorted = sorted(enumerate(contours), key=lambda x: cv2.contourArea(x[1]), reverse=True)
        for rank, (orig_idx, cnt) in enumerate(contours_sorted[:10]):
            area = int(cv2.contourArea(cnt))
            x, y, w, h = cv2.boundingRect(cnt)
            total = rgb.shape[0] * rgb.shape[1]
            solidity = area / (w * h) if w * h > 0 else 0
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                     (255, 0, 255), (0, 255, 255)][rank % 6]
            cv2.drawContours(vis, [cnt], -1, color, 2)
            cv2.putText(vis, f"#{rank} A={area}", (x, max(y - 5, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            print(f"    轮廓#{rank}: 面积={area} ({area/total*100:.1f}%画布) "
                  f"矩形=({x},{y},{w},{h}) 实心度(solidity)={solidity:.3f}")
        Image.fromarray(vis).save(out / "step2e_all_contours.png")
        print(f"  ✓ step2e_all_contours.png (前10个轮廓用不同颜色标注)")

        # 对每个大轮廓 fillPoly 并保存蒙版
        for rank, (orig_idx, cnt) in enumerate(contours_sorted[:5]):
            mask_arr = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
            cv2.fillPoly(mask_arr, [cnt], 255)
            Image.fromarray(mask_arr).save(out / f"step2f_mask_contour#{rank}.png")
            area_mask = (mask_arr > 128).sum()
            print(f"  ✓ step2f_mask_contour#{rank}.png  (蒙版填充面积={area_mask})")

        # ===== 关键测试：泛洪法（Alternative B 的原型）=====
        print(f"\n[步骤2-备选] 尝试泛洪法（从四角填充外部区域）...")
        # 线条像素（深色）作为墙
        walls = (gray < 220).astype(np.uint8) * 255
        # 先膨胀一下墙以防止小缝隙漏
        walls = cv2.dilate(walls, np.ones((5, 5), np.uint8), iterations=1)
        Image.fromarray(walls).save(out / "step2alt_a_walls.png")

        mask = np.zeros_like(gray)  # 0=未知
        h, w = gray.shape
        # 从4个角点开始flood fill 标记外部
        mask_flood = walls.copy()
        # 初始种子：四边中点 + 四角
        seeds = [(0,0), (w-1,0), (0,h-1), (w-1,h-1),
                 (w//2, 0), (w//2, h-1), (0, h//2), (w-1, h//2)]
        # 将mask_flood作为"0=可通过，255=墙"，外部区域先用临时值填充
        ff_mask = np.zeros((h+2, w+2), np.uint8)
        outside = np.zeros_like(gray)
        for sx, sy in seeds:
            if walls[sy, sx] == 0:  # 不是墙
                cv2.floodFill(outside, ff_mask, (sx, sy), 255, 10, 10,
                              cv2.FLOODFILL_MASK_ONLY if False else 0)
        # 现在 outside=255 标记外部(包括墙角通过线条开口连通的区域)
        Image.fromarray(outside).save(out / "step2alt_b_outside_flooded.png")

        # 内部 = 非外部 且 非线条
        inside = ((outside == 0) & (walls == 0)).astype(np.uint8) * 255
        # 合并线条本身也算"内部"（蒙版要覆盖到线条上，不然边框丢失）
        inside = cv2.bitwise_or(inside, walls)
        # 再做闭运算以修补小洞
        inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE,
                                   np.ones((15, 15), np.uint8), iterations=2)
        Image.fromarray(inside).save(out / "step2alt_c_inside_mask.png")
        print(f"  ✓ step2alt_c_inside_mask.png  (泛洪法推导出的内部蒙版)")
        flood_area = (inside > 128).sum()
        print(f"    泛洪法蒙版面积: {flood_area} ({flood_area/total*100:.1f}%画布)")

    except ImportError:
        print("  ✗ OpenCV未安装，跳过详细可视化")

    # 3. 调用正式的ContourExtractor并验证
    print(f"\n[步骤3] 正式 ContourExtractor 提取验证...")
    extractor = ContourExtractor()
    levels = extractor.extract_contours(eps_img, num_levels=5)
    print(f"  提取到 {len(levels)} 层轮廓")
    for i, lv in enumerate(levels):
        lv.mask.save(out / f"step3_level{i}_area{lv.area}.png")
        x, y, w, h = lv.bounding_rect
        total = eps_img.width * eps_img.height
        solidity = lv.area / (w * h) if w * h > 0 else 0
        print(f"    Level#{i}: area={lv.area} ({lv.area/total*100:.1f}%) "
              f"rect=({x},{y},{w},{h}) solidity={solidity:.3f}")

    # 4. 加载JPG并渲染
    print(f"\n[步骤4] 渲染测试 (使用正式流程)...")
    service = DesignService(engine=engine)
    cfg = ProcessConfig(
        eps_file=str(eps_path), psd_file=str(jpg_path),
        output_file=str(out / "RENDER_OUTPUT.jpg"),
        canvas_width_cm=width_cm, canvas_height_cm=height_cm, dpi=150,
    )
    layers = engine.load_psd_layers(jpg_path)
    psd_composite = service._prepare_psd_composite(jpg_path, layers)
    print(f"  JPG合成图: {psd_composite.size} mode={psd_composite.mode}")
    psd_composite.save(out / "step4_material_composite.png")

    rendered = service._render_with_psd_composite(eps_img, levels, psd_composite, cfg)
    rendered.save(out / "step4_rendered_level0.jpg", quality=95)
    print(f"  ✓ step4_rendered_level0.jpg (使用 levels[0] 作为蒙版的渲染结果)")

    # 验证：中心10%区域是否全白（如果是，说明蒙版错了）
    rarr = np.array(rendered.convert('L'))
    cy, cx = rarr.shape[0] // 2, rarr.shape[1] // 2
    ry = int(rarr.shape[0] * 0.05)
    rx = int(rarr.shape[1] * 0.05)
    center_region = rarr[cy-ry:cy+ry, cx-rx:cx+rx]
    white_ratio = (center_region > 250).sum() / center_region.size
    print(f"  中心10%区域白色像素占比: {white_ratio*100:.1f}%")
    if white_ratio > 0.95:
        print("  ⚠ 警告: 中心区域几乎全白 → 蒙版可能提取错误（里外搞反或选错轮廓）！")
    else:
        print("  ✓ 中心区域非白色 → JPG内容似乎已正确渲染")

    # 测试：手动用泛洪法得到的mask再渲染一次（如果存在）
    if 'inside' in locals():
        from PIL import Image as PILImage
        flood_mask = PILImage.fromarray(inside, mode='L')
        ys, xs = np.where(inside > 128)
        if len(xs) > 0:
            fx, fy = int(xs.min()), int(ys.min())
            fw, fh = int(xs.max() - fx + 1), int(ys.max() - fy + 1)
            bx, by, bw, bh = fx, fy, fw, fh

            composite = psd_composite.convert("RGBA")
            cw, ch = composite.size
            sx = bw / cw if cw > 0 else 1.0
            sy = bh / ch if ch > 0 else 1.0
            nw, nh = max(1, int(cw * sx)), max(1, int(ch * sy))
            scaled = composite.resize((nw, nh), PILImage.LANCZOS)
            canvas = PILImage.new("RGBA", (eps_img.width, eps_img.height), (255,255,255,255))
            canvas.paste(scaled, (bx, by), scaled)
            ma = np.array(flood_mask)
            mb = ma > 128
            ca = np.array(canvas)
            for c in range(3):
                ca[:,:,c] = np.where(mb, ca[:,:,c], 255)
            ca[:,:,3] = 255
            manual_render = PILImage.fromarray(ca, mode="RGBA").convert("RGB")
            # 叠加CAD边框
            base_arr = np.array(eps_img.convert("RGBA"))
            render_arr = np.array(manual_render.convert("RGBA"))
            is_dark = np.any(base_arr[:,:,:3] < 120, axis=2)
            is_border = is_dark & mb
            if np.any(is_border):
                for c in range(3):
                    render_arr[:,:,c] = np.where(is_border, base_arr[:,:,c], render_arr[:,:,c])
                render_arr[:,:,3] = np.where(is_border, 255, render_arr[:,:,3])
            manual_final = PILImage.fromarray(render_arr, mode="RGBA").convert("RGB")
            manual_final.save(out / "step4_rendered_floodmask.jpg", quality=95)
            print(f"  ✓ step4_rendered_floodmask.jpg (使用泛洪法蒙版手动渲染的结果 - 供对比)")

    print(f"\n{'='*60}")
    print(f"诊断完成！请打开目录查看: {out.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python diagnose_contour.py <your.eps> <your.jpg> [output_dir]")
        print("\n示例:")
        print("  python diagnose_contour.py \"D:/test/模板.eps\" \"D:/test/素材.jpg\"")
    else:
        eps = sys.argv[1]
        jpg = sys.argv[2]
        outd = sys.argv[3] if len(sys.argv) > 3 else "output_diagnosis"
        diagnose(eps, jpg, outd)
