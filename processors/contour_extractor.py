#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层轮廓提取模块

从EPS栅格化图像中提取多层嵌套闭合轮廓，为每个轮廓生成蒙版。
支持轮廓腐蚀/膨胀以匹配PSD各层的边框间距。

核心思路：
1. 检测外轮廓 → 生成基础蒙版
2. 通过形态学腐蚀/膨胀生成多级轮廓蒙版
3. 每层PSD素材使用对应的轮廓蒙版进行裁剪
   → 各层自然跟随CAD轮廓的弧度和角度
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ContourLevel:
    """单层轮廓信息"""
    level: int                    # 层级索引（0=最外层）
    mask: Image.Image             # 该轮廓的蒙版 (L模式, 255=填充区)
    bounding_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int                     # 轮廓面积（像素）
    hierarchy: int = 0            # 层次深度


class ContourExtractor:
    """多层轮廓提取器"""

    def __init__(self, max_levels: int = 10, min_area_ratio: float = 0.01):
        self.max_levels = max_levels
        self.min_area_ratio = min_area_ratio

    def extract_contours(self, base_img: Image.Image,
                       num_levels: int = 5) -> List[ContourLevel]:
        """从EPS栅格化图像提取多层嵌套轮廓

        策略（已增强v3）：
        1. 优先提取真实轮廓（通过fillPoly检测）
           ＋智能选择：排除"实心度>0.98且占画布>90%"的外裁切直角框，
             优先选择实心度较低的圆角异形轮廓
        2. 若真实轮廓不足，通过线条膨胀检测更多
        3. 若仍不足，基于最外层轮廓系统腐蚀生成虚拟轮廓
        4. ★新增：如果上述都失败（或最终第0层实心度过高像直角框），
           自动回退到【泛洪法】：将CAD线条当墙，从四周泛洪识别内部区域
        5. 最终返回不超过num_levels层的轮廓

        Args:
            base_img: EPS栅格化图像
            num_levels: 期望的轮廓层数

        Returns:
            按面积从大到小排序的轮廓列表
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV未安装，使用单层轮廓回退")
            single = self._extract_single_contour(base_img)
            if len(single) == 1 and num_levels > 1:
                return self._generate_virtual_contours(base_img, single[0], num_levels)
            return single

        total_area = base_img.width * base_img.height

        # 1. 尝试真实轮廓提取 + 智能选择
        raw_levels = self._extract_with_cv2(base_img)
        logger.info(f"真实轮廓提取: {len(raw_levels)} 层")

        # 智能挑选：优先选择"不是外层裁切直角框"的轮廓
        levels = self._smart_select_product_contour(raw_levels, total_area)
        if levels and levels is not raw_levels:
            logger.info("智能轮廓选择：已优先挑选圆角异形成品轮廓（跳过外层直角裁切框）")

        # 2. 若真实轮廓太少，尝试线条检测
        if len(levels) < num_levels:
            logger.info(f"真实轮廓不足，尝试线条膨胀检测...")
            line_levels = self._extract_line_nested_contours(base_img)
            line_levels = self._smart_select_product_contour(line_levels, total_area)
            if len(line_levels) > len(levels):
                logger.info(f"线条检测: {len(line_levels)} 层")
                levels = line_levels

        # 3. ★ 检查第0层是否合理（不是纯直角框）
        #    如果第0层的实心度太高（>0.985）或者中心区域蒙版不在内部，
        #    自动启用泛洪法重新提取
        if levels:
            top = levels[0]
            bx, by, bw, bh = top.bounding_rect
            solidity = top.area / (bw * bh) if bw * bh > 0 else 1.0
            canvas_ratio = top.area / total_area
            center_ok = self._check_mask_contains_center(top.mask, base_img.width, base_img.height)
            need_flood = False
            if solidity > 0.985:
                logger.info(f"第0层实心度={solidity:.3f}过高，疑似外层直角裁切框")
                need_flood = True
            if not center_ok:
                logger.info("第0层蒙版未完整覆盖画布中心区域，疑似里外搞反")
                need_flood = True
            if need_flood:
                flood_level = self._extract_by_flood_fill(base_img)
                if flood_level is not None:
                    fb, fs = flood_level.bounding_rect, flood_level.area
                    flood_solidity = fs / (fb[2]*fb[3]) if (fb[2]*fb[3]) > 0 else 1
                    logger.info(f"泛洪法回退: 面积={fs} 实心度={flood_solidity:.3f} "
                                f"(原第0层实心度={solidity:.3f})")
                    # 将泛洪法结果作为新的第0层，原来的移到后面
                    combined = [flood_level] + [l for l in levels if abs(l.area - fs)/max(fs,1) > 0.1]
                    levels = combined[:num_levels]
        else:
            # 完全没找到轮廓，直接用泛洪法
            flood_level = self._extract_by_flood_fill(base_img)
            if flood_level is not None:
                logger.info("回退：无轮廓，使用泛洪法蒙版作为第0层")
                levels = [flood_level]

        # 4. 若仍不足，基于第0层生成虚拟轮廓补充
        if len(levels) < num_levels and len(levels) >= 1:
            needed = num_levels - len(levels)
            logger.info(f"需要{needed}层虚拟轮廓补充...")
            base_contour = levels[0]
            real_areas = {c.area for c in levels}
            virtual = self._generate_virtual_contours(
                base_img, base_contour, num_levels
            )
            for v in virtual:
                is_new = True
                for ea in real_areas:
                    if abs(ea - v.area) / max(ea, 1) < 0.08:
                        is_new = False
                        break
                if is_new and len(levels) < num_levels:
                    levels.append(v)
                    real_areas.add(v.area)

        # 5. 按面积排序并限制层数
        levels.sort(key=lambda c: c.area, reverse=True)
        for i, m in enumerate(levels):
            m.level = i

        if len(levels) > num_levels:
            levels = levels[:num_levels]

        # ===== v3最终安全网：再校验一次第0层是否正常，否则尝试泛洪法 =====
        if levels:
            top = levels[0]
            bx, by, bw, bh = top.bounding_rect
            solidity = top.area / max(bw * bh, 1)
            canvas_ratio = top.area / max(base_img.width * base_img.height, 1)
            center_ok = self._check_mask_contains_center(top.mask, base_img.width, base_img.height)
            looks_bad = (solidity > 0.985 and canvas_ratio > 0.90) or (not center_ok)
            if looks_bad:
                logger.warning(f"[extract_contours] 最终第0层疑似问题 (solidity={solidity:.4f}, canvas={canvas_ratio:.3f}, center_ok={center_ok})，尝试泛洪法")
                flood = self._extract_by_flood_fill(base_img)
                if flood is not None:
                    new_list = [flood] + [l for l in levels if l.area != flood.area]
                    for i, m in enumerate(new_list):
                        m.level = i
                    levels = new_list[:num_levels]

        logger.info(f"最终轮廓: {len(levels)} 层")
        if levels:
            bx, by, bw, bh = levels[0].bounding_rect
            sol = levels[0].area / max(bw*bh, 1)
            logger.info(f"  第0层: 面积={levels[0].area}, 矩形=({bx},{by},{bw},{bh}), 实心度={sol:.3f}")
        return levels

    def _extract_with_cv2(self, base_img: Image.Image) -> List[ContourLevel]:
        """使用OpenCV提取真实轮廓

        使用RETR_CCOMP获取两级层次结构（外层+内层）。
        过滤掉面积过小的轮廓，并确保去重。

        优化v4：减小形态学参数，避免破坏EPS圆角异形轮廓。
        """
        import cv2

        arr = np.array(base_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 二值化：背景变白(255)，CAD线框/图形变黑(0)
        # 使用更高阈值(230)以捕获更多细线条
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

        # 形态学闭运算：使用3x3小核，迭代1次即可弥合微小缺口
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # 反相：线框变白(255)，背景变黑(0)
        inv = cv2.bitwise_not(closed)

        # 仅做轻度膨胀（1次3x3），不做腐蚀
        # 避免过度处理破坏圆角异形结构
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        inv = cv2.dilate(inv, kernel_dilate, iterations=1)

        # 使用RETR_CCOMP获取两级层次结构
        contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._extract_single_contour(base_img)

        total_area = base_img.width * base_img.height
        levels: List[ContourLevel] = []
        seen_areas = set()

        # 按面积排序
        sorted_contours = sorted(
            enumerate(contours),
            key=lambda x: cv2.contourArea(x[1]),
            reverse=True
        )

        for rank, (orig_idx, contour) in enumerate(sorted_contours):
            area = int(cv2.contourArea(contour))
            if area < total_area * self.min_area_ratio:
                continue
            if len(levels) >= self.max_levels:
                break

            # 去重
            is_dup = any(abs(sa - area) / max(sa, 1) < 0.05 for sa in seen_areas)
            if is_dup:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # 创建蒙版
            mask_arr = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
            cv2.fillPoly(mask_arr, [contour], 255)

            # 形态学开运算平滑边缘
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_arr = cv2.morphologyEx(mask_arr, cv2.MORPH_OPEN, kernel_open, iterations=1)

            # 验证蒙版有效性
            mask_pixels = np.sum(mask_arr > 128)
            if mask_pixels < total_area * 0.005:
                continue

            mask_img = Image.fromarray(mask_arr, mode="L")
            seen_areas.add(area)

            levels.append(ContourLevel(
                level=len(levels),
                mask=mask_img,
                bounding_rect=(x, y, w, h),
                area=area,
                hierarchy=rank,
            ))

        if not levels:
            return self._extract_single_contour(base_img)

        # ================ v3新增：智能排序 + 异常检测 + 泛洪法回退 ================
        total_area2 = base_img.width * base_img.height
        levels = self._smart_select_product_contour(levels, total_area2)

        logger.info(f"提取到 {len(levels)} 层真实轮廓 (CCOMP后排序): "
                     f"面积 {levels[0].area}~{levels[-1].area} 像素")
        if levels:
            top = levels[0]
            bx, by, bw, bh = top.bounding_rect
            solidity = top.area / (bw * bh) if bw * bh > 0 else 1.0
            center_ok = self._check_mask_contains_center(top.mask, base_img.width, base_img.height)
            logger.info(f"  第0层: area={top.area}, rect=({bx},{by},{bw},{bh}), solidity={solidity:.4f}, center_ok={center_ok}")
            # 异常诊断：实心度过高 (>0.985) 且 占画布过大 (>90%) OR 中心没被覆盖 → 启用泛洪法
            canvas_ratio = top.area / max(total_area2, 1)
            if (solidity > 0.985 and canvas_ratio > 0.90) or (not center_ok):
                logger.warning(f"  [!] 轮廓疑似外层直角裁切框或里外反 (solidity={solidity:.4f}, center_ok={center_ok})，启用泛洪法回退")
                flood_level = self._extract_by_flood_fill(base_img)
                if flood_level is not None:
                    # 将泛洪法结果放在第0位，原第0位往后挪（保留做参考层）
                    new_list = [flood_level] + levels
                    for i, m in enumerate(new_list):
                        m.level = i
                    return new_list
                else:
                    logger.warning("  泛洪法也失败，保留原轮廓结果")

        return levels

    # ---------- 新增v3: 智能轮廓选择 & 泛洪法蒙版 ----------

    def _smart_select_product_contour(self, levels: List[ContourLevel],
                                       total_image_area: int) -> List[ContourLevel]:
        """智能挑选"成品圆角异形"轮廓，排除外层直角裁切框。

        增强v2策略：
        1. 计算每个轮廓的"圆角程度"——通过比较面积与包围盒面积的比值
           矩形框实心度≈1.0，圆角色块实心度<0.95
        2. 检查轮廓的"角点数量"——矩形框有4个明显角点，圆角轮廓有更多
        3. 综合评分：优先选择圆角异形轮廓
        """
        if not levels:
            return levels

        scored_levels = []
        for level in levels:
            bx, by, bw, bh = level.bounding_rect
            solidity = level.area / (bw * bh) if bw * bh > 0 else 1.0
            canvas_ratio = level.area / total_image_area if total_image_area > 0 else 1.0

            # 计算圆角分数：实心度越低，圆角程度越高
            # 矩形框 solidity > 0.97, 圆角轮廓 solidity 通常 < 0.95
            roundness_score = max(0, 1.0 - solidity)

            # 计算角点分数：通过轮廓周长与面积的关系
            # 矩形框的周长/面积比低于圆角形状
            corner_score = self._estimate_roundness(level.mask)

            # 综合分数：roundness_weight * roundness + corner_weight * corner
            # 同时考虑面积占比（太小的不要）
            size_factor = min(1.0, canvas_ratio * 3.0)  # 太小的降权
            score = (roundness_score * 0.5 + corner_score * 0.5) * size_factor

            scored_levels.append((score, level, solidity, canvas_ratio))

        # 按分数排序，优先选择圆角异形
        scored_levels.sort(key=lambda x: x[0], reverse=True)

        best_score, best_level, best_solidity, best_canvas = scored_levels[0]

        # 如果最优轮廓是明显的矩形框（实心度>0.975且面积占比大），尝试找更好的
        if best_solidity > 0.975 and best_canvas > 0.50:
            logger.info(f"最优轮廓疑似矩形框 (solidity={best_solidity:.4f}, canvas={best_canvas:.3f})，尝试寻找圆角轮廓")
            for score, level, sol, cr in scored_levels[1:]:
                # 找一个实心度明显更低的候选
                if sol < 0.95 and cr > 0.05:
                    logger.info(f"  找到更合适的圆角轮廓: solidity={sol:.4f}, canvas={cr:.3f}, score={score:.3f}")
                    new_levels = [level] + [l for l in levels if l is not level]
                    for i, m in enumerate(new_levels):
                        m.level = i
                    return new_levels

        # 如果有圆角轮廓，把最优的放在第0位
        if best_score > 0.01:
            # 重新排序：最优的放最前
            if best_level != levels[0]:
                new_levels = [best_level] + [l for l in levels if l is not best_level]
                for i, m in enumerate(new_levels):
                    m.level = i
                logger.info(f"智能轮廓选择: 将最优轮廓(score={best_score:.3f})提到第0位")
                return new_levels

        return levels

    def _estimate_roundness(self, mask: Image.Image) -> float:
        """估算蒙版的圆角程度（0~1）。

        通过分析蒙版的边界特征：
        - 计算蒙版面积与包围盒面积的比值
        - 检查四个角点区域是否被蒙版填充（矩形框四个角都是实心的）
        - 分析边界的直线程度
        """
        import numpy as np
        try:
            arr = np.array(mask)
            h, w = arr.shape[:2]
            mask_bin = (arr > 128)

            # 计算面积比
            total = h * w
            area = np.sum(mask_bin)
            if total == 0 or area == 0:
                return 0.0

            # 检查四个角点区域（每个角点占5%x5%的区域）
            corner_size_x = max(3, int(w * 0.05))
            corner_size_y = max(3, int(h * 0.05))

            tl_corner = np.sum(mask_bin[:corner_size_y, :corner_size_x])
            tr_corner = np.sum(mask_bin[:corner_size_y, w - corner_size_x:])
            bl_corner = np.sum(mask_bin[h - corner_size_y:, :corner_size_x])
            br_corner = np.sum(mask_bin[h - corner_size_y:, w - corner_size_x:])

            corner_total = corner_size_x * corner_size_y * 4
            corner_filled = tl_corner + tr_corner + bl_corner + br_corner
            corner_ratio = corner_filled / max(corner_total, 1)

            # 矩形框的角点区域几乎完全填充，圆角轮廓的角点区域填充率低
            # corner_ratio > 0.9 表示角点填充率高（矩形框），分数低
            # corner_ratio < 0.6 表示角点填充率低（圆角轮廓），分数高
            corner_score = 1.0 - corner_ratio

            # 综合面积比
            area_ratio = area / total
            # 矩形框面积比高（>0.95），圆角轮廓低
            roundness_from_area = max(0, 1.0 - area_ratio / 0.95)

            return (corner_score * 0.6 + roundness_from_area * 0.4)
        except Exception:
            return 0.0

    def _check_mask_contains_center(self, mask: Image.Image,
                                     img_w: int, img_h: int) -> bool:
        """检查蒙版是否覆盖画布中心（中心10%区域的大多数像素应在蒙版内）。

        返回False意味着蒙版很可能里外搞反了。
        """
        arr = np.array(mask)
        cx, cy = img_w // 2, img_h // 2
        rx = max(2, int(img_w * 0.05))
        ry = max(2, int(img_h * 0.05))
        region = arr[cy - ry:cy + ry, cx - rx:cx + rx]
        if region.size == 0:
            return True
        inside_ratio = (region > 128).sum() / region.size
        # 中心10%区域至少60%应在蒙版内
        return inside_ratio >= 0.60

    def _extract_by_flood_fill(self, base_img: Image.Image) -> Optional[ContourLevel]:
        """【泛洪法】将CAD深色线条当作"墙"，从画布四周向外泛洪，
        未被泛洪到的区域就是"线条包围的内部区域"。

        对纯描边无填充的CAD图特别有效，不依赖轮廓是否闭合。

        优化v4：
        - 使用更小的阈值(200)捕获更多CAD线条
        - 膨胀核减小到3x3，避免所有边缘种子点都变成墙
        - 增加更多种子点（16个）均匀分布在边框区域
        - 放宽面积判断：允许面积较大的内部区域（EPS异形区域可能较大）
        """
        try:
            import cv2
        except ImportError:
            return None

        arr = np.array(base_img.convert("RGB"))
        h, w = arr.shape[:2]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 1. 生成"墙"：深色线条像素（CAD线稿）
        #    阈值200可捕获大部分CAD线稿（包括较浅的灰色线条）
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # 小膨胀：3x3核膨胀1次，弥合1像素的小缺口但不改变大形状
        wall_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        walls = cv2.dilate(th, wall_kernel, iterations=1)

        # 2. 检查边缘种子点是否可用
        #    如果大部分边缘点都是墙，说明需要调整策略
        edge_points = []
        for i in range(w):
            if walls[0, i] == 0:
                edge_points.append((i, 0))
            if walls[h - 1, i] == 0:
                edge_points.append((i, h - 1))
        for i in range(h):
            if walls[i, 0] == 0:
                edge_points.append((0, i))
            if walls[i, w - 1] == 0:
                edge_points.append((w - 1, i))

        if len(edge_points) < 4:
            logger.warning(f"泛洪法：边缘可用种子点过少({len(edge_points)})，尝试更小阈值")
            # 回退：使用更宽松的阈值
            _, th2 = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
            walls2 = cv2.dilate(th2, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
            edge_points2 = []
            for i in range(w):
                if walls2[0, i] == 0:
                    edge_points2.append((i, 0))
                if walls2[h - 1, i] == 0:
                    edge_points2.append((i, h - 1))
            for i in range(h):
                if walls2[i, 0] == 0:
                    edge_points2.append((0, i))
                if walls2[i, w - 1] == 0:
                    edge_points2.append((w - 1, i))
            if len(edge_points2) >= 4:
                walls = walls2
                edge_points = edge_points2
                logger.info(f"泛洪法：使用宽松阈值，边缘种子点={len(edge_points)}")

        if len(edge_points) < 4:
            logger.warning("泛洪法：无可用边缘种子点，放弃")
            return None

        # 3. 从可用种子点泛洪"外部"区域
        outside = np.zeros((h, w), dtype=np.uint8)
        ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

        # 均匀采样最多16个种子点
        step = max(1, len(edge_points) // 16)
        seeds_to_use = edge_points[::step][:16]

        filled_any = False
        for sx, sy in seeds_to_use:
            if walls[sy, sx] == 0:
                cv2.floodFill(outside, ff_mask, (sx, sy),
                              255, loDiff=1, upDiff=1,
                              flags=8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
                filled_any = True

        # 4. 如果泛洪失败，尝试从图像中心附近找一个"外部"点
        if not filled_any:
            logger.warning("泛洪法：边缘泛洪失败，尝试中心区域查找")
            # 尝试找一个确定在外部的点（通常在四角附近）
            for dx, dy in [(5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)]:
                if walls[dy, dx] == 0:
                    cv2.floodFill(outside, ff_mask, (dx, dy),
                                  255, loDiff=1, upDiff=1,
                                  flags=8 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE)
                    filled_any = True
                    break

        if not filled_any:
            logger.warning("泛洪法：所有种子点都失败，返回None")
            return None

        # 5. 内部 = 不是外部
        inside = np.ones((h, w), dtype=np.uint8) * 255
        inside[outside == 255] = 0

        # 6. 闭运算填充小空洞，开运算去除孤岛
        morph_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, morph_k, iterations=2)
        inside = cv2.morphologyEx(inside, cv2.MORPH_OPEN, morph_k, iterations=1)

        # 7. 验证：内部面积应在合理范围
        area = int(np.sum(inside > 128))
        total = w * h
        area_ratio = area / total

        # 放宽面积判断：
        # - 下限2%（避免捕获噪点）
        # - 上限98%（允许大的异形内部区域，但排除几乎全白的情况）
        if area < total * 0.02 or area > total * 0.98:
            logger.warning(f"泛洪法蒙版面积异常 area={area}({area_ratio*100:.2f}%), 放弃")
            return None

        ys, xs = np.where(inside > 128)
        if len(xs) == 0:
            return None

        x, y = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - x + 1), int(ys.max() - y + 1)

        mask_img = Image.fromarray(inside, mode="L")
        logger.info(f"泛洪法成功: area={area}({area_ratio*100:.1f}%), rect=({x},{y},{bw},{bh})")
        return ContourLevel(
            level=0,
            mask=mask_img,
            bounding_rect=(x, y, bw, bh),
            area=area,
            hierarchy=0,
        )

    def _extract_line_nested_contours(self, base_img: Image.Image) -> List[ContourLevel]:
        """通过线条膨胀检测嵌套轮廓（处理CAD线稿）

        思路：将CAD线稿逐步膨胀，每次膨胀后提取新的轮廓层。
        这样即使线稿是开口的或只有1像素宽，也能通过膨胀合并成闭合区域。

        优化v4：使用230阈值，减小初始膨胀步长。
        """
        import cv2

        arr = np.array(base_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 使用230阈值捕获更多细线条
        _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

        # 反相：线稿变白
        inv = cv2.bitwise_not(binary)

        total_area = base_img.width * base_img.height
        found_levels: List[ContourLevel] = []

        # 逐步膨胀：减小初始步长，更精细地检测嵌套结构
        max_iterations = 12
        for step in range(max_iterations):
            # 从3x3开始，步长2px增加
            kernel_size = 3 + step * 2
            if kernel_size > 25:
                break

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            dilated = cv2.dilate(inv, kernel, iterations=1)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < total_area * 0.02:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # 检查这个轮廓是否已被发现（位置相似）
                is_duplicate = False
                for existing in found_levels:
                    ex, ey, ew, eh = existing.bounding_rect
                    overlap = min(x + w, ex + ew) - max(x, ex)
                    overlap_ratio = overlap / min(w, ew) if min(w, ew) > 0 else 0
                    if overlap_ratio > 0.95 and abs(w - ew) / max(w, ew) < 0.1:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    mask_arr = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                    cv2.fillPoly(mask_arr, [contour], 255)
                    found_levels.append(ContourLevel(
                        level=len(found_levels),
                        mask=Image.fromarray(mask_arr, mode="L"),
                        bounding_rect=(x, y, w, h),
                        area=area,
                        hierarchy=step,
                    ))

        # 按面积排序
        found_levels.sort(key=lambda c: c.area, reverse=True)
        for i, level in enumerate(found_levels):
            level.level = i

        if found_levels:
            logger.info(f"线条检测提取到 {len(found_levels)} 层嵌套轮廓")

        return found_levels

    def _generate_virtual_contours(self, base_img: Image.Image,
                                     base_contour: ContourLevel,
                                     num_levels: int) -> List[ContourLevel]:
        """从单层基础轮廓生成多层虚拟轮廓

        通过系统腐蚀生成一系列内嵌轮廓层，
        每层都严格跟随外轮廓的形状。

        Args:
            base_contour: 基础轮廓（通常是最外层）
            num_levels: 生成的层数
        """
        import numpy as np

        levels = [base_contour]
        original_mask = base_contour.mask
        original_area = base_contour.area
        target_min_area = original_area * 0.05

        max_levels = min(num_levels, self.max_levels)
        min_dim = min(base_img.width, base_img.height)
        erosion_step = max(3, int(min_dim * 0.015))

        for i in range(1, max_levels):
            erosion_total = erosion_step * i

            # 每次从原始蒙版腐蚀（避免累积误差）
            eroded_mask = self.erode_mask(original_mask, erosion_total)
            eroded_arr = np.array(eroded_mask)
            area = int(np.sum(eroded_arr > 128))

            if area < target_min_area:
                break

            ys, xs = np.where(eroded_arr > 128)
            if len(xs) == 0:
                break

            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

            levels.append(ContourLevel(
                level=len(levels),
                mask=eroded_mask,
                bounding_rect=(x, y, w, h),
                area=area,
                hierarchy=i,
            ))

        logger.info(f"虚拟生成 {len(levels)} 层轮廓 (腐蚀步长={erosion_step}px)")
        return levels

    def _extract_single_contour(self, base_img: Image.Image) -> List[ContourLevel]:
        """回退方案：只提取单层外轮廓"""
        import numpy as np

        mask = self._create_single_mask(base_img)
        mask_arr = np.array(mask)
        ys, xs = np.where(mask_arr > 0)
        if len(xs) == 0:
            x, y, w, h = 0, 0, base_img.width, base_img.height
            area = base_img.width * base_img.height
        else:
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
            area = int(np.sum(mask_arr > 0))

        return [ContourLevel(
            level=0,
            mask=mask,
            bounding_rect=(x, y, w, h),
            area=area,
            hierarchy=0,
        )]

    def _create_single_mask(self, base_img: Image.Image) -> Image.Image:
        """创建单层外轮廓蒙版（优化v4：减小形态学参数）"""
        try:
            import cv2
            import numpy as np

            arr = np.array(base_img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            # 使用230阈值捕获更多细线条
            _, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

            # 3x3小核闭运算1次
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
            inv = cv2.bitwise_not(closed)

            # 3x3核膨胀1次，不做腐蚀
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            inv = cv2.dilate(inv, kernel_dilate, iterations=1)

            contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                mask_arr = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                cv2.fillPoly(mask_arr, [largest], 255)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask_arr = cv2.morphologyEx(mask_arr, cv2.MORPH_OPEN, kernel_open, iterations=1)
                return Image.fromarray(mask_arr, mode="L")
        except ImportError:
            pass

        return Image.new("L", base_img.size, 255)

    def erode_mask(self, mask: Image.Image, erosion_px: int) -> Image.Image:
        """腐蚀蒙版"""
        if erosion_px <= 0:
            return mask.copy()

        try:
            import numpy as np
            from scipy import ndimage

            mask_arr = np.array(mask)
            binary = mask_arr > 128
            eroded_binary = ndimage.binary_erosion(binary, iterations=erosion_px)
            eroded = (eroded_binary.astype(np.uint8)) * 255
            return Image.fromarray(eroded, mode="L")
        except ImportError:
            try:
                import cv2
                import numpy as np

                mask_arr = np.array(mask)
                # Add 1px black border to ensure erosion works even on full-white masks
                bordered = np.pad(mask_arr, 1, mode="constant", constant_values=0)

                # Use iterative erosion for large values to avoid huge kernels
                if erosion_px > 20:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    # Erode in chunks of 20 to avoid excessive iterations
                    remaining = erosion_px
                    result = bordered
                    while remaining > 0:
                        chunk = min(remaining, 20)
                        result = cv2.erode(result, kernel, iterations=chunk)
                        remaining -= chunk
                    eroded = result[1:-1, 1:-1]
                else:
                    kernel_size = max(1, erosion_px * 2 + 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    eroded = cv2.erode(bordered, kernel, iterations=1)
                    eroded = eroded[1:-1, 1:-1]

                return Image.fromarray(eroded, mode="L")
            except ImportError:
                return self._pil_erode(mask, erosion_px)

    def dilate_mask(self, mask: Image.Image, dilation_px: int) -> Image.Image:
        """膨胀蒙版"""
        if dilation_px <= 0:
            return mask.copy()

        try:
            import numpy as np
            from scipy import ndimage

            mask_arr = np.array(mask)
            binary = mask_arr > 128
            dilated_binary = ndimage.binary_dilation(binary, iterations=dilation_px)
            dilated = (dilated_binary.astype(np.uint8)) * 255
            return Image.fromarray(dilated, mode="L")
        except ImportError:
            try:
                import cv2
                import numpy as np

                mask_arr = np.array(mask)
                kernel_size = max(1, dilation_px * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                dilated = cv2.dilate(mask_arr, kernel, iterations=1)
                return Image.fromarray(dilated, mode="L")
            except ImportError:
                return self._pil_dilate(mask, dilation_px)

    def _pil_erode(self, mask: Image.Image, erosion_px: int) -> Image.Image:
        """PIL回退腐蚀"""
        from PIL import ImageFilter
        img = mask.copy()
        for _ in range(erosion_px):
            img = img.filter(ImageFilter.MinFilter(3))
        return img

    def _pil_dilate(self, mask: Image.Image, dilation_px: int) -> Image.Image:
        """PIL回退膨胀"""
        from PIL import ImageFilter
        img = mask.copy()
        for _ in range(dilation_px):
            img = img.filter(ImageFilter.MaxFilter(3))
        return img

    def create_border_ring_mask(self, outer_mask: Image.Image,
                                 inner_mask: Image.Image) -> Image.Image:
        """创建环形边框蒙版（两个蒙版的差集）

        用于生成"边框带"区域：外层蒙版 - 内层蒙版 = 边框环
        """
        import numpy as np

        outer_arr = np.array(outer_mask)
        inner_arr = np.array(inner_mask)

        ring_arr = np.where(
            (outer_arr > 128) & (inner_arr <= 128),
            255, 0
        ).astype(np.uint8)

        return Image.fromarray(ring_arr, mode="L")

    def compute_erosion_for_layer(self, outer_rect: Tuple[int, int, int, int],
                                   layer_rect: Tuple[int, int, int, int]) -> int:
        """计算PSD某层相对于外轮廓的腐蚀距离

        Args:
            outer_rect: 外轮廓边界 (x, y, w, h)
            layer_rect: PSD图层边界 (x, y, w, h)
        Returns:
            腐蚀像素数（取四边最小值）
        """
        ox, oy, ow, oh = outer_rect
        lx, ly, lw, lh = layer_rect

        # 各方向的距离（PSD层相对于外轮廓的内缩量）
        left_dist = lx - ox
        top_dist = ly - oy
        right_dist = (ox + ow) - (lx + lw)
        bottom_dist = (oy + oh) - (ly + lh)

        # 取最小值作为腐蚀量（保证不会超出边界）
        distances = [d for d in [left_dist, top_dist, right_dist, bottom_dist] if d > 0]
        if not distances:
            return 0

        # 取中位数避免极端值
        sorted_dists = sorted(distances)
        return int(sorted_dists[len(sorted_dists) // 2])
