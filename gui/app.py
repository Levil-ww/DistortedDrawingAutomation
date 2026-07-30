#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI 界面层 - 仅负责用户交互和状态显示

所有业务逻辑通过 DIContainer 获取 DesignService 处理，
避免直接依赖 engines/services 的具体实现。

设计：GUI 只依赖 di_container 和 models，不直接依赖 engines/services
"""

import os
import sys
import threading
from pathlib import Path
from tkinter import (
    Tk, Frame, Label, Button, Entry, Scale, Checkbutton,
    filedialog, messagebox, StringVar, IntVar, DoubleVar, BooleanVar,
    HORIZONTAL, Text, Scrollbar
)
from PIL import Image, ImageTk

# 将项目根目录加入路径，支持直接运行
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import ProcessConfig
from di_container import DIContainer


class DesignAutoGUI:
    """设计自动化图形界面"""

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("异形设计自动化工具 v2.0")
        self.root.geometry("920x800")
        self.root.configure(bg="#f5f5f5")

        self.work_dir = Path(__file__).parent.parent
        self._container: DIContainer = DIContainer()

        self._build_ui()
        self._load_defaults()

    # ---------- UI 构建 ----------

    def _build_ui(self):
        main = Frame(self.root, bg="#f5f5f5", padx=15, pady=10)
        main.pack(fill="both", expand=True)

        Label(main, text="异形设计自动化工具", font=("Microsoft YaHei", 18, "bold"),
              bg="#f5f5f5", fg="#2c3e50").pack(pady=(0, 5))

        self.engine_var = StringVar(value="引擎: 检测中...")
        Label(main, textvariable=self.engine_var, font=("Microsoft YaHei", 10),
              bg="#f5f5f5", fg="#27ae60").pack(pady=(0, 10))

        content = Frame(main, bg="#f5f5f5")
        content.pack(fill="both", expand=True)

        left = Frame(content, bg="#f5f5f5", width=500)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        self._build_files(left)
        self._build_align(left)
        self._build_color(left)
        self._build_output(left)
        self._build_buttons(left)

        right = Frame(content, bg="#ffffff", bd=1, relief="solid")
        right.pack(side="right", fill="both", expand=True)
        self._build_preview(right)
        self._build_log(right)

    def _build_files(self, parent):
        sec = Frame(parent, bg="#f5f5f5", pady=8)
        sec.pack(fill="x")
        Label(sec, text="文件选择", font=("Microsoft YaHei", 12, "bold"),
              bg="#f5f5f5", fg="#2c3e50").pack(anchor="w")

        for label, var_name, cmd in [
            ("EPS模板:", "eps_var", self._browse_eps),
            ("PSD素材:", "psd_var", self._browse_psd),
            ("输出路径:", "out_var", self._browse_out),
        ]:
            row = Frame(sec, bg="#f5f5f5")
            row.pack(fill="x", pady=3)
            Label(row, text=label, width=10, bg="#f5f5f5", anchor="e").pack(side="left")
            var = StringVar()
            setattr(self, var_name, var)
            Entry(row, textvariable=var, width=38).pack(side="left", padx=5)
            Button(row, text="浏览...", command=cmd, width=8).pack(side="left")

        self.out_var.set("output_v2.jpg")

    def _build_align(self, parent):
        sec = Frame(parent, bg="#f5f5f5", pady=8)
        sec.pack(fill="x")
        Label(sec, text="智能对齐", font=("Microsoft YaHei", 12, "bold"),
              bg="#f5f5f5", fg="#2c3e50").pack(anchor="w")

        self.smart_align_var = BooleanVar(value=True)
        Checkbutton(sec, text="启用智能对齐", variable=self.smart_align_var,
                    bg="#f5f5f5", font=("Microsoft YaHei", 10)).pack(anchor="w")

        self.auto_scale_var = BooleanVar(value=True)
        Checkbutton(sec, text="自动缩放", variable=self.auto_scale_var,
                    bg="#f5f5f5", font=("Microsoft YaHei", 10)).pack(anchor="w")

        row = Frame(sec, bg="#f5f5f5")
        row.pack(fill="x", pady=2)
        Label(row, text="边距(%):", width=10, bg="#f5f5f5", anchor="e").pack(side="left")
        self.margin_var = DoubleVar(value=2.0)
        Scale(row, from_=0, to=10, resolution=0.5, orient=HORIZONTAL,
              variable=self.margin_var, length=260, bg="#f5f5f5",
              highlightthickness=0).pack(side="left", padx=5)

        Label(sec, text="手动参数", font=("Microsoft YaHei", 10, "bold"),
              bg="#f5f5f5", fg="#7f8c8d").pack(anchor="w", pady=(8, 0))

        for label, var_name, min_v, max_v, res in [
            ("缩放:", "scale_var", 0.1, 3.0, 0.05),
            ("X偏移:", "ox_var", -500, 500, 10),
            ("Y偏移:", "oy_var", -500, 500, 10),
        ]:
            row = Frame(sec, bg="#f5f5f5")
            row.pack(fill="x", pady=1)
            Label(row, text=label, width=10, bg="#f5f5f5", anchor="e").pack(side="left")
            var = DoubleVar(value=1.0) if "scale" in var_name else IntVar(value=0)
            setattr(self, var_name, var)
            Scale(row, from_=min_v, to=max_v, resolution=res, orient=HORIZONTAL,
                  variable=var, length=260, bg="#f5f5f5",
                  highlightthickness=0).pack(side="left", padx=5)

    def _build_color(self, parent):
        sec = Frame(parent, bg="#f5f5f5", pady=8)
        sec.pack(fill="x")
        Label(sec, text="色彩调整", font=("Microsoft YaHei", 12, "bold"),
              bg="#f5f5f5", fg="#2c3e50").pack(anchor="w")

        for label, var_name, min_v, max_v in [
            ("亮度:", "bright_var", -100, 100),
            ("对比度:", "contrast_var", -100, 100),
            ("饱和度:", "sat_var", -100, 100),
            ("色相:", "hue_var", -180, 180),
            ("色温:", "warmth_var", -100, 100),
        ]:
            row = Frame(sec, bg="#f5f5f5")
            row.pack(fill="x", pady=1)
            Label(row, text=label, width=10, bg="#f5f5f5", anchor="e").pack(side="left")
            var = IntVar(value=0)
            setattr(self, var_name, var)
            Scale(row, from_=min_v, to=max_v, resolution=1, orient=HORIZONTAL,
                  variable=var, length=260, bg="#f5f5f5",
                  highlightthickness=0).pack(side="left", padx=5)
            Label(row, textvariable=var, bg="#f5f5f5", width=5).pack(side="left")

    def _build_output(self, parent):
        sec = Frame(parent, bg="#f5f5f5", pady=8)
        sec.pack(fill="x")
        Label(sec, text="输出设置", font=("Microsoft YaHei", 12, "bold"),
              bg="#f5f5f5", fg="#2c3e50").pack(anchor="w")

        row = Frame(sec, bg="#f5f5f5")
        row.pack(fill="x", pady=2)
        Label(row, text="JPG质量:", width=10, bg="#f5f5f5", anchor="e").pack(side="left")
        self.quality_var = IntVar(value=95)
        Scale(row, from_=50, to=100, resolution=1, orient=HORIZONTAL,
              variable=self.quality_var, length=260, bg="#f5f5f5",
              highlightthickness=0).pack(side="left", padx=5)
        Label(row, textvariable=self.quality_var, bg="#f5f5f5", width=5).pack(side="left")

    def _build_buttons(self, parent):
        sec = Frame(parent, bg="#f5f5f5", pady=12)
        sec.pack(fill="x")

        bf = Frame(sec, bg="#f5f5f5")
        bf.pack()

        Button(bf, text="生成预览", command=self._on_preview,
               font=("Microsoft YaHei", 11), width=12, bg="#3498db", fg="white").pack(side="left", padx=6)
        Button(bf, text="开始处理", command=self._on_process,
               font=("Microsoft YaHei", 11, "bold"), width=12, bg="#27ae60", fg="white").pack(side="left", padx=6)
        Button(bf, text="重置参数", command=self._on_reset,
               font=("Microsoft YaHei", 11), width=12, bg="#95a5a6", fg="white").pack(side="left", padx=6)

        self.status_var = StringVar(value="就绪")
        Label(sec, textvariable=self.status_var, font=("Microsoft YaHei", 10),
              bg="#f5f5f5", fg="#e74c3c").pack(pady=(10, 0))

    def _build_preview(self, parent):
        hdr = Frame(parent, bg="#ecf0f1", padx=10, pady=5)
        hdr.pack(fill="x")
        Label(hdr, text="效果预览", font=("Microsoft YaHei", 12, "bold"),
              bg="#ecf0f1", fg="#2c3e50").pack(side="left")

        self.preview_label = Label(parent, text="暂无预览\n点击「生成预览」",
                                   bg="#ffffff", fg="#bdc3c7", font=("Microsoft YaHei", 12))
        self.preview_label.pack(expand=True, padx=10, pady=10)

    def _build_log(self, parent):
        hdr = Frame(parent, bg="#ecf0f1", padx=10, pady=5)
        hdr.pack(fill="x")
        Label(hdr, text="处理日志", font=("Microsoft YaHei", 11, "bold"),
              bg="#ecf0f1", fg="#2c3e50").pack(side="left")

        lf = Frame(parent, bg="#ffffff")
        lf.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        sb = Scrollbar(lf)
        sb.pack(side="right", fill="y")
        self.log_text = Text(lf, height=8, wrap="word", yscrollcommand=sb.set,
                             font=("Consolas", 9), bg="#fafafa", fg="#2c3e50", relief="flat", padx=8, pady=5)
        self.log_text.pack(fill="both", expand=True)
        sb.config(command=self.log_text.yview)

    # ---------- 事件处理 ----------

    def _browse_eps(self):
        p = filedialog.askopenfilename(title="选择EPS模板", filetypes=[("EPS", "*.eps"), ("所有文件", "*.*")])
        if p:
            self.eps_var.set(p)

    def _browse_psd(self):
        p = filedialog.askopenfilename(title="选择PSD素材", filetypes=[("PSD", "*.psd"), ("所有文件", "*.*")])
        if p:
            self.psd_var.set(p)

    def _browse_out(self):
        p = filedialog.asksaveasfilename(title="输出路径", defaultextension=".jpg",
                                         filetypes=[("JPEG", "*.jpg"), ("所有文件", "*.*")])
        if p:
            self.out_var.set(p)

    def _load_defaults(self):
        eps = list(self.work_dir.glob("*.eps"))
        psd = list(self.work_dir.glob("*.psd"))
        if eps:
            self.eps_var.set(str(eps[0]))
        if psd:
            self.psd_var.set(str(psd[0]))

    def _collect_config(self) -> ProcessConfig:
        return ProcessConfig(
            eps_file=self.eps_var.get(),
            psd_file=self.psd_var.get(),
            output_file=self.out_var.get(),
            smart_align=self.smart_align_var.get(),
            auto_scale=self.auto_scale_var.get(),
            margin_percent=self.margin_var.get(),
            pattern_scale=self.scale_var.get(),
            pattern_offset_x=int(self.ox_var.get()),
            pattern_offset_y=int(self.oy_var.get()),
            brightness=self.bright_var.get(),
            contrast=self.contrast_var.get(),
            saturation=self.sat_var.get(),
            hue_shift=self.hue_var.get(),
            warmth=self.warmth_var.get(),
            jpg_quality=self.quality_var.get(),
        )

    def _log(self, msg: str):
        self.root.after(0, lambda m=msg: self._do_log(m))

    def _do_log(self, msg: str):
        self.log_text.insert("end", msg + "\n")
        self.log_text.see("end")
        self.log_text.update_idletasks()

    def _on_preview(self):
        self.status_var.set("生成预览中...")
        self._log("开始生成预览...")

        def task():
            try:
                cfg = self._collect_config()

                if not cfg.eps_file or not Path(cfg.eps_file).exists():
                    raise RuntimeError(f"EPS文件不存在: {cfg.eps_file}")
                if not cfg.psd_file or not Path(cfg.psd_file).exists():
                    raise RuntimeError(f"PSD文件不存在: {cfg.psd_file}")

                engine = self._container.register_engine("auto")
                self.engine_var.set(f"引擎: {engine.capabilities.name}")

                service = self._container.create_service()
                img = service.generate_preview(cfg, max_width=400)
                self._log(f"预览图生成: {img.width}x{img.height}, 模式={img.mode}")

                if img.width < 10 or img.height < 10:
                    self._log(f"警告: 预览图尺寸异常({img.width}x{img.height})，可能是EPS无法栅格化")
                    self._log("建议安装Ghostscript以获得正确的EPS渲染效果")

                if img.width <= 0 or img.height <= 0:
                    img = Image.new("RGB", (400, 300), "white")

                max_w, max_h = 380, 300
                ratio = min(max_w / img.width, max_h / img.height)
                ratio = max(ratio, 0.01)
                img = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS)

                self.root.after(0, lambda: self._show_image(img))
                self.root.after(0, lambda: self.status_var.set("预览就绪"))
                self.root.after(0, lambda: self._log(f"预览生成完成 ({img.width}x{img.height})"))
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                self.root.after(0, lambda: self._log(f"预览失败: {e}"))
                self.root.after(0, lambda: self._log(err))
                self.root.after(0, lambda: self.status_var.set("预览失败"))
            finally:
                self._container.cleanup()

        threading.Thread(target=task, daemon=True).start()

    def _show_image(self, img: Image.Image):
        self.preview_photo = ImageTk.PhotoImage(img)
        self.preview_label.config(
            image=self.preview_photo,
            text="",
            bg="#ffffff",
            compound="center"
        )
        self.preview_label.update_idletasks()

    def _on_process(self):
        if not self.eps_var.get() or not self.psd_var.get():
            messagebox.showwarning("提示", "请先选择EPS模板和PSD素材")
            return

        self.status_var.set("处理中...")
        self._log("=" * 40)
        self._log("开始处理")

        def progress(msg: str):
            self.root.after(0, lambda m=msg: self._log(m))

        def task():
            try:
                cfg = self._collect_config()

                # 使用 DI 容器创建引擎和服务
                engine = self._container.register_engine("auto")
                self.engine_var.set(f"引擎: {engine.capabilities.name}")

                service = self._container.create_service()
                out = service.process(cfg, progress_callback=progress)
                self.root.after(0, lambda: messagebox.showinfo("完成", f"已保存:\n{out}"))
                self.status_var.set("处理完成")
            except Exception as e:
                self._log(f"错误: {e}")
                self.status_var.set("处理失败")
            finally:
                self._container.cleanup()

        threading.Thread(target=task, daemon=True).start()

    def _on_reset(self):
        self.smart_align_var.set(True)
        self.auto_scale_var.set(True)
        self.margin_var.set(2.0)
        self.scale_var.set(1.0)
        self.ox_var.set(0)
        self.oy_var.set(0)
        for v in ["bright_var", "contrast_var", "sat_var", "hue_var", "warmth_var"]:
            getattr(self, v).set(0)
        self.quality_var.set(95)
        self._log("参数已重置")


def main():
    root = Tk()
    app = DesignAutoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
