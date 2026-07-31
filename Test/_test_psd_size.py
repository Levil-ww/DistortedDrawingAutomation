import sys, logging
sys.path.insert(0, '..')
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

from pathlib import Path
from PIL import Image
import numpy as np
from engines.pillow_engine import PillowEngine
from services.design_service import DesignService
from models import ProcessConfig

engine = PillowEngine()
service = DesignService(engine=engine)

eps_path = r'C:\Users\Administrator\Desktop\测试\260721-定制双面革蔓生花;80x150cm剪裁有图-1张.eps'
psd_path = r'C:\Users\Administrator\Desktop\测试\蔓生花80-140.psd'

print('=== Test: PSD Size Priority ===')
cfg = ProcessConfig(
    eps_file=eps_path, psd_file=psd_path,
    canvas_width_cm=80.0, canvas_height_cm=150.0,
    dpi=72, smart_align=True, auto_scale=True,
)

# 验证画布尺寸
w, h = service._resolve_canvas_size(Path(eps_path), cfg)
print(f'Canvas size: {w:.1f} x {h:.1f} cm')

# 生成预览
print('\n=== Generating Preview ===')
preview = service.generate_preview(cfg, max_width=600)
print(f'Preview: {preview.width}x{preview.height}')

# 检查留白
arr = np.array(preview.convert('RGB'))
total = arr.shape[0] * arr.shape[1]
non_white = np.sum(np.any(arr != [255,255,255], axis=2))
print(f'Non-white pixels: {non_white}/{total} ({non_white/total*100:.1f}%)')

# 保存预览
import tempfile, os
preview_path = os.path.join(tempfile.gettempdir(), 'preview_psd_size.jpg')
preview.save(preview_path, quality=95)
print(f'Preview saved: {preview_path}')

print('\n=== Done ===')