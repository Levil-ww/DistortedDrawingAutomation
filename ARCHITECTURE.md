# 设计自动化工具 v2.0 - 架构说明

## 1. 架构概览

本项目采用**分层架构 + 依赖注入**的设计模式，通过严格的依赖方向控制，彻底消除循环依赖，确保代码的可维护性和可扩展性。

```
┌─────────────────────────────────────────────────────────────┐
│  入口层 (Entry Points)                                        │
│  ├── main.py              # CLI 入口                          │
│  └── gui/app.py           # GUI 入口                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  依赖注入层 (DI Container)                                    │
│  └── di_container.py      # 集中管理所有组件创建和生命周期       │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   GUI 层      │   │  服务层       │   │ 配置管理       │
│  (用户界面)    │   │  (业务编排)    │   │              │
└───────────────┘   └───────────────┘   └───────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   引擎层      │   │  处理器层      │   │   模型层      │
│ (ImageEngine) │   │  (算法实现)    │   │  (数据结构)    │
│               │   │               │   │               │
│ • Photoshop   │   │ • SmartAligner│   │ • ProcessConfig│
│ • Pillow      │   │ • ColorAdjuster│  │ • LayerInfo   │
│               │   │ • Compositor  │   │ • EngineCaps  │
└───────────────┘   └───────────────┘   └───────────────┘
```

## 2. 核心设计原则

### 2.1 依赖方向规则

```
上层 ──────────────────────────────> 下层

  GUI → DI Container → Services → Engines
                          ↓
                    Processors → Models
```

- **上层可以导入下层**
- **下层严禁导入上层**
- **Models 是最底层，不依赖任何内部模块**

### 2.2 各层职责

| 层级 | 职责 | 禁止的行为 |
|------|------|-----------|
| **models** | 定义数据结构和常量 | 不得有任何业务逻辑 |
| **processors** | 纯算法实现（图像处理） | 不得依赖 engines/services/gui |
| **engines** | 图像引擎接口和实现 | 不得依赖 processors/services/gui |
| **services** | 业务流程编排 | 不得导入 gui，只通过 DI 获取依赖 |
| **gui** | 用户界面和交互 | 只通过 DI Container 获取服务 |
| **di_container** | 依赖管理和生命周期 | 是唯一切合点，其他层不互相导入 |

## 3. 重构前的架构问题

### 3.1 循环依赖风险

```python
# 问题：engines/photoshop_engine.py
from processors.color_adjuster import ColorAdjuster  # 跨层导入！

# 问题：engines/base.py
from processors import LayerInfo  # LayerInfo 应该在 models
```

### 3.2 重复定义

- `LayerInfo` 在 `engines/base.py` 和 `models.py` 重复定义
- `EngineCapabilities` 在两个位置定义

### 3.3 GUI 直接依赖服务层

```python
# 问题：GUI 直接创建引擎和服务
core_engine = DesignAutoCore()  # GUI 依赖了 core_engine
```

## 4. 重构方案

### 4.1 所有共享模型移至 models.py

```python
# models.py - 现在包含所有数据结构
@dataclass
class EngineCapabilities: ...

@dataclass  
class LayerInfo: ...

@dataclass
class ProcessConfig: ...
```

### 4.2 引擎层清理

移除了引擎层对 processors 的所有导入：

```python
# engines/base.py - 清理后
from models import EngineCapabilities, LayerInfo  # 只依赖 models

# engines/photoshop_engine.py - 清理后
# 删除了：from processors.color_adjuster import ColorAdjuster
# 引擎不再处理色彩调整，由上层服务层统一调用
```

### 4.3 引入 DI Container

```python
# di_container.py - 新增
class DIContainer:
    def register_engine(self, preferred: str) -> ImageEngine: ...
    def create_service(self, ...) -> DesignService: ...
    def load_config(self, ...) -> ProcessConfig: ...
```

### 4.4 GUI 只依赖 DI Container

```python
# gui/app.py - 重构后
from models import ProcessConfig  # 只导入模型
from di_container import DIContainer  # 只导入 DI 容器

# 不再直接导入：
# from core_engine import DesignAutoCore  # 删除
# from engines import create_engine  # 删除
# from services.design_service import DesignService  # 删除
```

## 5. 扩展性设计

### 5.1 添加新引擎

只需两步：

1. 继承 `ImageEngine` 实现新引擎
2. 在 `engines/__init__.py` 的 `create_engine()` 中添加分支

```python
# engines/gimp_engine.py
from .base import ImageEngine

class GimpEngine(ImageEngine):
    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(name="GIMP", ...)
    
    def open_eps(self, path: Path, dpi: int = 300) -> Image.Image: ...
    # ... 实现其他方法
```

### 5.2 添加新处理器

```python
# processors/blur_processor.py
from PIL import Image, ImageFilter

class BlurProcessor:
    def apply(self, image: Image.Image, radius: float) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius))
```

然后在 `services/design_service.py` 中注入使用：

```python
class DesignService:
    def __init__(self, ..., blur_processor: Optional[BlurProcessor] = None):
        self.blur_processor = blur_processor or BlurProcessor()
```

### 5.3 添加新界面

可以创建 `web/app.py` 或 `cli/app.py`，都只依赖 `DIContainer`：

```python
# web/app.py (Flask示例)
from flask import Flask
from di_container import DIContainer

app = Flask(__name__)
container = DIContainer()

@app.route('/process', methods=['POST'])
def process():
    config = ProcessConfig.from_dict(request.json)
    with DIContainer() as container:
        service = container.create_service()
        result = service.process(config)
    return jsonify({"output": str(result)})
```

## 6. 依赖图

### 6.1 重构后的干净依赖

```
design_automation_v2/
├── __init__.py          # 无依赖
├── models.py            # 仅依赖 PIL
├── config_manager.py    # 依赖 models
├── di_container.py      # 依赖 models, engines, processors, services, config_manager
├── main.py              # 依赖 di_container
│
├── engines/
│   ├── __init__.py      # 依赖 models, .base, .photoshop_engine, .pillow_engine
│   ├── base.py          # 依赖 models
│   ├── photoshop_engine.py  # 依赖 models, .base
│   └── pillow_engine.py     # 依赖 models, .base
│
├── processors/
│   ├── __init__.py      # 依赖 models, .aligner, .color_adjuster, .compositor
│   ├── aligner.py       # 依赖 models, PIL, numpy
│   ├── color_adjuster.py    # 依赖 models, PIL
│   └── compositor.py    # 依赖 models, PIL
│
├── services/
│   └── design_service.py    # 依赖 models, engines.base, processors
│
└── gui/
    └── app.py           # 依赖 models, di_container
```

### 6.2 无循环依赖验证

```bash
# 验证方法：从最底层开始逐层导入
python -c "from models import *; print('OK')"
python -c "from processors import *; print('OK')"  
python -c "from engines import *; print('OK')"
python -c "from services.design_service import DesignService; print('OK')"
python -c "from di_container import DIContainer; print('OK')"
python -c "from gui.app import DesignAutoGUI; print('OK')"
```

## 7. 测试建议

### 7.1 单元测试结构

```
tests/
├── test_models.py           # 测试数据模型
├── test_processors/         # 测试算法
│   ├── test_aligner.py
│   ├── test_color_adjuster.py
│   └── test_compositor.py
├── test_engines/            # 测试引擎（使用 mock）
│   ├── test_photoshop_engine.py
│   └── test_pillow_engine.py
├── test_services/           # 测试业务逻辑（使用 mock 引擎）
│   └── test_design_service.py
└── test_integration.py      # 集成测试
```

### 7.2 Mock 示例

```python
# 测试 service 时 mock 引擎
import unittest
from unittest.mock import Mock, MagicMock
from services.design_service import DesignService

class TestDesignService(unittest.TestCase):
    def setUp(self):
        self.mock_engine = Mock()
        self.mock_engine.open_eps.return_value = Mock(width=100, height=100)
        self.mock_engine.load_psd_layers.return_value = [
            LayerInfo("test", Mock(mode="RGBA", width=50, height=50))
        ]
        self.service = DesignService(engine=self.mock_engine)
    
    def test_process(self):
        config = ProcessConfig(eps_file="test.eps", psd_file="test.psd")
        self.service.process(config)
        self.mock_engine.open_eps.assert_called_once()
```

## 8. 最佳实践

### 8.1 添加新功能时的检查清单

- [ ] 新数据结构 → 添加到 `models.py`
- [ ] 新图像算法 → 添加到 `processors/`，只依赖 `models`
- [ ] 新文件格式支持 → 添加到 `engines/`，只依赖 `models`
- [ ] 新业务逻辑 → 添加到 `services/`，通过构造函数注入依赖
- [ ] 新 UI 界面 → 只导入 `models` 和 `di_container`
- [ ] 更新 DI Container → 在 `di_container.py` 中注册新组件

### 8.2 禁止事项

❌ 不要在 engines 中导入 processors  
❌ 不要在 processors 中导入 engines  
❌ 不要在 services 中导入 gui  
❌ 不要在 models 中添加业务逻辑  
❌ 不要跨层直接创建实例（使用 DI Container）

### 8.3 推荐做法

✅ 各层只通过构造函数接收依赖  
✅ 使用 `typing.Optional` 标记可选依赖  
✅ 使用 DI Container 管理单例生命周期  
✅ 使用上下文管理器确保资源清理

## 9. 总结

通过本次重构：

1. **消除循环依赖**：所有依赖方向从上层指向下层，无反向依赖
2. **单一职责**：每层只负责自己的事情
3. **易于测试**：每层可以独立测试，易于 mock
4. **易于扩展**：添加新引擎/处理器/界面都只需修改局部
5. **代码清晰**：依赖关系明确，通过 DI Container 一目了然
