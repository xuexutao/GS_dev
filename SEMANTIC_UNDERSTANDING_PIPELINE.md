# 3DGS 语义理解与查询功能增强

## 概述

本文档记录了在 3D Gaussian Splatting (3DGS) 代码库中实现的语义理解与查询功能的完整修改。该功能扩展了原始的3DGS系统，使其能够：

1. **语义标签分配**：为SAM生成的2D掩码分配语义类别标签
2. **3D高斯语义存储**：在3D高斯点云中存储语义标签信息
3. **场景语义查询**：支持基于语义的3D场景查询和交互式探索

## 整体架构

```
原始3DGS流程：
多视图图像 → COLMAP重建 → 3D高斯训练 → 渲染

增强后的语义理解流程：
多视图图像 → SAM掩码生成 → CLIP语义标签分配 → 3D高斯训练（带语义标签） → 语义场景查询
```

## 详细代码修改说明

### 文件：`scene/gaussian_model.py`

#### 1. 新增语义标签属性（第28行附近）
```python
# 新增语义标签属性
self._semantic = torch.empty(0, dtype=torch.long)
```

#### 2. 修改 `capture()` 方法（第60行附近）
```python
def capture(self):
    return (
        self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        self.xyz_gradient_accum,
        self.denom,
        self.optimizer.state_dict(),
        self.spatial_lr_scale,
        self._semantic  # 新增：包含语义标签
    )
```

#### 3. 修改 `restore()` 方法（第77行附近）
```python
def restore(self, model_args, training_args):
    (self.active_sh_degree, 
     self._xyz, 
     self._features_dc, 
     self._features_rest,
     self._scaling, 
     self._rotation, 
     self._opacity,
     self.max_radii2D, 
     xyz_gradient_accum, 
     denom,
     opt_dict, 
     self.spatial_lr_scale,
     self._semantic) = model_args  # 新增：解包语义标签
    self.training_setup(training_args)
    self.xyz_gradient_accum = xyz_gradient_accum
    self.denom = denom
    self.optimizer.load_state_dict(opt_dict)
```

#### 4. 修改 `construct_list_of_attributes()` 方法（第99行附近）
```python
def construct_list_of_attributes(self):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
        l.append('f_dc_{}'.format(i))
    for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(self._scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(self._rotation.shape[1]):
        l.append('rot_{}'.format(i))
    l.append('semantic')  # 新增：语义标签字段
    return l
```

#### 5. 修改 `save_ply()` 方法（第114行附近）
```python
def save_ply(self, path):
    # ... 现有代码 ...
    
    # 语义标签处理：如果为空则填充-1
    if self._semantic.numel() == 0:
        semantic = np.full((xyz.shape[0], 1), -1, dtype=np.float32)
    else:
        semantic = self._semantic.detach().cpu().numpy().reshape(-1, 1).astype(np.float32)
    
    # 将语义标签添加到属性列表
    dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotations, semantic), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # ... 现有代码 ...
```

#### 6. 修改 `load_ply()` 方法（第152行附近）
```python
def load_ply(self, path):
    # ... 现有代码 ...
    
    # 读取语义标签，如果没有则初始化为-1
    if "semantic" in plydata.elements[0].properties:
        semantic = np.asarray(plydata.elements[0]["semantic"])[..., np.newaxis].astype(np.int64)
    else:
        semantic = np.full((xyz.shape[0], 1), -1, dtype=np.int64)
    
    self._semantic = torch.tensor(semantic, dtype=torch.long, device="cuda").squeeze()
    
    # ... 现有代码 ...
```

#### 7. 修改 `create_from_pcd()` 方法（第209行附近）
```python
def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
    # ... 现有代码 ...
    
    # 初始化语义标签为-1（表示未标记）
    self._semantic = torch.full((fused_point_cloud.shape[0],), -1, dtype=torch.long, device="cuda")
    
    # ... 现有代码 ...
```

#### 8. 修改 `densification_postfix()` 方法（第289行附近）
```python
def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_semantic=None):
    # ... 现有代码 ...
    
    # 处理语义标签：如果提供了新的语义标签，则连接；否则用-1填充
    if new_semantic is not None:
        self._semantic = torch.cat((self._semantic, new_semantic))
    else:
        # 用-1填充新点的语义标签
        self._semantic = torch.cat((self._semantic, torch.full((new_xyz.shape[0],), -1, dtype=torch.long, device="cuda")))
    
    # ... 现有代码 ...
```

#### 9. 修改 `densify_and_split()` 和 `densify_and_clone()` 方法
```python
# densify_and_split() 方法中调用 densification_postfix() 时传递语义标签
self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_semantic)

# densify_and_clone() 方法中调用 densification_postfix() 时传递语义标签
self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_semantic)
```

#### 10. 修改 `prune_points()` 方法（第354行附近）
```python
def prune_points(self, mask):
    valid_points_mask = ~mask
    
    # ... 修剪其他属性 ...
    
    # 修剪语义标签
    self._semantic = self._semantic[valid_points_mask]
    
    # ... 现有代码 ...
```

### 文件：`train.py`

#### 1. 新增命令行参数（第30行附近）
```python
parser.add_argument("--semantic_labels", type=str, default=None,
                    help="Path to labels.json file containing semantic label mapping")
```

#### 2. 修改 `training()` 函数签名（第170行附近）
```python
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, semantic_labels_path=None):
```

#### 3. 添加语义标签加载逻辑（第180行附近）
```python
# 加载语义标签文件（如果提供）
semantic_labels = None
if semantic_labels_path and os.path.exists(semantic_labels_path):
    print(f"Loading semantic labels from {semantic_labels_path}")
    import json
    with open(semantic_labels_path, 'r') as f:
        semantic_labels = json.load(f)
    print(f"Loaded {len(semantic_labels)} semantic labels")
```

### 文件：`scene/__init__.py`

#### 1. 新增 `assign_semantic_labels()` 方法（第120行附近）
```python
def assign_semantic_labels(self, semantic_labels: Dict, category_to_id: Dict[str, int] = None):
    """为高斯点云分配语义标签。
    
    参数:
        semantic_labels: 从对象ID到类别信息的映射，格式为 {obj_id: {"category": "chair", "confidence": 0.9}}
        category_to_id: 从类别名称到整数ID的映射。如果为None，则自动创建。
    """
    # 实现细节：使用投票机制将2D掩码的语义标签投影到3D高斯点云
```


### 4. 新增脚本文件

#### `assign_semantic_labels.py` - 语义标签分配脚本

**功能**：使用CLIP模型为SAM生成的掩码分配语义标签

**主要特性**：
- 输入：类别列表（如 "chair,table,person,car"）
- 处理：为每个掩码区域提取CLIP特征
- 输出：`labels.json` (对象ID到类别和置信度的映射) 和 `labels_summary.csv`

**核心算法**：
1. **CLIP特征提取**：为每个类别名称计算文本特征向量
2. **掩码区域特征提取**：对每个对象的掩码区域裁剪图像并提取CLIP视觉特征
3. **特征匹配**：计算掩码区域特征与类别文本特征的余弦相似度
4. **多数投票**：对同一对象在不同视图中的特征进行平均，提高鲁棒性

**关键代码结构**：
```python
# 1. CLIP模型加载
model, preprocess = load_clip_model(device=device, model_name=args.clip_model)
text_features = get_text_features(model, categories, device=device)

# 2. 收集所有对象ID及其掩码
object_mask_map = defaultdict(list)
# 遍历所有视图目录，解析obj_XXXX.png文件名

# 3. 为每个对象提取特征并分类
for obj_id, items in object_mask_map.items():
    # 选择面积最大的几个掩码
    features = []
    for stem, mask_path, area in selected_items:
        # 提取掩码区域特征
        feat = extract_mask_features(model, preprocess, image_path, mask, device=device)
        features.append(feat)
    
    # 平均特征并计算相似度
    avg_feature = torch.stack(features).mean(dim=0)
    sim = (avg_feature @ text_features.T).cpu().numpy()
    best_idx = np.argmax(sim)
    category = categories[best_idx]
    confidence = float(sim[best_idx])
```

**命令行参数**：
- `--source_path`：数据集根目录（包含images/和masks_sam/）
- `--categories`：逗号分隔的类别列表（如 "chair,table,sofa,bed"）
- `--clip_model`：CLIP模型名称（默认："ViT-B/32"）
- `--max_images_per_object`：每个对象考虑的最大视图数（默认：3）
- `--min_mask_area`：最小掩码面积（像素，默认：100）
- `--output_labels`：输出JSON文件名（默认："labels.json"）

**使用方式**：
```bash
python assign_semantic_labels.py \
    --source_path /path/to/data \
    --categories "chair,table,sofa,bed" \
    --clip_model "openai/clip-vit-base-patch32"
```

**输出文件格式**（`labels.json`）：
```json
{
  "1": {
    "category": "chair",
    "confidence": 0.85
  },
  "2": {
    "category": "table", 
    "confidence": 0.92
  }
}
```

#### `query_semantic_scene.py` - 语义场景查询工具

**功能**：命令行工具，支持多种查询方式

**核心类**：`SemanticSceneQuery`
- `__init__(ply_path)`：加载PLY文件并解析语义标签
- `load_category_mapping(mapping_path)`：加载类别名称到ID的映射
- `query_by_category(category_name)`：根据类别名称查询点云
- `query_by_bounding_box(bbox_min, bbox_max)`：根据3D边界框查询点云
- `query_by_sphere(center, radius)`：根据球体查询点云
- `query_combined()`：组合多种查询条件
- `save_filtered_ply(output_path, mask)`：保存过滤后的点云
- `visualize()`：可视化点云（需要matplotlib）
- `export_statistics()`：导出统计信息

**查询类型**：
1. **类别查询**：查找指定类别的所有高斯点
2. **边界框查询**：查找指定3D边界框内的点
3. **球体查询**：查找指定球体内的点

**关键代码结构**：
```python
class SemanticSceneQuery:
    def __init__(self, ply_path: str):
        self.points = None  # (N, 3) 点云坐标
        self.semantic_labels = None  # (N,) 语义标签ID
        self.opacities = None  # (N,) 不透明度
        self.scales = None  # (N, 3) 缩放
        self.rotations = None  # (N, 4) 旋转四元数
        self.features_dc = None  # (N, 3) DC特征
        self.features_rest = None  # (N, 45) 其余特征
        self.load_ply()
    
    def query_by_category(self, category_name: str) -> np.ndarray:
        category_id = self.get_category_id(category_name)
        mask = self.semantic_labels == category_id
        return mask
    
    def query_by_bounding_box(self, bbox_min: List[float], bbox_max: List[float]) -> np.ndarray:
        mask = np.all((self.points >= bbox_min) & (self.points <= bbox_max), axis=1)
        return mask
    
    def query_by_sphere(self, center: List[float], radius: float) -> np.ndarray:
        distances = np.linalg.norm(self.points - center, axis=1)
        mask = distances <= radius
        return mask
```

**命令行参数**：
- `--ply_path`：输入PLY文件路径（必需）
- `--category_mapping`：类别映射JSON文件路径
- `--stats`：显示统计信息
- `--query_category`：查询指定类别的点
- `--query_bbox`：查询边界框内的点（6个参数：x_min y_min z_min x_max y_max z_max）
- `--query_sphere`：查询球体内的点（4个参数：x y z radius）
- `--save_filtered`：保存过滤后的点云到PLY文件
- `--visualize`：可视化点云
- `--export_stats`：导出统计信息到JSON文件

**使用方式**：
```bash
# 1. 显示统计信息
python query_semantic_scene.py --ply_path scene.ply --stats

# 2. 查询所有椅子
python query_semantic_scene.py --ply_path scene.ply --query_category chair

# 3. 查询边界框内的点
python query_semantic_scene.py --ply_path scene.ply --query_bbox -1 0 -1 1 1 1

# 4. 查询球体内的点
python query_semantic_scene.py --ply_path scene.ply --query_sphere 0 0 0 2.0

# 5. 保存过滤后的点云
python query_semantic_scene.py --ply_path scene.ply --query_category chair --save_filtered chairs_only.ply

# 6. 可视化点云
python query_semantic_scene.py --ply_path scene.ply --visualize
```

#### `interactive_semantic_query.py` - 交互式语义查询界面

**功能**：交互式命令行界面，支持实时查询

**核心类**：`InteractiveSemanticQuery`
- `__init__(ply_path, category_mapping)`：初始化交互式查询环境
- `show_help()`：显示帮助信息
- `show_stats()`：显示场景统计信息
- `list_categories()`：列出所有可用类别
- `query_category()`：执行类别查询
- `query_bbox()`：执行边界框查询
- `query_sphere()`：执行球体查询
- `save_filtered()`：保存过滤后的点云
- `visualize()`：可视化当前查询结果
- `export_stats()`：导出统计信息
- `clear_screen()`：清屏
- `exit_program()`：退出程序
- `run()`：运行交互式查询主循环

**命令系统**：
```python
self.commands = {
    "help": self.show_help,
    "stats": self.show_stats,
    "categories": self.list_categories,
    "query": self.query_category,
    "bbox": self.query_bbox,
    "sphere": self.query_sphere,
    "save": self.save_filtered,
    "visualize": self.visualize,
    "export": self.export_stats,
    "clear": self.clear_screen,
    "exit": self.exit_program,
    "quit": self.exit_program,
}
```

**交互式会话示例**：
```
============================================================
交互式语义场景查询工具
============================================================
加载的点云: scene.ply
总点数: 100000
输入 'help' 查看可用命令
输入 'exit' 或 'quit' 退出程序
============================================================

查询> stats
Total points: 100000
Labeled points: 75000 (75.0%)
Unlabeled points: 25000 (25.0%)
  chair (ID 0): 1254 points (1.3%)
  table (ID 1): 987 points (1.0%)
  sofa (ID 2): 654 points (0.7%)

查询> query chair
Found 1254 points of category 'chair' (ID 0)

查询> bbox -1 0 -1 1 1 1
Found 342 points within bounding box

查询> save selected_area.ply
Saved 342 points to selected_area.ply

查询> visualize
# 打开3D可视化窗口显示当前查询结果

查询> exit
再见！
```

**命令行参数**：
- `--ply_path`：输入PLY文件路径（必需）
- `--category_mapping`：类别映射JSON文件路径

**使用方式**：
```bash
python interactive_semantic_query.py --ply_path scene.ply --category_mapping categories.json
```

**功能特点**：
1. **命令自动补全**：使用readline库支持命令行历史记录和补全
2. **实时反馈**：查询结果立即显示，支持连续查询
3. **错误处理**：输入验证和友好的错误提示
4. **可视化集成**：支持3D点云可视化（需要matplotlib）
5. **数据导出**：可保存过滤后的点云和统计信息

## 完整工作流程

### 步骤1：数据准备

1. **运行SAM掩码生成**：
   ```bash
   python generate_multiview_sam_masks.py \
       --source_path /path/to/data \
       --out_subdir sam_masks
   ```
   <!-- python generate_multiview_sam_masks.py --source_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room -->

2. **分配语义标签**：
   ```bash
   python assign_semantic_labels.py \
       --source_path /path/to/data \
       --categories "chair,table,sofa,bed,lamp"
   ```
   <!-- python assign_semantic_labels.py --source_path /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev/data/gs_data/room --categories "chair,table,sofa,lamp" --mask_subdir images_4/masks_sam --images_subdir images_4 -->

### 步骤2：3D高斯训练（带语义标签）

```bash
python train.py \
    -s /path/to/data \
    -m /path/to/model_output \
    --mask_dir /path/to/data/sam_masks \
    --semantic_labels /path/to/semantic_labels/labels.json
```

<!-- python train.py -s data/gs_data/room -m ./output/model_output_0328 --mask_dir data/gs_data/room/images_4/masks_sam --semantic_labels data/gs_data/room/images_4/masks_sam/labels.json -->

### 步骤3：语义场景查询

1. **命令行查询**：
   ```bash
   python query_semantic_scene.py \
       --ply_path /path/to/model_output/point_cloud/iteration_30000/point_cloud.ply \
       --query_type category \
       --category "chair" \
       --output_stats chair_stats.csv
   ```

2. **交互式查询**：
   ```bash
   python interactive_semantic_query.py \
       --ply_path /path/to/model_output/point_cloud/iteration_30000/point_cloud.ply
   ```

## 数据格式说明

### 语义标签文件格式 (`labels.json`)

```json
{
  "obj_0001": {
    "category": "chair",
    "confidence": 0.85,
    "view_count": 5
  },
  "obj_0002": {
    "category": "table",
    "confidence": 0.92,
    "view_count": 7
  }
}
```

### PLY文件语义标签存储

在PLY文件中，语义标签存储为额外的属性：
- 属性名：`semantic`
- 数据类型：`float` (实际存储为整数)
- 值范围：`-1` 表示未标记，`>=0` 表示类别ID

## 示例用例

### 用例1：查找场景中的所有椅子

```bash
python query_semantic_scene.py \
    --ply_path scene.ply \
    --query_type category \
    --category "chair" \
    --output_filtered chairs_only.ply \
    --visualize
```

输出：
```
找到 1254 个属于类别 'chair' 的高斯点
边界框范围: x=[-1.2, 2.3], y=[0.5, 1.8], z=[-0.3, 0.9]
已保存到: chairs_only.ply
```

### 用例2：交互式场景探索

```bash
python interactive_semantic_query.py --ply_path scene.ply
```

交互式会话示例：
```
> query chair
找到 1254 个 'chair' 点

> bbox -1 0 -1 1 1 1
找到 342 个在边界框内的点

> save selected_area.ply
已保存 342 个点到 selected_area.ply
```

### 用例3：组合查询（类别+空间约束）

```bash
# 查询客厅区域（边界框）内的所有桌子
python query_semantic_scene.py \
    --ply_path scene.ply \
    --query_category table \
    --query_bbox -2 0 -2 2 2.5 2 \
    --save_filtered livingroom_tables.ply
```

输出：
```
Found 3 points of category 'table' (ID 1)
Found 156 points within bounding box
Combined query found 3 points
Saved 3 points to livingroom_tables.ply
```

### 用例4：批量处理多个场景

```bash
#!/bin/bash
# batch_process.sh

SCENES=("scene1" "scene2" "scene3")
CATEGORIES="chair,table,sofa,bed,lamp"

for scene in "${SCENES[@]}"; do
    echo "Processing $scene..."
    
    # 1. 生成掩码
    python generate_multiview_sam_masks.py \
        --source_path "data/$scene" \
        --out_subdir sam_masks
    
    # 2. 分配语义标签
    python assign_semantic_labels.py \
        --source_path "data/$scene" \
        --categories "$CATEGORIES" \
        --output_dir "semantic_labels"
    
    # 3. 训练带语义标签的3DGS
    python train.py \
        -s "data/$scene" \
        -m "models/$scene" \
        --mask_dir "data/$scene/sam_masks" \
        --semantic_labels "data/$scene/semantic_labels/labels.json"
    
    # 4. 导出统计信息
    python query_semantic_scene.py \
        --ply_path "models/$scene/point_cloud/iteration_30000/point_cloud.ply" \
        --stats \
        --export_stats "stats/$scene_stats.json"
done
```

## 完整示例工作流程

以下是一个完整的端到端示例，展示从原始数据到语义查询的完整流程：

### 数据准备阶段
```bash
# 假设数据集位于 /data/living_room，包含 images/ 目录
cd /mnt/bn/aidp-data-3d-lf1/xxt/merlin/gs/workspace/GS_dev

# 1. 生成多视图一致的SAM掩码
python generate_multiview_sam_masks.py \
    --source_path /data/living_room \
    --out_subdir sam_masks

# 2. 为掩码分配语义标签
python assign_semantic_labels.py \
    --source_path /data/living_room \
    --categories "chair,table,sofa,TV,plant,lamp" \
    --clip_model "ViT-B/32" \
    --max_images_per_object 5

# 检查生成的标签
cat /data/living_room/images/masks_sam/labels.json | head -5
```

### 训练阶段
```bash
# 3. 训练带语义标签的3D高斯模型
python train.py \
    -s /data/living_room \
    -m /models/living_room_semantic \
    --mask_dir /data/living_room/images/masks_sam \
    --semantic_labels /data/living_room/images/masks_sam/labels.json \
    --iterations 30000

# 训练完成后，点云保存在：
# /models/living_room_semantic/point_cloud/iteration_30000/point_cloud.ply
```

### 查询与分析阶段
```bash
# 4. 分析场景统计信息
python query_semantic_scene.py \
    --ply_path /models/living_room_semantic/point_cloud/iteration_30000/point_cloud.ply \
    --stats

# 输出示例：
# Total points: 150000
# Labeled points: 120000 (80.0%)
# Unlabeled points: 30000 (20.0%)
#   chair (ID 0): 1254 points (0.8%)
#   table (ID 1): 987 points (0.7%)
#   sofa (ID 2): 654 points (0.4%)
#   TV (ID 3): 123 points (0.1%)
#   plant (ID 4): 456 points (0.3%)
#   lamp (ID 5): 321 points (0.2%)

# 5. 提取所有椅子并保存
python query_semantic_scene.py \
    --ply_path /models/living_room_semantic/point_cloud/iteration_30000/point_cloud.ply \
    --query_category chair \
    --save_filtered /output/chairs_only.ply

# 6. 交互式探索
python interactive_semantic_query.py \
    --ply_path /models/living_room_semantic/point_cloud/iteration_30000/point_cloud.ply \
    --category_mapping /data/living_room/images/masks_sam/labels.json
```

### 预期结果
1. **语义标签质量**：CLIP分类准确率通常在70-90%之间，取决于类别区分度和图像质量
2. **查询性能**：百万级点云的类别查询可在毫秒级完成
3. **存储开销**：语义标签增加约8字节/点，对于100万点的场景增加约8MB存储
4. **可视化效果**：不同类别的高斯点可用不同颜色区分，便于场景理解

## 性能考虑

1. **内存使用**：语义标签使用 `torch.long` 类型，每个点增加8字节存储
2. **查询性能**：类别查询使用向量化操作，复杂度为 O(n)
3. **CLIP推理**：`assign_semantic_labels.py` 支持批处理，可调整 `--batch_size` 参数

## 依赖项

### 新增依赖
- `transformers`：用于CLIP模型
- `Pillow`：图像处理
- `open-clip-torch`：可选的CLIP实现

### 安装命令
```bash
uv pip install transformers Pillow open-clip-torch
```

## 已知限制与未来改进

### 当前限制
1. 语义标签分配依赖于2D掩码的质量
2. 多视图一致性依赖于COLMAP的跟踪质量
3. CLIP模型可能对某些特定领域类别识别不准确

### 未来改进方向
1. **3D感知的语义分割**：直接对3D高斯点云进行语义分割
2. **多模态查询**：支持文本、图像等多种查询方式
3. **关系推理**：识别对象之间的空间和语义关系
4. **动态场景支持**：扩展到时序变化的场景

## 总结

本实现为3DGS系统添加了完整的语义理解与查询功能链，从数据准备到场景交互的全流程支持。通过结合SAM、CLIP和3DGS，实现了从像素级分割到3D语义场景理解的完整转换。

该功能为以下应用场景提供了基础：
- 智能家居场景理解
- 机器人环境感知
- AR/VR场景交互
- 三维场景检索系统