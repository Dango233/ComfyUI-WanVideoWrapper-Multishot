# 重构文档：MultiShot（Holocine Shot Attention）在 `reimplement/upstream-2026-01-21` 的引入

> 目标：在 `reimplement/upstream-2026-01-21` 上，以 **upstream/Kijai 引入新功能的风格** 重新引入 main 分支的 MultiShot 功能；遵循 **最小干涉** 原则，不影响 upstream 现有节点正常工作；新节点与旧实现（当前 main）**结构一致兼容**。

## 0. 基线信息（必读）
- 当前工作分支：`reimplement/upstream-2026-01-21`
- 分叉点（merge-base）：`393fe78ec2d5a515ce4ff794067ea4d72c830643`
- 对比范围：`393fe78..main`
- MultiShot 涉及的文件（来自 main 侧 diff）：
  - `__init__.py`
  - `custom_linear.py`
  - `nodes.py`
  - `nodes_model_loading.py`
  - `nodes_sampler.py`
  - `wanvideo/modules/attention.py`
  - `wanvideo/modules/model.py`
  - `wanvideo/modules/shot_utils.py`（新文件）
  - `wanvideo/modules/tokenizers.py`
  - `wanvideo/modules/t5.py`

> 参考 diff（按文件查看）：
> - `git diff 393fe78..main -- <file>`

---

## 1. 设计原则（按 upstream/Kijai 风格落地）
- **最小干涉**：优先新增节点/函数；尽量不改动 upstream 既有逻辑，除非必须。
- **可选启用**：shot attention/shot mask/shot embedding 必须显式启用，不影响默认行为。
- **兼容老结构**：节点类名、输入输出类型、返回顺序、CATEGORY/显示名与 main 保持一致。
- **显式校验与错误提示**：对缺失参数/不匹配配置给出明确错误信息。

### 1.1 贴近 Kijai 风格的补充要求（相对“现在 main”的实现）
**目标：** 在不改变节点结构与行为的前提下，将实现方式更贴近 upstream/Kijai 的“可选注入 + 低侵入 + 清晰失败”的风格。

- **默认路径零干扰**
  - MultiShot 相关功能全部通过显式节点启用（`WanVideoHolocineSetShotAttention`）。
  - 不在默认采样路径上做隐式开关或自动推断。

- **配置注入集中化**
  - 所有 shot attention 相关配置仅放在 `transformer_options["shot_attention"]` 内统一解析。
  - 不在 sampler/model 内分散多处写入同类参数，避免“隐式状态”。

- **清晰失败/回退路径**
  - 缺少 `[global caption]`、`holocine_args`、`text_cut_positions` 直接报错并说明原因。
  - sparse backend 不可用时明确降级并记录日志（而不是 silent fallback）。

- **模块边界明确**
  - 结构化 prompt、shot indices、cross-attn mask 的核心逻辑集中在 `shot_utils.py`。
  - sampler/model 仅接入，不扩散逻辑。

- **日志与校验一致**
  - 使用 log.info/warning/error，避免 print。
  - 对 LoRA 类型、token ratio、mask 通道进行严格校验并提供可读提示。

---

## 2. MultiShot 功能拆解（总览）

### 2.1 节点层（Node API）
**新增节点（main 里的定义）**：
1) `WanVideoHolocineShotArgs`
2) `WanVideoHolocineShotBuilder`
3) `WanVideoHolocinePromptEncode`
4) `WanVideoHolocineSetShotAttention`

**节点兼容性清单**（必须完全一致）：
- [ ] 类名、`NODE_CLASS_MAPPINGS` key 与 main 一致
- [ ] `CATEGORY` 与 `NODE_DISPLAY_NAME_MAPPINGS` 一致
- [ ] 输入/输出类型名一致（如 `HOLOCINE_SHOTARGS`, `WANVID_HOLOCINE_SHOT_LIST`）
- [ ] 参数默认值、tooltip、可选项范围一致
- [ ] 输出顺序与 main 一致

**节点 I/O 结构速查（以 main 为准）**：
- `WanVideoHolocineShotArgs`
  - 输入：`image_embeds(WANVIDIMAGE_EMBEDS)`, `shot_cut_frames(STRING)`
  - 输出：`holocine_args(HOLOCINE_SHOTARGS)`
  - 行为：推断总帧数 + 解析/规整 shot cut 帧（4t+1）

- `WanVideoHolocineShotBuilder`
  - 输入：`shot_caption(STRING)` + 可选 `shot_list(WANVID_HOLOCINE_SHOT_LIST)`, `shot_lora(WANVIDLORA)`, `smooth_window(INT)`
  - 输出：`shot_list(WANVID_HOLOCINE_SHOT_LIST)`
  - 行为：链式构建 shot list；每 shot 可挂 LoRA 和 smooth_window

- `WanVideoHolocinePromptEncode`
  - 输入：`global_caption(STRING)`, `shot_list(WANVID_HOLOCINE_SHOT_LIST)`, `negative_prompt(STRING)`, `t5(WANTEXTENCODER)`, `image_embeds(WANVIDIMAGE_EMBEDS)`
  - 可选：`custom_shot_cut_frames(STRING)`, `append_shot_summary(BOOLEAN)`, `force_offload(BOOLEAN)`, `model_to_offload(WANVIDEOMODEL)`, `use_disk_cache(BOOLEAN)`, `device(gpu/cpu)`
  - 输出：`text_embeds(WANVIDEOTEXTEMBEDS)`, `holocine_args(HOLOCINE_SHOTARGS)`, `positive_prompt(STRING)`
  - 行为：生成结构化 prompt + 计算 shot_cuts + 产生 holocine_args（含 shot_loras & smooth_windows）

- `WanVideoHolocineSetShotAttention`
  - 输入：`model(WANVIDEOMODEL)`, `enable(BOOLEAN)`, `global_token_ratio_or_number(FLOAT)`
  - 可选：`mask_type(none/id/normalized/alternating)`, `backend(full/sparse_fallback/sparse_flash_attn)`, `i2v_mode(BOOLEAN)`
  - 输出：`model(WANVIDEOMODEL)`
  - 行为：写入 `transformer_options["shot_attention"]`

> 参考 diff：`git diff 393fe78..main -- nodes.py`

---

### 2.2 结构化 prompt（MultiShot 语义入口）
- Prompt 约定标签：`[global caption]`, `[per shot caption]`, `[shot cut]`
- 文本解析：`parse_structured_prompt()` 解析 offset_mapping 得到 token span
- 产物：`text_embeds["text_cut_positions"]`

> 参考 diff：
> - `git diff 393fe78..main -- wanvideo/modules/shot_utils.py`
> - `git diff 393fe78..main -- nodes.py`
> - `git diff 393fe78..main -- wanvideo/modules/tokenizers.py`
> - `git diff 393fe78..main -- wanvideo/modules/t5.py`

---

### 2.3 Shot Attention（self-attn + cross-attn）
- self-attn：`sparse_shot_attention()` + fallback
- cross-attn：`build_cross_attention_mask()` 构建 per-shot mask
- backend：`full | sparse_fallback | sparse_flash_attn`
- smooth_windows：相邻 shot 共享 token
- i2v_mode：首帧作为 global prefix

> 参考 diff：
> - `git diff 393fe78..main -- wanvideo/modules/attention.py`
> - `git diff 393fe78..main -- wanvideo/modules/model.py`
> - `git diff 393fe78..main -- nodes_sampler.py`
> - `git diff 393fe78..main -- wanvideo/modules/shot_utils.py`

---

### 2.4 Per-shot LoRA（按镜头挂 LoRA）
- `CustomLinear` 增加 per-shot LoRA 注入点
- sampler 将 LoRA 拆分为 shot payload
- 仅支持 **线性 LoRA**；非线性/DoRA/reshape 直接跳过并提示

> 参考 diff：
> - `git diff 393fe78..main -- custom_linear.py`
> - `git diff 393fe78..main -- nodes_sampler.py`
> - `git diff 393fe78..main -- wanvideo/modules/model.py`

---

### 2.5 Shot Embedding + Shot Mask Channel（模型输入增强）
- load 阶段检测 `shot_embedding.weight`
- shot mask 类型：`none | id | normalized | alternating`
- 若 checkpoint 需要额外通道但 mask_type=none，必须报错

> 参考 diff：
> - `git diff 393fe78..main -- nodes_model_loading.py`
> - `git diff 393fe78..main -- wanvideo/modules/model.py`

---

## 3. 具体重构 Checklist（可按步骤逐项验证）

### A. 新增 shot_utils 基础能力
- [x] 添加 `wanvideo/modules/shot_utils.py`（保持 main 内容）
- [x] `__init__.py` 中做 `sys.modules.setdefault` 兼容映射
- [x] `parse_structured_prompt()` 可处理 fast/slow tokenizer offsets

**Diff 指向**：
- `git diff 393fe78..main -- wanvideo/modules/shot_utils.py`
- `git diff 393fe78..main -- __init__.py`

---

### B. 节点：Holocine Shot / Prompt / SetShotAttention
- [x] `WanVideoHolocineShotArgs` 添加并保持 I/O 一致
- [x] `WanVideoHolocineShotBuilder` 支持 shot_list 链式构建 + LoRA + smooth_window
- [x] `WanVideoHolocinePromptEncode`：结构化 prompt + holocine_args 输出
- [x] `WanVideoHolocineSetShotAttention`：写入 shot_attention config
- [x] NODE 映射、显示名、CATEGORY 完全对齐
- [x] prompt 标签格式与 main 一致（`[global caption]`, `[per shot caption]`, `[shot cut]`）
- [x] `text_cut_positions` 写入 `text_embeds` 字典

**Diff 指向**：
- `git diff 393fe78..main -- nodes.py`

---

### C. Sampler 注入（shot_indices + text_cut_positions）
- [x] `nodes_sampler.py` 接收 `holocine_args` 并校验
- [x] `build_shot_indices()` 构建 shot_indices
- [x] 将 `shot_indices/shot_attention_cfg/shot_mask_type/text_cut_positions/smooth_windows` 注入 model
- [x] CFG negative pass 时临时关闭 shot attention

**Diff 指向**：
- `git diff 393fe78..main -- nodes_sampler.py`

---

### D. 模型层 shot attention + cross-attn mask
- [x] `wanvideo/modules/attention.py` 加入 `sparse_shot_attention` + fallback
- [x] `wanvideo/modules/model.py` 解析 `shot_attention_cfg`
- [x] cross-attn mask 逻辑接入 `build_cross_attention_mask()`
- [x] `shot_embedding` 与 `shot_mask` 拼接逻辑完整
- [x] `nodes_model_loading.py` 识别 `shot_embedding.weight` 并写入模型配置

**Diff 指向**：
- `git diff 393fe78..main -- wanvideo/modules/attention.py`
- `git diff 393fe78..main -- wanvideo/modules/model.py`

---

### E. Per-shot LoRA 支持
- [x] `custom_linear.py` 增加 shot_lora cache 与注入
- [x] `nodes_sampler.py` 将 LoRA payload 按 shot 聚合并注入 transformer
- [x] `wanvideo/modules/model.py` 设定 `CustomLinear.runtime_context`

**Diff 指向**：
- `git diff 393fe78..main -- custom_linear.py`
- `git diff 393fe78..main -- nodes_sampler.py`
- `git diff 393fe78..main -- wanvideo/modules/model.py`

---

### F. Tokenizer 兼容与日志
- [x] `tokenizers.py` fast 失败 fallback slow
- [x] `t5.py` 日志风格与 upstream 一致

**Diff 指向**：
- `git diff 393fe78..main -- wanvideo/modules/tokenizers.py`
- `git diff 393fe78..main -- wanvideo/modules/t5.py`

---

### Z. 文档与流程
- [x] 建立并维护 MultiShot 重构文档（含 checklist 与实施记录）

## 4. 关键风险点（实现时重点检查）
- **不破坏默认路径**：shot attention 必须显式开启，默认 `None` 不影响 upstream 行为。
- **prompt tag 缺失**：缺少 `[global caption]` 应明确报错，避免 silent failure。
- **mask channel**：模型需要额外通道时，mask_type=none 必须报错（防止 shape 不一致）。
- **LoRA 兼容性**：非线性 LoRA 必须跳过且告警，避免 silent wrong output。

---

## 5. 建议的验证步骤（每一步完成后执行）
1. 仅添加 `shot_utils.py` + `__init__.py` 映射，跑一次导入测试：
   - `python -c "import wanvideo.modules.shot_utils"`
2. 添加节点并注册，启动 ComfyUI 检查节点是否出现、输入输出是否一致。
3. 使用 main 的示例 prompt（带 `[global caption]` 等标签）跑一次：
   - 仅文本编码（不启用 shot_attention）
4. 开启 `WanVideoHolocineSetShotAttention` 后跑短视频，确认：
   - shot_indices 计算、cross-attn mask 构造无异常
5. 添加 per-shot LoRA，观察日志中 LoRA 注入与跳过提示是否合理。

---

## 6. 备注
- 本文档仅涵盖 **MultiShot（Holocine）相关功能**；其余 main 分支新增功能不在本次范围。
- 若需要扩展到其他功能，请另行补充拆解文档。

---

## 7. 实施记录
- 2026-01-21 Commit 1：新增 `docs/multishot_refactor_plan.md`，删除旧文档 `docs/reimplement_main_since_upstream.md`（文档基线确立）。**Checklist 更新：** `Z. 文档与流程` 完成。
- 2026-01-21 Commit 2：新增 `wanvideo/modules/shot_utils.py` 并在 `__init__.py` 建立 `wanvideo.modules.shot_utils` 映射。**Checklist 更新：** `A. 新增 shot_utils 基础能力` 完成。
- 2026-01-21 Commit 3：`tokenizers.py` 增加 fast->slow fallback，`t5.py` 统一日志输出。**Checklist 更新：** `F. Tokenizer 兼容与日志` 完成。
- 2026-01-21 Commit 4：新增 Holocine 节点与结构化 prompt 流程，`WanVideoTextEncode` 输出 `text_cut_positions`。**Checklist 更新：** `B. 节点：Holocine Shot / Prompt / SetShotAttention` 完成。
- 2026-01-21 Commit 5：`custom_linear.py` 增加 per-shot LoRA 注入与 runtime_context 支持。**Checklist 更新：** `E.custom_linear.py` 完成。
- 2026-01-21 Commit 6：`wanvideo/modules/attention.py` 新增 sparse shot attention 与 fallback。**Checklist 更新：** `D.attention.py` 完成。
- 2026-01-21 Commit 7：`wanvideo/modules/model.py` 接入 shot attention、cross-attn mask、shot embedding/mask 与 runtime_context。**Checklist 更新：** `D.model.py` 与 `E.runtime_context` 完成。
- 2026-01-21 Commit 8：`nodes_model_loading.py` 增加 shot embedding 权重检测并传入模型配置。**Checklist 更新：** `D.nodes_model_loading.py` 完成。
- 2026-01-21 Commit 9：`nodes_sampler.py` 接入 holocine_args、shot attention 参数、shot_indices/smooth_windows 与 per-shot LoRA，CFG negative pass 暂时关闭 shot attention。**Checklist 更新：** `C. Sampler 注入`、`E.nodes_sampler.py` 完成。
