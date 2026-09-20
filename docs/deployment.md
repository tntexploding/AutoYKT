# 首次部署与标定

本文使用 Windows PowerShell，假定已按 [README](../README.md#全新安装) 创建 `.venv`、安装项目并执行 `init`。所有命令在仓库根目录运行。下面用 `example_yuketang` 作页面 ID；可以重命名，但命令和 `runtime.active_profiles` 必须一致。

首次标定需要看到真实题目、选中状态、提交按钮和完成反馈。回顾页只能提供部分样式，不能证明正式提交按钮的位置或效果。请先保存不同状态的原尺寸截图，之后可离线完成大部分工作。

## 1. 确认私有配置

打开 `init` 显示的文件路径，设置模型密钥。保留模板的 Flash 配置、`runtime.dry_run: true`、`answering.auto_apply: false` 和关闭通知。先准备一个页面，其他页面暂时禁用。

也可显式创建到自己选择的私有目录：

```powershell
.\.venv\Scripts\python.exe -m autoykt init --path 'D:\AutoYKT-private\config.yaml'
```

使用非默认目录时，每个命令都需带 `--config`，或者在 Windows 用户环境变量中设置 `AUTOYKT_CONFIG` 为该文件的绝对路径并重新打开终端。以下命令省略 `--config`，假定已配置好发现路径。相对模板、截图和知识库路径都从这份配置的目录解析。

## 2. 绑定正式浏览器窗口

在浏览器中打开正式答题标签页，固定窗口大小、浏览器缩放、页面滚动位置和侧栏状态。执行：

```powershell
.\.venv\Scripts\python.exe -m autoykt windows
```

在输出中找到目标窗口，复制 `class_name` 以及 `client` 最后两个数（宽、高）。在私有页面配置填入：

```yaml
target_window:
  title_pattern: '<只匹配自己课堂窗口的正则表达式>'
  class_name: '<windows 输出的 class_name>'
  client_size: [1280, 800]  # 替换为实际 client 宽、高
  background_monitoring: true
  focus_point: null
```

标题模式必须唯一；`.`、`(` 等是正则符号，需按需转义。不要复制窗口句柄。`client` 前两个数是屏幕位置，不填入 `client_size`。绑定后的全部区域以**客户区左上角**为原点，使用原尺寸物理像素。窗口可以移动；尺寸或浏览器缩放变化后需重新标定。

后台监测只适用于仍完整可见的窗口，不能穿透遮挡、监测隐藏标签页或最小化窗口。输入前会请求激活；仅在需要且经过检查时，将 `focus_point` 配成客户区内不会操作控件的空白点。

## 3. 采集题目与完成状态

以下命令等待 3 秒，让你把浏览器置于前台，再保存截图；不会答题或点击：

```powershell
.\.venv\Scripts\python.exe -m autoykt capture-state --profile example_yuketang --label question
.\.venv\Scripts\python.exe -m autoykt capture-state --profile example_yuketang --label selected
.\.venv\Scripts\python.exe -m autoykt capture-state --profile example_yuketang --label success
```

分别在未作答、已有选中项、成功提交后执行对应命令，保留每次输出的 `report.json` 路径。最好还采集多选、过期、新题提示和下课状态。每个报告目录的 `full.png` 是原尺寸客户区截图。不要用聊天预览中被缩小的图片推算实际坐标。

## 4. 标定区域、字母和颜色

打开题目状态的报告：

```powershell
.\.venv\Scripts\python.exe -m autoykt calibrate --profile example_yuketang --from-state '私有采集目录\report.json'
```

按数字或字母选择模式，再依次点击左上角和右下角圈定区域。标定器展示的是静态截图；图中不发生真实点击。

| 键 | 选择内容 |
| --- | --- |
| `1` | 左上角题型的搜索范围，可容纳单选/多选标题 |
| `2` | 完整题干与全部选项文字；排除倒计时、侧栏和结果区域 |
| `3` | 全部可能出现的字母按钮及选项；可与 `2` 相同 |
| `4` | 清晰的“单选题”或“多选题”标题模板（更新第一个触发器） |
| `5` | 依次圈定 A、B、C……字母模板，支持到 Z；重新按 `5` 从 A 开始 |
| `8` | 真正的提交按钮区域 |
| `9` | “已完成”或“已过期”等反馈所在范围 |
| `0` | 用于区分前后题目的稳定内容范围，通常与 `2` 相同 |
| `g` | 正式页稳定且唯一的身份标识，例如地址栏中的固定路径前缀 |
| `b` / `u` | 已选中 / 未选中按钮的纯色背景小区域，避开白色字母与边缘 |
| `c` | 一个完整按钮的外接矩形，用来测量宽、高 |
| `v` | 明确的“已完成”状态模板，之后用成功截图标定 |
| `s` / `q` | 保存私有配置 / 退出 |

单选圆形与多选方形若使用相同字母，可只圈中央字母并保留少量纯色背景，用 `c` 记录完整按钮尺寸。初次采色时，选一张同时有蓝色选中项和灰色未选中项的截图，完成 `b`、`u`、`c` 后保存。两种颜色都采到后才能首次保存颜色配置；之后只改某一项会保留其他项。采色会启用灰度字形匹配，运行时仍用 RGB 颜色检查是否选中。

标定工具每次保存会合并本次修改，不删除未修改的选项模板或密钥。按 `s` 保存后退出，再用完成状态的报告标定 `v` 和 `9`。`v` 只圈完成提示，不能把倒计时或选中颜色当成成功证据。提交坐标应在未提交状态中用 `8` 标定。

`7` 是可选的固定题目入口，`6` 是旧版备用坐标；正式运行保持 `fallback_positions: {}`，通过字母动态定位。

### 同时支持单选和多选

当前交互标定的 `4` 修改第一个触发器。先标定一种题型并保存，再把生成的 `templates/example_yuketang/question.png` 复制为 `single.png`；之后用另一题型报告标定 `4`，把生成文件复制为 `multiple.png`。仅在私有配置中改成两条触发器：

```yaml
triggers:
  - name: single_choice
    path: templates/example_yuketang/single.png
    threshold: 0.94
    consecutive_hits: 2
    action: answer
    question_type: single
  - name: multiple_choice
    path: templates/example_yuketang/multiple.png
    threshold: 0.94
    consecutive_hits: 2
    action: answer
    question_type: multiple
```

阈值需用实际截图验证。两种题型同时匹配会暂停，不能靠匹配分数猜题型。选项模板库存应包含课程可能出现的每个字母；每题只使用当页识别到的选项。出现未标定字母或无法确认的按钮时停止本题操作。

## 5. 新题提示与其他状态

课堂失焦后不自动翻页时，需要配置新题提示入口。先采集提示，裁出紫色提示的正文部分，保存为私有 `templates/example_yuketang/new_question.png`；不包含右侧关闭按钮。在 `page_flow.before_question` 添加：

```yaml
before_question:
  - name: new_question
    when:
      path: templates/example_yuketang/new_question.png
      region: [0, 100, 1280, 700]  # 替换为客户区内提示可能出现的搜索范围
      threshold: 0.90
      match_grayscale: true
      scales: [1.0]
    click_match: true
    stable_hits: 2
```

`click_match` 点击实际匹配正文的中心，不依赖固定提示位置。通过 `preview --profile example_yuketang --step new_question` 检查计划，确认不会点到关闭按钮。没有配置这个入口时，程序只能检测当前已经显示的题目，不能保证发现浏览器尚未切入的新题。

过期模板放入 `verification.failure_templates`，范围需在 `regions.verification` 内。下课、加载、要求人工操作等状态放入相应 `page_flow` 字段，详见 [配置手册](configuration.md)。不要把普通“结束放映”直接当作课堂结束，后续可能还有课件或题目。

## 6. 核验配置并开始使用

在私有 YAML 中确认：

- 页面 `enabled: true`，`runtime.active_profiles` 为 `[example_yuketang]`。
- `submit_target.coordinate_space: window`，坐标来自真正提交按钮；`submit_on_select: false`。
- 身份模板、单选/多选题型、完整字母库存、两种按钮颜色和成功反馈都已标定。
- 一分钟题目可设 `question_budget` 为总计 50 秒、上下文 5 秒、模型 35 秒；程序还会从模型时间中预留选中和提交验证时间。将 `recovery.maximum_attempts: 1` 可避免整轮重试。

保留一组私有截图，在 `calibration_samples` 登记预期题型、选项和状态，启动检查会自动重验。写法见 [截图回归检查](configuration.md#私有截图回归检查)。至少检查新题、已完成、普通课件；有单选/多选时分别检查。

```powershell
.\.venv\Scripts\python.exe -m autoykt inspect-image --profile example_yuketang --image '私有采集目录\full.png'
.\start-class.cmd --check
.\.venv\Scripts\python.exe -m autoykt run --profile example_yuketang --once --dry-run
```

查看 `annotated.png` 和 `plan_preview.png`：字母对应正确位置，选中状态正确，提交点在实际按钮内。`run --once --dry-run` 会等待一题并调用配置的模型，但不点击；旧的已完成题会被跳过。新题只有一分钟时，应在课前完成准备，避免在倒计时中临时安装或标定。

完成检查后，启动两小时的正式运行：

```powershell
.\start-class.cmd
```

第一批真实题请在场核对“出现新题 → 选中正确集合 → 点击提交 → 平台显示完成 → 下一题恢复等待”。回放的虚拟提交只能证明本地流程。保持终端不遮挡浏览器；查看状态、主动停止和人工恢复的方法见 [课堂运行](class-session.md)。
