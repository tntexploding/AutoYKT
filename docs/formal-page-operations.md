# 正式页面操作与验收

首次安装请先完成 [部署指南](deployment.md)。本文说明较详细的现场采集、单题检查和恢复操作。

当前已实现窗口保护、定位与点击预览、状态采集、离线标定、提交后翻页、整题时间预算和中断恢复保护。前几次真实课堂按有人现场监控的方式验收。
没有正式页面时可以准备配置和验证程序逻辑；正式页面的实际布局、提交结果标识及端到端成功率仍需现场确认。

## 已实现的操作边界

- `target_window` 按窗口标题正则和可选窗口类选择唯一的 Windows 窗口。运行期间绑定窗口句柄和进程，不自动接管替换后的窗口。
- 配置中的区域以窗口**客户区**左上角为原点，使用物理像素。窗口移动后重新换算坐标；客户区尺寸与标定值不一致时停止操作。浏览器缩放、页面滚动、侧栏和响应式布局变化仍需重新核对标定。
- 默认要求窗口在前台。私有配置设置 `target_window.background_monitoring: true` 后，可持续截取未被遮挡的后台窗口；最小化或实际遮挡仍暂停。输入前请求激活窗口；Windows 拒绝时，可用私有 `focus_point` 标定的空白位置进行普通点击激活。激活后丢弃旧识别结果，重新检查题面，点击点仍须属于目标窗口。
- `page_guard` 核对稳定的页面标识；每次输入前还会检查题目内容，选择选项前同时核对选项区域。窗口标题相同不代表标签页相同，必须选择能区分正式答题页与回顾页的视觉标识。
- 答案模板出现多个明显分离的匹配位置时拒绝操作，即使配置了备用坐标也不会绕过歧义。正式启动禁止备用固定坐标，找不到模板时停止本题。
- 预览与执行使用同一份选项、入口及提交坐标计划。窗口坐标到屏幕坐标的转换在实际输入前完成。提交动作仅在选项点击后尚未出现成功状态时执行。
- **窗口绑定的真实点击必须配置 `page_guard` 和 `verification.success_templates`。** `failure_templates` 匹配后停止当前尝试；超时或无法验证也停止尝试，不重复提交同一道题。
- 旧版显示器配置保留读入和演练兼容行为；正式启动必须补齐窗口绑定、身份和成功模板。

这些保护依赖操作系统窗口信息和截图，是尽力核对，不能原子锁定网页。网页内部弹层、具有相同外观的标签页、点击后立即变化的布局仍可能需要专门适配。成功模板代表可见反馈；应在正式验收中确认它确实意味着平台已接收本次答案。

## 私有配置

以下命令使用已安装的 `autoykt`。源码环境也可以将其替换为 `.\.venv\Scripts\python.exe -m autoykt`。

新安装时运行一次 `autoykt init`；已有私有配置则继续使用原文件。Windows 默认配置位置为 `%LOCALAPPDATA%\AutoYKT\config.yaml`，也可通过 `--config` 指定。

```powershell
$quizConfig = Join-Path $env:LOCALAPPDATA 'AutoYKT\config.yaml'
autoykt windows
```

`windows` 输出可见窗口的 `title`、`class_name`、`client` 等信息；`client` 的格式为 `[屏幕 x, 屏幕 y, 宽度, 高度]`。把选定窗口的信息写入私有配置。不要把窗口句柄写入配置，它只在本次运行中使用。

为正式页面建立独立的 `pages` 条目，例如 `id: formal`，复用课程的 `course_id`。回顾页面使用另一条配置，不沿用它的区域、选项位置和成功模板。完整结构参见根目录的 `config.example.yaml`。

下列片段仅说明字段，标题、尺寸和路径均需替换为自己的采集结果：

```yaml
target_window:
  title_pattern: '<只匹配目标窗口的标题正则>'
  class_name: '<windows 命令输出的窗口类>'
  client_size: [1280, 800]  # 替换为实际客户区宽、高
page_guard:
  path: templates/formal/page_guard.png
  region: [20, 80, 200, 40]  # 由标定产生
  threshold: 0.90
answer_style:
  option_templates:
    A: templates/formal/options/A.png
    B: templates/formal/options/B.png
    C: templates/formal/options/C.png
    D: templates/formal/options/D.png
  fallback_positions: {}
  fallback_coordinate_space: window
submit_target:
  region: [900, 700, 100, 40]  # 由标定产生
  coordinate_space: window
verification:
  require_success_template: true
  success_templates:
    - path: templates/formal/success.png
      threshold: 0.90
  failure_templates: []
```

所有 `regions`、入口、提交区域和备用点都必须位于客户区以内。绑定窗口时，入口、提交和备用点的 `coordinate_space` 必须为 `window`。模板定位结果会自动使用客户区坐标。

准备阶段保留 `runtime.dry_run: true`、`answering.auto_apply: false`。状态采集和手动预览允许选择尚未启用的页面配置；`run --profile` 要求该配置已经 `enabled: true`。

## 正式页面短暂可用时先采集

状态采集不调用模型、OCR 或通知，也不点击或提交。命令默认留出 3 秒切换到目标窗口。

```powershell
autoykt capture-state --config $quizConfig --profile formal --label question --delay 3
```

`--label` 可选 `waiting`、`question`、`selected`、`submitting`、`success`、`failure`、`next`、`loading`、`manual`、`finished`。在自己实际看到对应状态时分别运行。标签是人工记录，不会自动把一张截图认定为提交成功。

每次生成独立的私有目录：

```text
<storage.data_dir>/runs/<profile_id>/<时间与随机编号>/
  report.json
  full.png
  detection.png
  question.png
  answers.png
  verification.png
  rearm.png
```

命令输出报告和完整截图路径。截图包含所选客户区或显示器内容，请保存在私有数据目录。采集阶段不要求题目、选项或成功模板已经存在；现有区域需在采集范围内。首次可以保留模板中的小区域，之后使用完整截图重新标定。

## 页面关闭后离线标定

使用上一步输出的 `report.json`，不会访问桌面或调用模型：

```powershell
autoykt calibrate --config $quizConfig --profile formal --from-state '<私有采集目录>\report.json'
```

标定器检查报告所属配置、窗口参数和截图尺寸，避免把另一页面的采集记录误用过来。一次选择两个角点圈定区域。

| 键 | 用途 |
| --- | --- |
| `1` / `2` / `3` | 检测区域 / 题目截图区域 / 选项搜索区域 |
| `4` | 触发模板 |
| `5` / `6` | A–Z 选项模板 / 旧版备用点击点 |
| `7` / `8` | 入口区域 / 提交区域 |
| `9` / `0` | 结果验证区域 / 下一题识别区域 |
| `g` | 页面标识模板和对应区域 |
| `v` | 成功模板 |
| `b` / `u` / `c` | 选中底色 / 未选中底色 / 完整按钮尺寸 |
| `s` / `q` | 保存私有配置 / 退出 |

分别用题目状态和成功状态的采集报告完成相应标定。失败模板可以从失败状态截图裁剪后，写入 `failure_templates`。成功、失败模板必须能在 `regions.verification` 中找到。

`g` 应选择稳定、能区分页面的内容，例如正式页面专有的标题或页面路径显示；避免计时器、题号和答题高亮。`regions.rearm` 应覆盖题目身份信息，尽量避开选择高亮和提交反馈。`regions.question` 应包含模型解题所需的题干和完整选项；回顾页不要混入正确答案或解析。

也可以直接运行 `calibrate --profile formal --delay 3` 抓取当前窗口。窗口绑定模式下，标定界面本身会成为前台窗口，因此刷新另一个实时状态时应关闭后重新运行，或使用新的状态采集报告。离线标定中的 `r` 恢复当前保存的截图。

## 先看点击计划

手动把正式页面打开到题目，指定一个选项检查位置，不需要模型密钥：

```powershell
autoykt preview --config $quizConfig --profile formal --option B --delay 3
```

查看输出目录中的 `plan_preview.png` 和 `report.json`。图中的红色十字和编号表示选项及条件提交位置；报告记录坐标系、预览区域及计划中的精确点。

需要验证 AI 答题到定位的流程时：

```powershell
autoykt run --config $quizConfig --profile formal --once --dry-run
```

`--dry-run` 只对本次运行生效，不改写配置，并禁止所有鼠标动作。若配置了入口点击，该模式生成 `entry_preview.png` 后结束，不对尚未打开的题目调用模型。要测试已手动打开的题目，可用独立的题目状态配置，把入口设为 `null`、触发模板设为题目页的对应样式。

## 真实单题验收

拿到正式页面并完成以上检查后，在私有配置中设置：

```yaml
runtime:
  dry_run: false
answering:
  auto_apply: true
```

保留各节的其他字段，再执行：

```powershell
autoykt check --config $quizConfig
autoykt run --config $quizConfig --profile formal --once
```

`--once` 在一次答题尝试结束后退出，拒答、定位失败、验证失败也会结束。它不会因为失败继续寻找下一题；等待触发期间可用 Ctrl+C 退出。多个配置同时启用时，必须用 `--profile` 选择一个。

报告记录题目截图路径、模型最终选项、操作计划、实际点击坐标、选择前后截图和验证证据。结束状态的含义：

| 状态 | 含义 | 单题退出码 |
| --- | --- | --- |
| `verified` | 达到了该配置的提交验证规则 | 0 |
| `preview_only` / `entry_preview` | 只生成预览，没有点击 | 0 |
| `captured` | 仅完成检测截图 | 0 |
| `answer_rejected` / `failed` | 当前尝试未能完成或无法确认结果 | 1 |
| `manual_required` | 暂停操作，等待人工检查 | 1 |
| `class_finished` | 配置的课堂结束状态连续出现 | 0 |
| 配置错误 / Ctrl+C | 未开始或被用户中断 | 2 / 130 |

失败状态并不等于平台一定没有收到答案。选择或提交已尝试、结果无法确认时，本页面进入人工接管状态，即使页面随后变化也不会继续输入；全部活动页面都结束或需要人工处理时，监控命令退出。空闲时的窗口失焦、遮挡等暂停会在条件恢复后继续检测。

正式验收应观察：题目识别正确、选项定位正确、实际选中、平台确认提交成功，以及下一道题能重新触发。单题模式用于前四项；验证连续流程时再去掉 `--once`，观察至少两道不同题目及一次窗口失焦恢复。已支持标定过的可见窗口单选和多选流程；隐藏网页操作、填空和自动处理未标定弹层不在支持范围内。连续课堂测试可使用[两小时启动入口](class-session.md)。

## 结果页与下一题

在私有页面配置的 `page_flow` 中登记结果页操作。每项的 `when` 是一个带区域的状态模板，`target` 是按钮的位置。下列坐标仅说明格式，必须替换为正式页面的标定结果：

```yaml
page_flow:
  after_submit:
    - name: close_result
      when:
        path: templates/formal/close_result.png
        region: [760, 480, 240, 120]
        threshold: 0.93
      target:
        region: [840, 540, 100, 40]
        coordinate_space: window
      stable_hits: 2
    - name: next_question
      when:
        path: templates/formal/next_question.png
        region: [800, 640, 240, 100]
        threshold: 0.93
      target:
        region: [900, 690, 100, 40]
        coordinate_space: window
      stable_hits: 2
  loading: []
  manual: []
  finished: []
  transition_timeout_seconds: 10
```

从采集的 `full.png` 裁剪状态模板并存入私有模板目录，记录裁剪区域和按钮区域。`when` 应能区分结果页与答题页；不要仅使用每个页面都出现的通用“下一步”图标。`loading`、`manual`、`finished` 使用同样的 `{path, region, threshold}` 格式，例如将登录失效或题目过期的标识放入 `manual`。

运行规则：

- 只有明确成功模板验证通过后才执行 `after_submit`；配置了这些动作的真实运行必须提供成功模板。
- 根据当前画面选择命中的动作，不要求每次出现所有按钮。同一时刻命中多个动作时暂停，避免猜测。
- 每个动作在一道题内最多尝试一次。点击后等待该状态消失；超时、已执行动作重新出现或输入失败时请求人工处理。
- `loading` 阻止答题和输入，消失后继续；持续超过 `transition_timeout_seconds` 时请求人工处理。
- `manual` 立即暂停本页面。`finished` 连续出现两次后结束本页面监控。
- 结果反馈消失且满足原有 `rearm` 条件后，才恢复下一题检测。自动换题的页面可保持 `after_submit: []`。

首次遇到相应结果页时，可以先单独预览按钮，不调用 AI，也不点击：

```powershell
autoykt preview --config $quizConfig --profile formal --step close_result --delay 3
autoykt preview --config $quizConfig --profile formal --step next_question --delay 3
```

查看生成的 `plan_preview.png`。真实执行也会保存每一步的 `advance_<name>_preview.png`、操作结果截图和实际输入坐标。

## 时间预算与有限重试

每个页面可单独调整：

```yaml
question_budget:
  total_seconds: 45
  context_seconds: 5
  answer_seconds: 25
recovery:
  maximum_attempts: 2
  delay_seconds: 1
```

`total_seconds` 从触发题目开始，包含入口、题面稳定、OCR、知识检索、模型答题、选择和验证。它不是网页倒计时的读取结果；应依据实际题目限时和触发延迟留出余量。结果页到下一题的等待由 `page_flow` 和 `rearm` 控制。

OCR/知识检索超过 `context_seconds` 且配置了视觉模型时可降级为截图输入；默认纯文本 Flash 无法在没有可用 OCR 时作答。模型请求和排队共用 `answer_seconds`，同时预留定位、逐项选中检查和 `verification.timeout_seconds + 1` 秒的提交验证时间；到期取消未完成的模型请求，只对按时返回的答案计算原有共识门槛，不降低最低票数。余额不足时不开始点击。

只有所有模型响应都属于可重试的临时连接、超时、限流或服务错误时才重试。`maximum_attempts` 包含首次请求；所有尝试共享同一答题截止时间。重试前重新核对页面和原题，入口、选项、提交不会跟随请求重试。平票、无效输出、定位失败和无法验证的提交交给现场人员判断。

## 现场监控与重启

完成单题验收后，启动连续运行：

```powershell
autoykt run --config $quizConfig --profile formal
```

前几次课堂保持目标窗口在前台，观察至少两道题的识别、实际选中、平台成功反馈和结果页跳转。控制台会输出当前状态、暂停原因和报告路径；启用的通知后端也会接收错误事件。需要接手时按 Ctrl+C，等监控命令结束后再操作页面。

每次真实输入前都会保存私有 `data/runs/<profile_id>/pending.json`。重启规则：

- 已验证提交：恢复原题的识别基线和已执行的结果页动作，等待离开旧题，不重新答题。
- 输入结果不明确：拒绝启动真实运行，输出原报告路径。不要通过删除记录来绕过检查。
- 已变更页面配置或原证据损坏：要求人工复核，避免套用旧坐标和旧基线。

人工核对网页与报告、完成必要处理后，先停止该配置的所有运行进程，再记录“跳过当前页”：

```powershell
autoykt recover --config $quizConfig --profile formal --skip-current --delay 3
autoykt run --config $quizConfig --profile formal
```

`recover` 仅保存当前页面的基线与人工处理记录，不调用模型、不点击、不补提交。重启后会等当前页改变或触发样式消失，才处理之后的题目；如果此时已经显示新题，这道新题也会被跳过，因此请在希望跳过的页面执行。普通 `preview`、状态采集和 `--dry-run` 不修改真实运行的待核对记录。

当前同一桌面只运行一个控制进程；多个已配置页面串行检查，共用鼠标。隐藏窗口操作、自动登录、系统锁屏恢复及多个页面同时抢答不在支持范围内。

## 实现与验证说明

全新部署支持 Python 3.10–3.12，推荐经过独立安装验证的 3.12。当前审计与验证结果见 [阶段审计记录](audit-2026-09-20.md)，早期结果保留在 [历史审计](audit-2026-09-05.md)。

回归检查覆盖合成图像、假窗口和虚拟鼠标下的定位、选中、提交反馈、超时、下一题及重启恢复。这些检查不能替代正式网页和平台提交结果的验收。实际运行应使用本机已登录的 Windows 桌面会话。

窗口客户区与输入命中语义参考 Microsoft 的 [GetClientRect](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-getclientrect) 和 [WindowFromPoint](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-windowfrompoint) 文档。个人配置、截图、运行报告和课程数据继续位于私有目录，仓库只保存代码、配置模板与合成测试。


## 从已有图片做离线标定

只有截图时，可使用 `calibrate --from-image`。它不会连接浏览器或截图设备，
也不会把图片像素当成已经确认的桌面坐标。保存后该页面保持 `enabled: false`；
真实运行前仍需核对窗口客户区、缩放和提交控件。已有窗口配置时，图片尺寸必须
与 `client_size` 一致。图片包含浏览器边框时，应先确认它与客户区的对应关系。

```powershell
autoykt calibrate --config <个人配置.yaml> --profile <页面ID> --from-image <图片.png>
```

也可加 `--selections <个人裁剪.json>`，直接应用已核对的裁剪框而不打开标定窗口。
JSON 中包含 `regions`（与页面配置相同）、`trigger`、`options`，以及可选的
`success`、`page_guard`、`selected_sample`、`unselected_sample`。所有裁剪框
均为原图像素 `[x, y, width, height]`；`options` 是字母到裁剪框的映射，两个
颜色样本字段分别填写图中已选、未选按钮的字母。完整导入会替换该页的选项模板
并清空固定坐标回退，其他页面配置保持原样。交互标定的选项循环支持 A–Z；颜色与尺寸可用 `b`、`u`、`c` 采集。

裁剪应紧贴字母按钮，包含底色和字母，不包含旁边的题目文字。对颜色变化的按钮，
`answer_style.match_grayscale` 可用字母形状定位，`button_colors` 则根据裁剪
样本的底色判断已选和未选。位置、排列和可见数量可以变化，但按钮尺寸、字形
和截图缩放需与标定一致。重复字母、两个模板占用同一个位置或陌生底色均不能
视为已确认的选中状态。这些观察结果用于多选执行前的完整性检查。

用项目检查图片中的识别结果，不需要模型密钥：

```powershell
autoykt inspect-image --config <个人配置.yaml> --profile <页面ID> --image <图片.png>
```

输出位于该配置的私有 `data/runs/<页面ID>/` 中，包括原图、题目区域、带框标注
和 `report.json`。报告中的坐标明确属于图片，`selected` 为 true/false/null，
页面状态区分 `completed`、`question`、`failure`、`ambiguous` 和 `unrecognized`。
`question` 仅表示题型模板可见且没有完成/失败标志，不代表当前可提交。
报告不会触发模型、鼠标操作或修改运行恢复记录。

“已完成”截图可用于完成状态模板；不能据此推定倒计时、提交按钮、单选题标题
或新题弹窗的样式。缺失状态应从对应实际画面补采。离线识别通过也不等于多选
连续点击、反选、提交及其反馈已完成实测。


### 单选、多选与按钮状态

在每个题型标题模板的 `triggers` 项中指定 `question_type: single` 或
`question_type: multiple`，默认是单选。题型取决于实际匹配的标题模板，
不能用网页内部的 CSS 类名推断；同一平台可能复用类名。

启用 `answer_style.button_colors` 后，系统从当前截图识别可见选项及其选中
状态，再把实际出现的选项提供给模型。配置应覆盖该样式所有可能出现的字母。
圆形、方形或不同缩放比例的按钮需要对应的标定模板；截图检查通过不代表
桌面坐标已确认。Edge 插件截图使用浏览器视口坐标，不能直接用作 Windows
桌面或窗口客户区坐标。

多选回答使用 `{"answer":["A","C"],"confidence":0.9}`，按整组选项投票。
系统先反选多余项，再选中缺少项，保留已经正确选中的项。每次点击后在
`selection_timeout_seconds` 内核对颜色状态；按钮移动、出现歧义、题目文字
变化或点击未生效时停止操作。整组状态正确后才提交。多选正式运行要求颜色
标定、选项模板、提交目标和明确的成功模板，不能依赖固定选项坐标兜底。

`answering.minimum_confidence` 默认是 `0.6`。低于门槛或未提供置信度的回答
不参与投票；无法读清题目时模型应返回 `answer: null`。模型报告的置信度
只是提交条件之一，不代表已证明答案正确。自定义提示词也须输出上述格式。

提供方省略 `max_output_tokens` 时默认是 `2048`；当前 Flash 配置模板显式设为
`8192`，为可能计入输出额度的推理内容留出空间。
如果接口返回截断状态，报告显示 `output token limit reached` 并拒绝使用
该回答。仍可在私有配置中按模型调整 `max_output_tokens`。建议保留题目 OCR；
文本模型仅收到 OCR 和课程资料；视觉模型还会收到题图。


不同输入类型的模型须分别配置 provider。`input_mode: text` 只发送 OCR 文字，
且要求每个可见选项有明确的 `A: 文本` 字母映射；OCR 未能提供完整映射时拒绝
调用该文字模型。`input_mode: vision` 发送截图和可用的 OCR。OCR 会保留文本行
的位置，并结合识别到的字母按钮匹配选项文字，不依赖 A/B/C 的排列顺序。

`reasoning_effort` 默认不发送；支持该参数的接口可以在私有配置中选择推理
预算。例如 [DeepSeek 的接口说明](https://api-docs.deepseek.com/guides/thinking_mode/)
提供 `low`，可在课堂限时场景验证其延迟与准确性。模型名称和输入能力须匹配，
参见其 [视觉输入说明](https://api-docs.deepseek.com/guides/vision/)；不能把仅支持
文字的模型当作视觉模型使用。


## 后台新题提示

`page_flow.before_question` 在等待和上一题结束后的状态中监测新题入口。结构与 `after_submit` 相同，可以配置固定 `target`，也可以设置 `click_match: true` 并省略 `target`，点击唯一匹配图案的中心。模板只截取提示主体，排除关闭按钮。输入前再次定位；位置改变、多个匹配或提示未消失时不重复点击。

状态模板的 `scales` 可配置已知显示比例（默认 `[1.0]`）；`match_grayscale: true` 可加快大区域搜索。坐标和截图保存在私有配置目录。新题提示首次命中后即开始整题预算，入口等待占用同一预算，不在进入题目后重新计时。真实运行仍需验证弹窗出现、进入题面、选择、提交和成功反馈的完整流程。
