import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
  name: "ComfyUI-MakkiTools.Environment_INFO.INFO",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // 只针对Environment_INFO节点
    if (nodeData.name === "Environment_INFO") {
      const onExecuted = nodeType.prototype.onExecuted;

      nodeType.prototype.onExecuted = function (message) {
        // 调用原始onExecuted方法（如果有）
        const r = onExecuted?.apply?.(this, arguments);

        // 查找现有的info控件位置
        const pos = this.widgets.findIndex((w) => w.name === "info");

        // 清理旧控件
        if (pos !== -1) {
          // 移除名为"info"的控件
          this.widgets[pos].onRemove?.();
          this.widgets.splice(pos, 1);
        }

        // 创建新的多行文本框
        if (message.info) {
          // 处理不同格式的数据
          let infoText;
          if (Array.isArray(message.info)) {
            // 数组类型：用换行符连接元素
            infoText = message.info.join("");
          } else if (typeof message.info === "object") {
            // 对象类型：格式化为JSON字符串
            infoText = JSON.stringify(message.info, null, 2);
          } else {
            // 其他类型：直接转换为字符串
            infoText = message.info.toString();
          }

          // 创建多行文本框控件
          const w = ComfyWidgets["STRING"](
            this,
            "info",
            [
              "STRING",
              {
                multiline: true,
              },
            ],
            app
          ).widget;

          // 配置文本框属性
          w.inputEl.readOnly = true; // 设为只读
          w.inputEl.style.opacity = 0.8; // 半透明效果
          w.value = infoText; // 填充处理后的文本
        }

        // 调整节点大小以适应新控件
        this.onResize?.(this.size);

        return r;
      };
    }
  },
});
