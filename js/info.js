import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";
const TARGET_NODES = new Set([
  "Environment_INFO_makki",
  "show_type_makki",
  "timer_makki",
  "UniversalInstaller_makki",
]);

app.registerExtension({
  name: "ComfyUI-MakkiTools.Environment_INFO.INFO",
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // 从TARGET_NODES获取节点名称
    if (TARGET_NODES.has(nodeData.name)) {
      // 添加持久化处理所需的符号
      const INFO_VALUE = Symbol("infoValue");

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

          // 保存文本值到节点实例
          this[INFO_VALUE] = infoText;

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

      // 配置方法重写
      const configure = nodeType.prototype.configure;
      nodeType.prototype.configure = function () {
        // 保存当前的info值
        const config = arguments[0];
        if (this[INFO_VALUE] && config) {
          if (!config.widgets_values) {
            config.widgets_values = [];
          }
          // 将info值添加到配置末尾
          config.widgets_values.push(this[INFO_VALUE]);
        }
        return configure?.apply(this, arguments);
      };

      // 配置加载处理
      const onConfigure = nodeType.prototype.onConfigure;
      nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);

        if (this.widgets_values?.length) {
          // 从末尾获取保存的info值
          const savedInfoValue =
            this.widgets_values[this.widgets_values.length - 1];

          // 延迟执行确保UI就绪
          requestAnimationFrame(() => {
            if (savedInfoValue) {
              // 清理现有控件
              const pos = this.widgets.findIndex((w) => w.name === "info");
              if (pos !== -1) {
                this.widgets[pos].onRemove?.();
                this.widgets.splice(pos, 1);
              }

              // 创建控件并恢复值
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
              w.value = savedInfoValue; // 填充处理后的文本

              // 保存到节点实例
              this[INFO_VALUE] = savedInfoValue;

              // 调整节点大小
              this.onResize?.(this.size);
            }
          });
        }
      };
    }
  },
});
