import { app } from "../../scripts/app.js";

// 核心动态连接管理函数
export const manageDynamicInputs = (
  node,
  index,
  connected,
  connectionPrefix = "input_",
  connectionType = "IMAGE"
) => {
  // 初始化动态输入集合
  if (!node._isRestoring) {
    node._dynamicInputs = new Set();
  }

  // 移除未使用的动态输入
  const removeUnusedInputs = () => {
    // 确保至少保留一个动态输入
    const minDynamicInputs = 1;

    for (let i = node.inputs.length - 1; i >= 0; i--) {
      const input = node.inputs[i];
      const isDynamic = node._dynamicInputs?.has(input.name);

      // 保留连接的输入和恢复中的输入
      if (input.link || (isDynamic && node._isRestoring)) continue;

      // 只删除多余的动态输入（确保最小数量）
      if (isDynamic && node._dynamicInputs.size > minDynamicInputs) {
        node.removeInput(i);
        node._dynamicInputs.delete(input.name);
      }
    }
  };

  const prevRestoring = node._isRestoring;

  // 恢复模式处理
  if (node._isRestoring) {
    node._dynamicInputs = new Set(
      node.inputs
        .filter((input) => input.name.startsWith(connectionPrefix))
        .map((input) => input.name)
    );
  }

  removeUnusedInputs();

  // 添加新输入（当最后一个输入被连接时）
  const lastInput = node.inputs[node.inputs.length - 1];
  if (lastInput?.link && !node._isRestoring) {
    const newIndex =
      node.inputs.filter((input) => input.name.startsWith(connectionPrefix))
        .length + 1;
    const newName = `${connectionPrefix}${newIndex}`;

    // 避免重复添加
    if (!node._dynamicInputs.has(newName)) {
      const newInput = node.addInput(newName, connectionType);
      node._dynamicInputs.add(newName);
    }
  }

  // 重新编号所有动态输入
  let counter = 1;
  node.inputs.forEach((input) => {
    if (input.name.startsWith(connectionPrefix)) {
      const newName = `${connectionPrefix}${counter++}`;

      // 仅当名称实际变化时更新
      if (input.name !== newName) {
        input.name = newName;
        input.label = newName;
      }
    }
  });

  node._isRestoring = prevRestoring;
};

// 通用节点扩展函数
function extendNodeForDynamicInputs(nodeType, inputName = "image") {
  // 增强序列化功能
  const originalSerialize = nodeType.prototype.serialize;
  nodeType.prototype.serialize = function () {
    const data = originalSerialize?.call(this) || {};
    data._dynamicInputs = Array.from(this._dynamicInputs || []);
    return data;
  };

  // 配置恢复处理
  const originalConfigure = nodeType.prototype.onConfigure;
  nodeType.prototype.onConfigure = function (data) {
    this._isRestoring = true;
    this._dynamicInputs = new Set(data?._dynamicInputs || []);
    const result = originalConfigure?.call(this, data);

    manageDynamicInputs(this, -1, false, inputName, "IMAGE");
    this._isRestoring = false;

    return result;
  };

  // 节点创建处理
  const onNodeCreated = nodeType.prototype.onNodeCreated;
  nodeType.prototype.onNodeCreated = function () {
    const res = onNodeCreated?.apply(this, arguments);

    // 仅当没有动态输入时添加初始输入
    if (!this._dynamicInputs || this._dynamicInputs.size === 0) {
      this.addInput(`${inputName}1`, "IMAGE");
      this._dynamicInputs = new Set([`${inputName}1`]);
    }

    return res;
  };

  // 连接变化处理
  const onConnectionsChange = nodeType.prototype.onConnectionsChange;
  nodeType.prototype.onConnectionsChange = function (
    type,
    index,
    connected,
    linkInfo
  ) {
    if (!linkInfo) return;
    const res = onConnectionsChange?.apply(this, arguments);

    // 获取实际的输入类型（默认为 IMAGE）
    const inputType = this.inputs[0]?.type || "IMAGE";
    manageDynamicInputs(this, index, connected, inputName, inputType);

    return res;
  };
}

// 统一注册节点扩展
const NODES_TO_EXTEND = [
  "ImageCountConcatenate",
  "ImageWidthStitch",
  "ImageHeigthStitch",
];

app.registerExtension({
  name: "ComfyUI-MakkiTools.Wid",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (NODES_TO_EXTEND.includes(nodeData.name)) {
      extendNodeForDynamicInputs(nodeType);
    }
  },
});
