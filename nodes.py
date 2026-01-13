class GetImageNthCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image batch\n输入图像批次"}),
                "Nth_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2 * 31 - 1,
                        "step": 1,
                        "tooltip": "The index of the image to extract (starting from 1)\n要提取的图像索引（从1开始）",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "GetImageNthCount"
    CATEGORY = "MakkiTools"
    DESCRIPTION = (
        "Extract the Nth image from an image batch.\n从图像批次中提取第N张图像。"
    )

    def GetImageNthCount(self, image, Nth_count):
        return (image[Nth_count - 1 : Nth_count],)


class ImageChannelSeparate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image\n输入图像"}),
                "channel": (
                    ["red", "green", "blue", "alpha"],
                    {
                        "default": "red",
                        "tooltip": "Select which channel to extract\n选择要提取的通道",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ImageChannelSeparate"
    CATEGORY = "MakkiTools"
    DESCRIPTION = (
        "Extract a specific color channel from an image.\n从图像中提取特定颜色通道。"
    )

    def ImageChannelSeparate(self, image, channel):
        channel_map = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
        channel_index = channel_map[channel]
        num_channels = image.shape[3]
        if num_channels < 4 and channel_index == 3:
            return (image,)
        else:
            import torch

            separate = torch.zeros_like(image)
            separate[..., channel_index] = image[..., channel_index]
            if num_channels >= 4 and channel_index != 3:
                separate[..., 3] = image[..., 3]
            return (separate,)


class MergeImageChannels:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "red_channel": (
                    "IMAGE",
                    {"tooltip": "Red channel image\n红色通道图像"},
                ),
                "green_channel": (
                    "IMAGE",
                    {"tooltip": "Green channel image\n绿色通道图像"},
                ),
                "blue_channel": (
                    "IMAGE",
                    {"tooltip": "Blue channel image\n蓝色通道图像"},
                ),
                "alpha_channel": (
                    "IMAGE",
                    {"tooltip": "Alpha channel image\nAlpha通道图像"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "MergeImageChannels"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Merge separate color channels into a single image.\n将分离的颜色通道合并为单个图像。"

    def MergeImageChannels(
        self,
        red_channel=None,
        green_channel=None,
        blue_channel=None,
        alpha_channel=None,
    ):
        ref_tensor = next(
            ch
            for ch in [red_channel, green_channel, blue_channel, alpha_channel]
            if ch is not None
        )
        base_shape = ref_tensor.shape[:-1]
        device, dtype = ref_tensor.device, ref_tensor.dtype
        has_alpha = alpha_channel is not None
        num_channels = 4 if has_alpha else 3

        def _rebuild_channel(input_tensor, target_idx):
            import torch

            if input_tensor is None:
                return torch.zeros(
                    *base_shape, num_channels, device=device, dtype=dtype
                )
            source_idx = min(target_idx, input_tensor.shape[-1] - 1)
            rebuilt = torch.zeros(*base_shape, num_channels, device=device, dtype=dtype)
            rebuilt[..., target_idx] = input_tensor[..., source_idx]
            return rebuilt

        final_red = _rebuild_channel(red_channel, 0)
        final_green = _rebuild_channel(green_channel, 1)
        final_blue = _rebuild_channel(blue_channel, 2)
        final_alpha = _rebuild_channel(alpha_channel, 3) if has_alpha else None
        merged = final_red + final_green + final_blue
        if has_alpha:
            merged += final_alpha
        return (merged,)


class ImageCountConcatenate:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ImageCountConcatenate"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Concatenate multiple image batches into a single batch.\n将多个图像批次连接成一个批次。"

    def ImageCountConcatenate(self, **kwargs):
        images = list(kwargs.values())
        ref_h, ref_w, ref_c = images[0].shape[1], images[0].shape[2], images[0].shape[3]
        processed_images = []
        import torch

        for img in images:
            current_c = img.shape[3]
            if current_c != ref_c:
                if current_c == 4 and ref_c == 3:
                    rgb, alpha = img[..., :3], img[..., 3:]
                    white = torch.ones_like(rgb)
                    img = rgb * alpha + white * (1 - alpha)
                elif current_c == 3 and ref_c == 4:
                    alpha = torch.ones(
                        (*img.shape[:-1], 1), dtype=img.dtype, device=img.device
                    )
                    img = torch.cat([img, alpha], dim=-1)
                elif current_c == 1 and ref_c == 3:
                    img = img.expand(-1, -1, -1, 3)
                elif current_c == 3 and ref_c == 1:
                    img = (
                        0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                    ).unsqueeze(-1)
            current_h, current_w = img.shape[1], img.shape[2]
            if (current_h, current_w) != (ref_h, ref_w):
                import comfy.utils

                image = img.movedim(-1, 1)
                new_image = comfy.utils.common_upscale(
                    image, ref_w, ref_h, "bicubic", "center"
                )
                img = new_image.movedim(1, -1)

            processed_images.append(img)

        combined = torch.cat(processed_images, dim=0)
        current_count = combined.shape[0]
        return (combined[:current_count],)


class ImageWidthStitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ImageWidthStitch"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Stitch images horizontally (deprecated).\n水平拼接图像（已弃用）。"

    def ImageWidthStitch(self, **kwargs):
        images = list(kwargs.values())
        first_img = images[0]
        ref_H = first_img.shape[1]

        resized_images = []
        for img in images:
            W = img.shape[2]
            H = img.shape[1]
            aspect_ratio = W / H

            new_H = ref_H
            new_W = int(ref_H * aspect_ratio)

            if H != new_H or W != new_W:
                import comfy.utils

                image = img.movedim(-1, 1)
                new_image = comfy.utils.common_upscale(
                    image, new_W, new_H, "bicubic", "disabled"
                )
                img = new_image.movedim(1, -1)
            resized_images.append(img)
        import torch

        concatenated = torch.cat(resized_images, dim=2)
        return (concatenated,)


class ImageHeigthStitch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ImageHeigthStitch"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Stitch images vertically (deprecated).\n垂直拼接图像（已弃用）。"

    def ImageHeigthStitch(self, **kwargs):
        images = list(kwargs.values())
        first_img = images[0]
        ref_W = first_img.shape[2]

        resized_images = []
        for img in images:
            W = img.shape[2]
            H = img.shape[1]
            aspect_ratio = W / H

            new_H = int(ref_W / aspect_ratio)
            new_W = ref_W

            if H != new_H or W != new_W:
                import comfy.utils

                image = img.movedim(-1, 1)
                new_image = comfy.utils.common_upscale(
                    image, new_W, new_H, "bicubic", "disabled"
                )
                img = new_image.movedim(1, -1)
            resized_images.append(img)
        import torch

        concatenated = torch.cat(resized_images, dim=1)
        return (concatenated,)


class AutoLoop_create_pseudo_loop_video:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames": ("IMAGE", {"tooltip": "Input video frames\n输入视频帧"}),
                "transition_duration": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0,
                        "max": 0.5,
                        "step": 0.01,
                        "tooltip": "Transition duration as a proportion of total frames\n过渡持续时间占总帧数的比例",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "AutoLoop_create_pseudo_loop_video"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Create a pseudo loop video (deprecated).\n创建伪循环视频（已弃用）。"

    def ease_in_out(self, t):
        return t * t * (3 - 2 * t)

    def AutoLoop_create_pseudo_loop_video(self, frames, transition_duration):
        B, H, W, C = frames.shape
        assert B >= 4, "required 4+ frames."

        L = max(1, min(B // 2, int(B * transition_duration)))

        best_score = -float("inf")
        best_idx = B // 2

        search_start = max(L, B // 4)
        search_end = min(B - L, 3 * B // 4)

        for idx in range(search_start, search_end):
            front_end = frames[idx - 1 : idx + 1].flatten(1)
            back_start = frames[idx : idx + 2].flatten(1)

            import torch.nn.functional as F

            score = F.cosine_similarity(front_end, back_start).mean()

            if score > best_score:
                best_score = score
                best_idx = idx

        split_index = best_idx

        front = frames[:split_index]
        back = frames[split_index:]

        back_no_tail = back[:-L] if L < len(back) else back[0:0]
        front_no_head = front[L:] if L < len(front) else front[0:0]

        import torch

        alphas = self.ease_in_out(torch.linspace(0, 1, L))
        alphas = alphas.view(L, 1, 1, 1)

        transition = (1 - alphas) * back[-L:] + alphas * front[:L]

        loop = torch.cat([back_no_tail, transition, front_no_head], dim=0)

        return (loop,)


class Environment_INFO:
    from .environment_info import AlwaysEqualProxy

    any_type = AlwaysEqualProxy("*")

    def __init__(self):
        from .environment_info import format_environment_info, get_environment_info

        self.format_environment_info = format_environment_info
        self.get_environment_info = get_environment_info

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "SYSTEM_INFO": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Display system information\n显示系统信息",
                    },
                ),
                "HARDWARE_INFO": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Display hardware information\n显示硬件信息",
                    },
                ),
                "GPU_INFO": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Display GPU information\n显示GPU信息",
                    },
                ),
                "DEEP_LEARNING_FRAMEWORKS_INFO": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Display deep learning framework information\n显示深度学习框架信息",
                    },
                ),
                "ALL_INSTALLED_PACKAGES_INFO": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Display all installed packages information\n显示所有已安装包的信息",
                    },
                ),
                "CUSTOM_NODES_FOLDERS_INFO": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Display custom nodes folders information\n显示自定义节点文件夹信息",
                    },
                ),
            },
            "optional": {
                "anything": (
                    s.any_type,
                    {
                        "tooltip": "Any input (not used, just for chaining)\n任意输入（不使用，仅用于链式连接）"
                    },
                )
            },
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("INFO", "anything")
    OUTPUT_NODE = True
    FUNCTION = "Environment_INFO"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Display system environment information.\n显示系统环境信息。"

    def Environment_INFO(
        self,
        SYSTEM_INFO,
        HARDWARE_INFO,
        GPU_INFO,
        DEEP_LEARNING_FRAMEWORKS_INFO,
        ALL_INSTALLED_PACKAGES_INFO,
        CUSTOM_NODES_FOLDERS_INFO,
        anything=None,
    ):
        env_info = self.get_environment_info()
        full_report = self.format_environment_info(
            env_info,
            SYSTEM_INFO,
            HARDWARE_INFO,
            GPU_INFO,
            DEEP_LEARNING_FRAMEWORKS_INFO,
            ALL_INSTALLED_PACKAGES_INFO,
            CUSTOM_NODES_FOLDERS_INFO,
        )

        return {"ui": {"info": (full_report,)}, "result": (full_report, anything)}


class translators:
    def __init__(self):
        import os

        os.environ["translators_default_region"] = "EN"
        import translators as ts
        from .translators_map import LANGUAGE_MAPPING

        self._pre_acceleration_done = False
        self.ts = ts
        self.LANGUAGE_MAPPING = LANGUAGE_MAPPING

    @classmethod
    def INPUT_TYPES(s):
        import os

        os.environ["translators_default_region"] = "EN"
        import translators as ts
        from .translators_map import Supported_Languages

        return {
            "required": {
                "query_text": (
                    "STRING",
                    {"multiline": True, "tooltip": "Text to translate\n要翻译的文本"},
                ),
                "translator": (
                    list(ts.translators_pool),
                    {
                        "default": list(ts.translators_pool)[0],
                        "tooltip": "Translation service to use\n要使用的翻译服务",
                    },
                ),
                "from_language": (
                    ["auto"] + Supported_Languages,
                    {
                        "default": "auto",
                        "tooltip": "Source language (auto for automatic detection)\n源语言（auto为自动检测）",
                    },
                ),
                "to_language": (
                    Supported_Languages,
                    {
                        "default": Supported_Languages[0],
                        "tooltip": "Target language\n目标语言",
                    },
                ),
                "if_use_preacceleration": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable pre-acceleration for faster translation\n启用预加速以提高翻译速度",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translators"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Translate text using various online translation services.\n使用各种在线翻译服务翻译文本。"

    def normalize_language_code(self, translator, Language):
        return self.LANGUAGE_MAPPING.get(translator, {}).get(Language, Language)

    def translators(
        self,
        query_text,
        translator,
        from_language,
        to_language,
        if_use_preacceleration,
    ):
        pattern = r"\(([^()]+)\)[^()]*$"
        import re

        if from_language != "auto":
            match = re.search(pattern, from_language)
            from_language = match.group(1)
            from_language = self.normalize_language_code(translator, from_language)

        match = re.search(pattern, to_language)
        to_language = match.group(1)
        to_language = self.normalize_language_code(translator, to_language)

        if if_use_preacceleration and not self._pre_acceleration_done:
            _ = self.ts.preaccelerate_and_speedtest()
            self._pre_acceleration_done = True

        try:
            import threading
            import queue

            result_queue = queue.Queue()

            def translate_worker():
                try:
                    output = self.ts.translate_text(
                        query_text,
                        translator=translator,
                        from_language=from_language,
                        to_language=to_language,
                        if_use_preacceleration=if_use_preacceleration,
                        if_ignore_limit_of_length=True,
                    )
                    result_queue.put(("success", output))
                except Exception as e:
                    result_queue.put(("error", e))

            thread = threading.Thread(target=translate_worker)
            thread.start()

            try:
                result_type, result = result_queue.get(timeout=10)
                if result_type == "error":
                    raise result
                output = result
            except queue.Empty:
                print(
                    "[Translators Node] Translation timed out, returning original text"
                )
                return (query_text,)

        except Exception as e:
            error_str = str(e)
            print(f"[Translators Node] Chunk translation failed: {error_str}")

            if "Unsupported" in error_str or "same" in error_str:
                print(
                    f"[Translators Node] Fallback to original text for chunk.Please check the settings of 'from_language' and 'to_language'"
                )

            return (query_text,)

        return (output,)


class translator_m2m100:
    def __init__(self):
        from .m2m100 import M2M100Translator

        self.M2M100Translator = M2M100Translator

    @classmethod
    def INPUT_TYPES(s):
        from .m2m100 import m2m100map

        return {
            "required": {
                "query_text": (
                    "STRING",
                    {"multiline": True, "tooltip": "Text to translate\n要翻译的文本"},
                ),
                "model": (
                    [
                        "facebook/m2m100_418M",
                        "facebook/m2m100_1.2B",
                        "facebook/m2m100-12B-avg-5-ckpt",
                        "facebook/m2m100-12B-avg-10-ckpt",
                        "facebook/m2m100-12B-last-ckpt",
                    ],
                    {
                        "default": "facebook/m2m100_418M",
                        "tooltip": "M2M100 model to use\n要使用的M2M100模型",
                    },
                ),
                "from_language": (
                    ["auto"] + m2m100map,
                    {
                        "default": "auto",
                        "tooltip": "Source language (auto for automatic detection)\n源语言（auto为自动检测）",
                    },
                ),
                "to_language": (
                    m2m100map,
                    {"default": "English (en)", "tooltip": "Target language\n目标语言"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {
                        "default": "8bit",
                        "tooltip": "Model quantization level\n模型量化级别",
                    },
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {
                        "default": "sdpa",
                        "tooltip": "Attention implementation to use\n要使用的注意力实现",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translator_m2m100"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Translate text using the M2M100 model.\n使用M2M100模型翻译文本。"

    def translator_m2m100(
        self,
        query_text,
        model,
        from_language,
        to_language,
        quantization,
        attention,
    ):
        pattern = r"\(([^()]+)\)[^()]*$"
        import re

        if from_language != "auto":
            match = re.search(pattern, from_language)
            from_language = match.group(1)

        match = re.search(pattern, to_language)
        to_language = match.group(1)

        translator = self.M2M100Translator(
            model_repo=model, quantization=quantization, attention=attention
        )
        output = translator.translate_preserve_format(
            query_text, from_language, to_language
        )

        return (output,)


class random_any:
    from .environment_info import AlwaysEqualProxy

    any_type = AlwaysEqualProxy("*")

    def __init__(self):
        import random

        self.random = random

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "anything": (
                    s.any_type,
                    {
                        "tooltip": "Any input (passed through unchanged)\n任意输入（原样传递）"
                    },
                )
            }
        }

    RETURN_TYPES = (any_type, "INT", "FLOAT")
    RETURN_NAMES = ("any", "int", "float")
    FUNCTION = "random_any"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Generate random integer and float values.\n生成随机整数和浮点数值。"

    def random_any(self, anything=None):
        return (
            anything,
            self.random.Random().randint(0, 0xFFFFFFFFFFFFFFFF),
            self.random.Random().random(),
        )


class AnyImageStitch:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dimension": (
                    ["horizontal", "vertical"],
                    {
                        "default": "horizontal",
                        "tooltip": "Direction to stitch images\n拼接图像的方向",
                    },
                ),
                "reference_type": (
                    ["first image", "custom"],
                    {
                        "default": "first image",
                        "tooltip": "Reference size type\n参考尺寸类型",
                    },
                ),
                "reference_value": (
                    "INT",
                    {
                        "default": 512,
                        "min": 1,
                        "max": 4096,
                        "step": 1,
                        "tooltip": "Custom reference size\n自定义参考尺寸",
                    },
                ),
                "upscale_method": (
                    s.upscale_methods,
                    {
                        "default": "bicubic",
                        "tooltip": "Method to upscale images\n图像放大方法",
                    },
                ),
                "crop": (
                    s.crop_methods,
                    {
                        "default": "disabled",
                        "tooltip": "Crop method for upscaled images\n放大图像的裁剪方法",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "AnyImageStitch"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Stitch any number of images in horizontal or vertical direction.\n在水平或垂直方向上拼接任意数量的图像。"

    def AnyImageStitch(
        self, dimension, reference_type, reference_value, upscale_method, crop, **kwargs
    ):
        import torch

        # 获取图像列表并过滤非图像输入
        images = [
            img
            for img in kwargs.values()
            if isinstance(img, torch.Tensor) and img.dim() == 4
        ]

        # 确定参考尺寸
        if reference_type == "first image":
            ref_dim = (
                images[0].shape[1] if dimension == "horizontal" else images[0].shape[2]
            )
        else:
            ref_dim = reference_value

        resized_images = []
        for img in images:
            batch, H, W, channels = img.shape

            # 计算目标尺寸
            if dimension == "horizontal":
                target_H = ref_dim
                aspect_ratio = W / H
                target_W = int(target_H * aspect_ratio)
            else:  # vertical
                target_W = ref_dim
                aspect_ratio = H / W
                target_H = int(target_W * aspect_ratio)

            # 调整图像尺寸（如果需要）
            if H != target_H or W != target_W:
                import comfy.utils

                image = img.movedim(-1, 1)  # [batch, channels, H, W]
                new_image = comfy.utils.common_upscale(
                    image, target_W, target_H, upscale_method, crop
                )
                img = new_image.movedim(1, -1)  # 恢复原始维度

            resized_images.append(img)

        # 确定拼接维度
        dim = 2 if dimension == "horizontal" else 1

        # 拼接所有图像
        concatenated = torch.cat(resized_images, dim=dim)
        return (concatenated,)


class AnyImagetoConditioning_flux_kontext:
    PREFERED_KONTEXT_RESOLUTIONS = [
        (672, 1568),
        (688, 1504),
        (720, 1456),
        (752, 1392),
        (800, 1328),
        (832, 1248),
        (880, 1184),
        (944, 1104),
        (1024, 1024),
        (1104, 944),
        (1184, 880),
        (1248, 832),
        (1328, 800),
        (1392, 752),
        (1456, 720),
        (1504, 688),
        (1568, 672),
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": (
                    "CONDITIONING",
                    {"tooltip": "Input conditioning\n输入条件"},
                ),
                "vae": ("VAE", {"tooltip": "VAE model\nVAE模型"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "AnyImagetoConditioning_flux_kontext"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Convert any number of images to conditioning for flux kontext.\n将任意数量的图像转换为flux kontext的条件。"

    def AnyImagetoConditioning_flux_kontext(self, conditioning, vae, **kwargs):
        for img in kwargs.values():
            pixels = self.scale(img)
            t = vae.encode(pixels[:, :, :, :3])
            latent = {"samples": t}
            import node_helpers

            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": [latent["samples"]]}, append=True
            )

        return (conditioning,)

    def scale(self, image):
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min(
            (abs(aspect_ratio - w / h), w, h)
            for w, h in self.PREFERED_KONTEXT_RESOLUTIONS
        )
        import comfy.utils

        image = comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "lanczos", "center"
        ).movedim(1, -1)
        return image


class show_type:
    from .environment_info import AlwaysEqualProxy

    any_type = AlwaysEqualProxy("*")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "anything": (
                    s.any_type,
                    {
                        "tooltip": "Input of any type to show its type\n任意类型的输入以显示其类型"
                    },
                )
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "show_type"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Show the type of input data.\n显示输入数据的类型。"

    def show_type(self, anything):
        type_name = type(anything).__name__.lower()
        return {"ui": {"info": (type_name,)}, "result": (type_name,)}


class timer:
    from .environment_info import AlwaysEqualProxy

    any_type = AlwaysEqualProxy("*")

    def __init__(self):
        import time

        self.time = time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed = 0
            self.elapsed_str = "0ms"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["start", "stop"], {"tooltip": "Timer mode\n计时器模式"}),
                "format": (
                    ["ms", "s", "m", "auto"],
                    {"default": "auto", "tooltip": "Time display format\n时间显示格式"},
                ),
            },
            "optional": {
                "any": (
                    s.any_type,
                    {"tooltip": "Any input to pass through\n任意输入通过"},
                ),
                "timer": ("TIMER_MAKKI", {"tooltip": "Timer object\n计时器对象"}),
            },
        }

    RETURN_TYPES = (any_type, "TIMER_MAKKI", "INT", "STRING")
    RETURN_NAMES = ("any", "timer", "time_ms", "time_str")
    OUTPUT_NODE = True
    FUNCTION = "timer"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Measure execution time between nodes.\n测量节点间的执行时间。"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def format_time(self, ms, format_type):
        if format_type == "ms" or (format_type == "auto" and ms < 1000):
            return f"{ms}ms"

        seconds = ms / 1000.0
        if format_type == "s" or (format_type == "auto" and seconds < 60):
            return f"{seconds:.3f}s"

        minutes = int(seconds // 60)
        remaining_sec = seconds % 60
        return f"{minutes}m {remaining_sec:.1f}s"

    def timer(self, mode, format, any=None, timer=None):
        if mode == "start":
            if timer is None or timer.start_time is not None:
                timer = self.Timer()
            timer.start_time = self.time.time()
            timer.elapsed = 0
            timer.elapsed_str = "0ms"
            return (any, timer, 0, "0ms")

        elif mode == "stop":
            if timer is None:
                raise ValueError("Stop mode requires a valid timer input")
            if timer.start_time is None:
                return (any, timer, timer.elapsed)

            end_time = self.time.time()
            elapsed_ms = int((end_time - timer.start_time) * 1000)
            timer.elapsed = elapsed_ms
            timer.elapsed_str = self.format_time(elapsed_ms, format)
            timer.start_time = None
            return {
                "ui": {"info": (timer.elapsed_str,)},
                "result": (any, timer, timer.elapsed, timer.elapsed_str),
            }


class Image_Resize:
    upscale_methods = [
        "auto",
        "nearest-exact",
        "bilinear",
        "area",
        "bicubic",
        "lanczos",
    ]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_method": (
                    s.upscale_methods,
                    {
                        "default": "auto",
                        "tooltip": "Method to upscale the image\n图像放大方法",
                    },
                ),
                "crop_method": (
                    s.crop_methods,
                    {
                        "default": "center",
                        "tooltip": "Method to crop the image\n图像裁剪方法",
                    },
                ),
                "size": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "tooltip": "Target size (based on selected dimension)\n目标尺寸（基于选定的维度）",
                    },
                ),
                "GCD": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Greatest Common Divisor constraint\n最大公约数约束",
                    },
                ),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Input image\n输入图像"}),
                "width": (
                    "INT",
                    {
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "defaultInput": True,
                        "tooltip": "Target width\n目标宽度",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "defaultInput": True,
                        "tooltip": "Target height\n目标高度",
                    },
                ),
                "short_side": (
                    "INT",
                    {
                        "min": 1,
                        "max": 8192,
                        "step": 1,
                        "defaultInput": True,
                        "tooltip": "Short side length\n短边长度",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "Image_Resize"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Resize images with various methods and constraints.\n使用多种方法和约束调整图像大小。"

    def Image_Resize(
        self,
        upscale_method,
        crop_method,
        size,
        GCD,
        image=None,
        width=None,
        height=None,
        short_side=None,
    ):
        if image is not None:
            original_height, original_width = image.shape[1:3]
        else:
            original_width = original_height = size

        original_pixels = original_width * original_height
        target_pixels = size**2

        aspect_ratio = original_width / original_height

        if short_side is not None and width is None and height is None:
            if original_width <= original_height:
                new_width = short_side
                new_height = short_side / aspect_ratio
                new_height = max(GCD, round(new_height / GCD) * GCD)
            else:
                new_height = short_side
                new_width = short_side * aspect_ratio
                new_width = max(GCD, round(new_width / GCD) * GCD)
            new_width = int(new_width)
            new_height = int(new_height)

        elif width is not None and height is not None:
            new_width, new_height = width, height
        elif width is not None:
            new_width = width
            new_height = int(new_width / aspect_ratio)
        elif height is not None:
            new_height = height
            new_width = int(new_height * aspect_ratio)
        else:
            new_height = int((target_pixels / aspect_ratio) ** 0.5)
            new_width = int(new_height * aspect_ratio)

        if short_side is None or width is not None or height is not None:
            new_width = max(GCD, round(new_width / GCD) * GCD)
            new_height = max(GCD, round(new_height / GCD) * GCD)

        if image is not None:
            import comfy.utils

            if upscale_method == "auto":
                if original_pixels > target_pixels:
                    upscale_method = "area"
                elif original_pixels < target_pixels:
                    upscale_method = "lanczos"
                else:
                    upscale_method = "nearest-exact"

            image = image.movedim(-1, 1)
            new_image = comfy.utils.common_upscale(
                image, new_width, new_height, upscale_method, crop_method
            )
            new_image = new_image.movedim(1, -1)
        else:
            import torch

            new_image = torch.zeros((1, new_height, new_width, 3), dtype=torch.float32)

        return (new_image, new_width, new_height)


class Prism_Mirage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "exterior_image": ("IMAGE", {"tooltip": "Exterior image\n外部图像"}),
                "interior_image": ("IMAGE", {"tooltip": "Interior image\n内部图像"}),
                "exterior_level": (
                    "INT",
                    {
                        "default": 42,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Exterior transparency level\n外部透明度级别",
                    },
                ),
                "interior_level": (
                    "INT",
                    {
                        "default": 24,
                        "min": 0,
                        "max": 255,
                        "step": 1,
                        "tooltip": "Interior transparency level\n内部透明度级别",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "Prism_Mirage"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Apply a prism mirage effect by combining two images.\n通过合并两张图像应用棱镜幻影效果。"

    def Prism_Mirage(
        self, exterior_image, interior_image, exterior_level, interior_level
    ):
        if exterior_level < interior_level:
            interior_level == exterior_level
        from .Prism_Mirage import process_images

        image = process_images(
            exterior_image, interior_image, exterior_level, interior_level
        )

        return (image,)


class int_calculate_statistics:
    from .environment_info import AlwaysEqualProxy

    any_type = AlwaysEqualProxy("*")

    def __init__(self):
        import math
        from collections import Counter

        self.math = math
        self.Counter = Counter

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "stat_mode": (
                    [
                        "count",
                        "sum",
                        "mean",
                        "min",
                        "max",
                        "range",
                        "median",
                        "quartiles",
                        "iqr",
                        "mode",
                        "variance",
                        "std_dev",
                        "mad",
                        "skewness",
                        "kurtosis",
                        "frequency",
                        "sorted_list",
                        "unique_values",
                        "cumulative_sum",
                        "proportions",
                        "z_scores",
                    ],
                    {
                        "default": "count",
                        "tooltip": "Statistical measure to calculate\n要计算的统计量",
                    },
                ),
            }
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "int_calculate_statistics"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Calculate various statistical measures from integer inputs.\n从整数输入计算各种统计量。"

    def int_calculate_statistics(self, stat_mode, **kwargs):
        return (self.calculate_statistics(stat_mode, **kwargs),)

    def calculate_statistics(self, stat_mode, **kwargs):
        """
        计算并返回指定的整数统计结果

        参数:
            stat_mode: 要计算的统计量名称
            **kwargs: 任意数量的整数关键字参数

        返回:
            指定的统计量结果
        """
        # 提取所有整数值
        ints = list(kwargs.values())
        n = len(ints)

        # 空输入处理
        if n == 0:
            if stat_mode == "count":
                return 0
            return None

        # 计算基本统计量
        def calculate_basic():
            total = sum(ints)
            mean = total / n
            sorted_ints = sorted(ints)
            min_val = sorted_ints[0]
            max_val = sorted_ints[-1]

            return {
                "count": n,
                "sum": total,
                "mean": mean,
                "min": min_val,
                "max": max_val,
                "sorted_list": sorted_ints,
            }

        # 计算中位数和四分位数
        def calculate_quartiles(sorted_ints):
            def get_quartile_index(p: float) -> float:
                pos = (n - 1) * p
                i = int(pos)
                frac = pos - i
                return (
                    sorted_ints[i] + frac * (sorted_ints[i + 1] - sorted_ints[i])
                    if i < n - 1
                    else sorted_ints[i]
                )

            q1 = get_quartile_index(0.25) if n > 1 else sorted_ints[0]
            median = get_quartile_index(0.5) if n > 1 else sorted_ints[0]
            q3 = get_quartile_index(0.75) if n > 1 else sorted_ints[0]
            return q1, median, q3

        # 计算离散度指标
        def calculate_dispersion(mean: float):
            variance = sum((x - mean) ** 2 for x in ints) / n
            std_dev = self.math.sqrt(variance)
            mad = sum(abs(x - mean) for x in ints) / n
            return {"variance": variance, "std_dev": std_dev, "mad": mad}

        # 计算分布特征
        def calculate_distribution(mean: float, std_dev: float):
            if std_dev == 0:
                return {"skewness": 0, "kurtosis": 0}

            skewness = (sum((x - mean) ** 3 for x in ints) / n) / (std_dev**3)
            kurtosis = (sum((x - mean) ** 4 for x in ints) / n) / (std_dev**4)
            return {"skewness": skewness, "kurtosis": kurtosis - 3}  # 返回超额峰度

        # 根据请求的统计量计算并返回结果
        basic = calculate_basic()
        sorted_ints = basic["sorted_list"]

        if stat_mode == "count":
            return basic["count"]

        if stat_mode == "sum":
            return basic["sum"]

        if stat_mode == "mean":
            return basic["mean"]

        if stat_mode == "min":
            return basic["min"]

        if stat_mode == "max":
            return basic["max"]

        if stat_mode == "range":
            return basic["max"] - basic["min"]

        if stat_mode == "median":
            return calculate_quartiles(sorted_ints)[1]

        if stat_mode == "quartiles":
            return calculate_quartiles(sorted_ints)

        if stat_mode == "iqr":
            q1, _, q3 = calculate_quartiles(sorted_ints)
            return q3 - q1

        if stat_mode == "mode":
            count_dict = self.Counter(ints)
            max_count = max(count_dict.values())
            return [k for k, v in count_dict.items() if v == max_count]

        if stat_mode == "variance":
            return calculate_dispersion(basic["mean"])["variance"]

        if stat_mode == "std_dev":
            return calculate_dispersion(basic["mean"])["std_dev"]

        if stat_mode == "mad":
            return calculate_dispersion(basic["mean"])["mad"]

        if stat_mode == "skewness":
            dispersion = calculate_dispersion(basic["mean"])
            return calculate_distribution(basic["mean"], dispersion["std_dev"])[
                "skewness"
            ]

        if stat_mode == "kurtosis":
            dispersion = calculate_dispersion(basic["mean"])
            return calculate_distribution(basic["mean"], dispersion["std_dev"])[
                "kurtosis"
            ]

        if stat_mode == "frequency":
            return dict(self.Counter(ints))

        if stat_mode == "sorted_list":
            return sorted_ints

        if stat_mode == "unique_values":
            return sorted(set(ints))

        if stat_mode == "cumulative_sum":
            return [sum(ints[: i + 1]) for i in range(n)]

        if stat_mode == "proportions":
            total = basic["sum"]
            return [x / total for x in ints]

        if stat_mode == "z_scores":
            dispersion = calculate_dispersion(basic["mean"])
            std_dev = dispersion["std_dev"]
            if std_dev == 0:
                return [0] * n
            return [(x - basic["mean"]) / std_dev for x in ints]

        raise ValueError(f"未知的统计量名称: {stat_mode}")


class BatchLoraLoader:
    import folder_paths
    import comfy.utils
    import comfy.sd

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        loras = ["None"] + s.folder_paths.get_filename_list("loras")
        inputs = {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "The diffusion model the LoRA will be applied to.\nLoRA将应用到的扩散模型。"
                    },
                ),
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "The CLIP model the LoRA will be applied to.\nLoRA将应用到的CLIP模型。"
                    },
                ),
                "loras_count": (
                    "INT",
                    {"default": 3, "min": 1, "max": 50, "step": 1},
                ),
                "loras_num": (
                    "INT",
                    {"default": -1, "min": -1, "max": 50, "step": 1},
                ),
            }
        }

        for i in range(1, 50):
            inputs["required"][f"lora_name_{i}"] = (
                loras,
                {"tooltip": f"LoRA model {i}\nLoRA模型 {i}"},
            )
            inputs["required"][f"strength_model_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How strongly to modify the diffusion model. This value can be negative.\n修改扩散模型的强度。此值可以为负。",
                },
            )
            inputs["required"][f"strength_clip_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How strongly to modify the CLIP model. This value can be negative.\n修改CLIP模型的强度。此值可以为负。",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "BatchLoraLoader"

    CATEGORY = "MakkiTools"
    DESCRIPTION = "Load and apply multiple LoRA models to diffusion and CLIP models.\n加载并将多个LoRA模型应用于扩散和CLIP模型。"

    def BatchLoraLoader(self, model, clip, loras_count, loras_num, **kwargs):
        if loras_num == 0:  # 为0时全部跳过
            return (model, clip)

        if loras_num > loras_count:  # 越界时加载全部LoRA
            loras_num = -1

        # 提取所有LoRA参数
        selected_lora_name = [
            kwargs[f"lora_name_{i}"] for i in range(1, loras_count + 1)
        ]
        selected_strength_model = [
            kwargs[f"strength_model_{i}"] for i in range(1, loras_count + 1)
        ]
        selected_strength_clip = [
            kwargs[f"strength_clip_{i}"] for i in range(1, loras_count + 1)
        ]

        # 确定要加载的LoRA索引
        if loras_num == -1:  # 加载全部
            lora_indices = range(loras_count)
        else:  # 加载指定索引的LoRA
            lora_indices = [loras_num - 1]  # 转换为0-based索引

        current_model, current_clip = model, clip

        # 遍历所有需要加载的LoRA
        for i in lora_indices:
            lora_name = selected_lora_name[i]
            strength_model = selected_strength_model[i]
            strength_clip = selected_strength_clip[i]

            # 跳过None
            if lora_name == "None":
                continue

            # 跳过强度为0的LoRA
            if strength_model == 0 and strength_clip == 0:
                continue

            # 获取LoRA路径
            lora_path = self.folder_paths.get_full_path_or_raise("loras", lora_name)
            lora = None

            # 检查缓存
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_path:
                    lora = self.loaded_lora[1]
                else:
                    self.loaded_lora = None  # 缓存不匹配时清除

            # 需要时加载LoRA
            if lora is None:
                lora = self.comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)  # 更新缓存

            # 应用LoRA到当前模型
            current_model, current_clip = self.comfy.sd.load_lora_for_models(
                current_model, current_clip, lora, strength_model, strength_clip
            )

        return (current_model, current_clip)


class UniversalInstaller:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "package_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The name of the package to install, such as 'requests' or 'tipo-kgen==0.2.0'.\n要安装的包名称，例如'requests'或'tipo-kgen==0.2.0'。",
                    },
                ),
                "backend": (
                    ["pip", "uv"],
                    {
                        "default": "pip",
                        "tooltip": "Package manager to use\n要使用的包管理器",
                    },
                ),
                "mirror_source": (
                    ["default", "tsinghua", "aliyun", "douban", "ustc", "custom"],
                    {
                        "default": "default",
                        "tooltip": "Select the mirror source, use the content in extra_index_url when custom.\n选择镜像源，自定义时使用extra_index_url中的内容。",
                    },
                ),
                "update_if_no_version": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When the package name does not specify a version, should the -U parameter be used to update to the latest version.\n当包名称未指定版本时，是否应使用-U参数更新到最新版本。",
                    },
                ),
            },
            "optional": {
                "extra_index_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Additional index URLs (e.g., private sources), optional. Only used when 'mirror_source' is set to 'custom'.\n附加索引URL（例如私有源），可选。仅在'mirror_source'设置为'custom'时使用。",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "UniversalInstaller"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Install Python packages dynamically.\n动态安装Python包。"

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return float("nan")

    def UniversalInstaller(
        self,
        package_name,
        backend,
        mirror_source,
        update_if_no_version,
        extra_index_url="",
    ):
        """
        安装指定的Python库
        """
        import subprocess
        import sys

        if not package_name.strip():
            return (
                "❌ 错误：请输入要安装的库名称。\n❌ Error: Please enter the name of the package to install.",
            )

        install_cmd = []

        if backend == "uv":
            install_cmd = [sys.executable, "-m", "uv", "pip", "install"]
        else:  # pip
            install_cmd = [sys.executable, "-m", "pip", "install"]

        mirror_urls = {
            "tsinghua": "https://pypi.tuna.tsinghua.edu.cn/simple",
            "aliyun": "https://mirrors.aliyun.com/pypi/simple/",
            "douban": "https://pypi.douban.com/simple/",
            "ustc": "https://pypi.mirrors.ustc.edu.cn/simple/",
        }

        if mirror_source != "default":
            if mirror_source == "custom" and extra_index_url:
                install_cmd.extend(["--extra-index-url", extra_index_url.strip()])
            elif mirror_source in mirror_urls:
                install_cmd.extend(["-i", mirror_urls[mirror_source]])

        if (
            update_if_no_version
            and "==" not in package_name
            and ">" not in package_name
            and "<" not in package_name
        ):
            install_cmd.append("-U")

        install_cmd.append(package_name.strip())

        try:
            print(
                f"开始安装库: \nStart installing package: \n{package_name}\n使用后端: \nUsing backend: \n{backend}\n"
            )
            print(f"{' '.join(install_cmd)}")

            process = subprocess.Popen(
                install_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            output_log = []

            for line in iter(process.stdout.readline, ""):
                print(line, end="")
                output_log.append(line)
            process.wait()

            if process.returncode == 0:
                status_message = f"✅ 成功安装Successfully installed: {package_name}\n\n请完全重启 ComfyUI 以使更改生效。Please fully restart ComfyUI to apply the changes.\n\n--- 安装日志Installation Log ---\n{''.join(output_log)}"
            else:
                full_log = "".join(output_log)
                status_message = f"❌ 安装失败 (Installation failed)\n返回码 (Return): {process.returncode}:\n{full_log}"

        except subprocess.CalledProcessError as e:
            status_message = f"❌ 安装过程出错 (CalledProcessError):\n{str(e)}"
        except Exception as e:
            status_message = f"❌ 发生意外错误 (Exception):\n{str(e)}"

        return {"ui": {"info": (status_message,)}, "result": (status_message,)}


class get_folder_info:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "要扫描的文件夹路径，例如 'C:/Users/MyUser/Documents' 或 '/home/user/documents'\nThe folder path to scan, e.g. 'C:/Users/MyUser/Documents' or '/home/user/documents'",
                    },
                ),
                "file_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "要获取的第几个文件名（按名称排序），从1开始计数\nThe index of the file to get (sorted by name), counting from 1",
                    },
                ),
                "output_full_path": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "如果为True，输出完整路径；否则只输出文件名\nIf True, output full path; otherwise output only filename",
                    },
                ),
                "include_subfolders": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否包含子文件夹中的文件\nWhether to include files in subfolders",
                    },
                ),
                "file_filter": (
                    "STRING",
                    {
                        "default": "*",
                        "tooltip": "文件筛选器，例如 '*.txt' 或 'image*.png'\nFile filter, e.g. '*.txt' or 'image*.png'",
                    },
                ),
                "file_types": (
                    ["all", "image", "text", "audio", "video"],
                    {
                        "default": "all",
                        "tooltip": "文件类型筛选\nFile type filter",
                    },
                ),
                "sort_by": (
                    ["name", "date_modified", "size"],
                    {
                        "default": "name",
                        "tooltip": "排序方式\nSort method",
                    },
                ),
                "sort_order": (
                    ["ascending", "descending"],
                    {
                        "default": "ascending",
                        "tooltip": "排序顺序\nSort order",
                    },
                ),
                "limit": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "tooltip": "限制返回的文件数量，0表示无限制\nLimit the number of files returned, 0 means no limit",
                    },
                ),
                "regex_filter": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "正则表达式筛选，例如 '^image.*\\.png$'\nRegular expression filter, e.g. '^image.*\\.png$'",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "LIST", "STRING", "STRING", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = (
        "output_string",
        "output_list",
        "nth_item",
        "tree_structure",
        "file_count",
        "has_files",
        "error_message",
    )
    FUNCTION = "get_folder_info"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Get information about files in a folder.\n获取文件夹中文件的信息。"

    def get_folder_info(
        self,
        folder_path,
        file_index,
        output_full_path=False,
        include_subfolders=False,
        file_filter="*",
        file_types="all",
        sort_by="name",
        sort_order="ascending",
        limit=0,
        regex_filter="",
    ):
        import os

        error_msg = ""
        files = []

        try:
            # 检查文件夹是否存在
            if not os.path.exists(folder_path):
                error_msg = f"The folder does not exist: {folder_path}"
                return self.return_error(error_msg)

            if not os.path.isdir(folder_path):
                error_msg = f"The path is not a folder: {folder_path}"
                return self.return_error(error_msg)

            # 收集文件
            if include_subfolders:
                all_items = []
                for root, dirs, walk_files in os.walk(folder_path):
                    for file in walk_files:
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, folder_path)

                        # 应用所有筛选条件
                        if self.apply_filters(
                            file, full_path, file_filter, file_types, regex_filter
                        ):
                            all_items.append(
                                (
                                    rel_path,
                                    full_path,
                                    os.path.getmtime(full_path),
                                    os.path.getsize(full_path),
                                )
                            )
            else:
                all_items = []
                for item in os.listdir(folder_path):
                    full_path = os.path.join(folder_path, item)
                    if os.path.isfile(full_path):
                        # 应用所有筛选条件
                        if self.apply_filters(
                            item, full_path, file_filter, file_types, regex_filter
                        ):
                            all_items.append(
                                (
                                    item,
                                    full_path,
                                    os.path.getmtime(full_path),
                                    os.path.getsize(full_path),
                                )
                            )

            # 排序
            if sort_by == "name":
                index = 0
            elif sort_by == "date_modified":
                index = 2
            elif sort_by == "size":
                index = 3

            all_items.sort(key=lambda x: x[index], reverse=(sort_order == "descending"))

            # 限制数量
            if limit > 0:
                all_items = all_items[:limit]

            # 提取文件名和路径
            files = [item[0] for item in all_items]  # 相对路径或文件名
            full_paths = [item[1] for item in all_items]  # 完整路径

            # 根据 output_full_path 参数选择输出格式
            if output_full_path:
                output_items = full_paths
            else:
                output_items = files

            # 输出第n个文件
            nth_item = (
                output_items[file_index - 1]
                if 0 < file_index <= len(output_items)
                else "Index out of range."
            )

            # 生成表状文件夹结构字符串
            tree_structure = self.generate_tree_structure(
                folder_path, files, include_subfolders
            )

            # 返回所有信息
            return (
                "\n".join(output_items),  # 所有文件（字符串，每行一个）
                output_items,  # 所有文件（列表）
                nth_item,  # 第n个文件
                tree_structure,  # 表状文件夹结构
                len(output_items),  # 文件总数
                len(output_items) > 0,  # 是否有文件
                "",  # 错误信息（空表示无错误）
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return self.return_error(error_msg)

    def return_error(self, error_msg):
        """返回错误信息"""
        return (
            "",  # output_string
            [],  # output_list
            "",  # nth_item
            "",  # tree_structure
            0,  # file_count
            False,  # has_files
            error_msg,  # error_message
        )

    def apply_filters(self, filename, full_path, file_filter, file_types, regex_filter):
        """应用所有筛选条件"""
        import os
        import re
        import fnmatch

        # 通配符筛选
        if not fnmatch.fnmatch(filename, file_filter):
            return False

        # 正则表达式筛选
        if regex_filter and not re.search(regex_filter, filename):
            return False

        # 文件类型筛选
        if file_types != "all":
            ext = os.path.splitext(filename)[1].lower()
            if file_types == "image" and ext not in [
                ".jpg",
                ".jpeg",
                ".png",
                ".gif",
                ".bmp",
                ".tiff",
                ".webp",
            ]:
                return False
            elif file_types == "text" and ext not in [
                ".txt",
                ".csv",
                ".json",
                ".xml",
                ".html",
                ".htm",
                ".md",
            ]:
                return False
            elif file_types == "audio" and ext not in [
                ".mp3",
                ".wav",
                ".ogg",
                ".flac",
                ".aac",
            ]:
                return False
            elif file_types == "video" and ext not in [
                ".mp4",
                ".avi",
                ".mov",
                ".wmv",
                ".flv",
                ".webm",
            ]:
                return False

        return True

    def generate_tree_structure(self, folder_path, items, include_subfolders=False):
        """生成文件夹的表状树结构字符串"""
        import os

        lines = [os.path.basename(folder_path) + "/"]

        if include_subfolders:
            # 对于包含子文件夹的情况，需要更复杂的树结构生成
            dir_structure = {}
            for item in items:
                parts = item.split(os.sep)
                current = dir_structure
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                if "files" not in current:
                    current["files"] = []
                current["files"].append(parts[-1])

            # 递归生成树结构
            def build_tree(structure, prefix=""):
                result = []
                keys = sorted(structure.keys())
                for i, key in enumerate(keys):
                    is_last = i == len(keys) - 1
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    result.append(prefix + ("└── " if is_last else "├── ") + key + "/")

                    if isinstance(structure[key], dict):
                        result.extend(build_tree(structure[key], new_prefix))
                    elif key == "files":
                        files = sorted(structure[key])
                        for j, file in enumerate(files):
                            file_is_last = j == len(files) - 1
                            result.append(
                                new_prefix + ("└── " if file_is_last else "├── ") + file
                            )
                return result

            lines.extend(build_tree(dir_structure))
        else:
            # 简单情况：只显示当前文件夹
            for i, item in enumerate(sorted(items)):
                is_last = i == len(items) - 1
                prefix = "└── " if is_last else "├── "
                lines.append(prefix + item)

        return "\n".join(lines)


class BooleanSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_bool": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Input boolean value\n输入布尔值"},
                )
            }
        }

    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("input", "opposite")
    FUNCTION = "BooleanSplitter"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Split a boolean value into itself and its opposite.\n将布尔值拆分为自身和其相反值。"

    def BooleanSplitter(self, input_bool):
        opposite = not input_bool
        return (input_bool, opposite)


class ImageTriangulationWarp:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_image": (
                    "IMAGE",
                    {"tooltip": "Source image to be warped\n源图像，将被变形"},
                ),
                "target_mask": (
                    "MASK",
                    {
                        "tooltip": "Target mask to warp the source image to\n目标遮罩，将源图像变形到此形状"
                    },
                ),
                "n_points": (
                    "INT",
                    {
                        "default": 100,
                        "min": 10,
                        "max": 500,
                        "step": 10,
                        "tooltip": "Number of points to sample from contours\n从轮廓采样的点数",
                    },
                ),
            },
            "optional": {
                "source_mask": (
                    "MASK",
                    {
                        "tooltip": "Mask for the source image (optional, if not provided, the whole image will be used)\n源图像的遮罩（可选，如果不提供，将使用整个图像）"
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ImageTriangulationWarp"
    CATEGORY = "MakkiTools"
    DESCRIPTION = "Warp an image using triangulation based on source and target masks.\n基于源遮罩和目标遮罩使用三角剖分对图像进行变形。"

    def ImageTriangulationWarp(
        self, source_image, target_mask, n_points, source_mask=None
    ):
        import comfy.utils
        import torch

        # 如果没有提供source_mask，则创建一个与source_image同样大小的全白mask
        if source_mask is None:
            source_mask = torch.ones_like(source_image[..., 0])

        # 获取目标尺寸（以target_mask为准）
        target_h, target_w = target_mask.shape[-2], target_mask.shape[-1]

        # 统一调整图像和mask的尺寸
        # 调整source_image尺寸
        if source_image.shape[1] != target_h or source_image.shape[2] != target_w:
            source_image = source_image.permute(
                0, 3, 1, 2
            )  # [B, H, W, C] -> [B, C, H, W]
            source_image = comfy.utils.common_upscale(
                source_image, target_w, target_h, "bicubic", "center"
            )
            source_image = source_image.permute(
                0, 2, 3, 1
            )  # [B, C, H, W] -> [B, H, W, C]

        # 调整source_mask尺寸
        if source_mask.shape[-2] != target_h or source_mask.shape[-1] != target_w:
            source_mask = source_mask.unsqueeze(
                1
            )  # 添加通道维度 [B, H, W] -> [B, 1, H, W]
            source_mask = comfy.utils.common_upscale(
                source_mask, target_w, target_h, "bicubic", "center"
            )
            source_mask = source_mask.squeeze(
                1
            )  # 移除通道维度 [B, 1, H, W] -> [B, H, W]

        # 调整target_mask尺寸（以防万一）
        if target_mask.shape[-2] != target_h or target_mask.shape[-1] != target_w:
            target_mask = target_mask.unsqueeze(
                1
            )  # 添加通道维度 [B, H, W] -> [B, 1, H, W]
            target_mask = comfy.utils.common_upscale(
                target_mask, target_w, target_h, "bicubic", "center"
            )
            target_mask = target_mask.squeeze(
                1
            )  # 移除通道维度 [B, 1, H, W] -> [B, H, W]

        result = self._execute_warping(source_image, source_mask, target_mask, n_points)
        return (result,)

    def _execute_warping(self, source_image, source_mask, target_mask, n_points):
        from .warp_triangulation import warp_image_triangulation_comfyui

        # 调整mask维度为4D以适配函数（如果需要）
        if source_mask.dim() == 3:
            source_mask = source_mask.unsqueeze(0)
        if target_mask.dim() == 3:
            target_mask = target_mask.unsqueeze(0)

        # 如果source_image维度是3（单张图片），也添加批次维度
        if source_image.dim() == 3:
            source_image = source_image.unsqueeze(0)

        result = warp_image_triangulation_comfyui(
            source_image, source_mask, target_mask, n_points
        )
        return result


NODE_CLASS_MAPPINGS = {
    "GetImageNthCount_makki": GetImageNthCount,
    "ImageChannelSeparate_makki": ImageChannelSeparate,
    "ImageCountConcatenate_makki": ImageCountConcatenate,
    "MergeImageChannels_makki": MergeImageChannels,
    "ImageWidthStitch_makki": ImageWidthStitch,
    "ImageHeigthStitch_makki": ImageHeigthStitch,
    "AutoLoop_create_pseudo_loop_video_makki": AutoLoop_create_pseudo_loop_video,
    "Environment_INFO_makki": Environment_INFO,
    "translators_makki": translators,
    "translator_m2m100_makki": translator_m2m100,
    "random_any_makki": random_any,
    "AnyImageStitch_makki": AnyImageStitch,
    "AnyImagetoConditioning_flux_kontext_makki": AnyImagetoConditioning_flux_kontext,
    "show_type_makki": show_type,
    "timer_makki": timer,
    "Image_Resize_makki": Image_Resize,
    "Prism_Mirage_makki": Prism_Mirage,
    "int_calculate_statistics_makki": int_calculate_statistics,
    "BatchLoraLoader_makki": BatchLoraLoader,
    "UniversalInstaller_makki": UniversalInstaller,
    "get_folder_info_makki": get_folder_info,
    "BooleanSplitter_makki": BooleanSplitter,
    "ImageTriangulationWarp_makki": ImageTriangulationWarp,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageNthCount_makki": "GetImageNthCount(mki-获取第N张图像)",
    "ImageChannelSeparate_makki": "ImageChannelSeparate(mki-图像通道分离)",
    "ImageCountConcatenate_makki": "ImageCountConcatenate(mki-图像批次拼接)",
    "MergeImageChannels_makki": "MergeImageChannels(mki-图像通道合并)",
    "ImageWidthStitch_makki": "ImageWidthStitch(已弃用-mki-图像横向拼接)",
    "ImageHeigthStitch_makki": "ImageHeigthStitch(已弃用-mki-图像纵向拼接)",
    "AutoLoop_create_pseudo_loop_video_makki": "AutoLoop_create_pseudo_loop_video(已弃用-mki-创建伪循环视频)",
    "Environment_INFO_makki": "Environment_INFO(mki-环境信息显示)",
    "translators_makki": "translators(mki-多语言翻译)",
    "translator_m2m100_makki": "translator_m2m100(mki-多语言翻译-m2m100)",
    "random_any_makki": "random_any(mki-随机数输出)",
    "AnyImageStitch_makki": "AnyImageStitch(mki-任意数量图像拼接)",
    "AnyImagetoConditioning_flux_kontext_makki": "AnyImagetoConditioning_flux_kontext(mki-任意数量图像转条件-flux_kontext)",
    "show_type_makki": "show_type(mki-数据类型显示)",
    "timer_makki": "timer(mki-计时器)",
    "Image_Resize_makki": "Image_Resize(mki-图像大小调整)",
    "Prism_Mirage_makki": "Prism_Mirage(mki-光棱坦克)",
    "int_calculate_statistics_makki": "int_calculate_statistics(mki-整数计算统计)",
    "BatchLoraLoader_makki": "BatchLoraLoader(mki-批量LoRA加载)",
    "UniversalInstaller_makki": "UniversalInstaller(mki-安装Python包)",
    "get_folder_info_makki": "get_folder_info(mki-获取文件夹文件信息)",
    "BooleanSplitter_makki": "BooleanSplitter(mki-布尔值拆分)",
    "ImageTriangulationWarp_makki": "ImageTriangulationWarp(mki-三角剖分图像变形)",
}
