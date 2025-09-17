class GetImageNthCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "Nth_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 2 * 31 - 1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "GetImageNthCount"
    CATEGORY = "MakkiTools"

    def GetImageNthCount(self, image, Nth_count):
        return (image[Nth_count - 1 : Nth_count],)


class ImageChannelSeparate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["red", "green", "blue", "alpha"], {"default": "red"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ImageChannelSeparate"
    CATEGORY = "MakkiTools"

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
                "red_channel": ("IMAGE",),
                "green_channel": ("IMAGE",),
                "blue_channel": ("IMAGE",),
                "alpha_channel": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "MergeImageChannels"
    CATEGORY = "MakkiTools"

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
                "frames": ("IMAGE",),
                "transition_duration": (
                    "FLOAT",
                    {"default": 0.2, "min": 0, "max": 0.5, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "AutoLoop_create_pseudo_loop_video"
    CATEGORY = "MakkiTools"

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
                "SYSTEM_INFO": ("BOOLEAN", {"default": True}),
                "HARDWARE_INFO": ("BOOLEAN", {"default": True}),
                "GPU_INFO": ("BOOLEAN", {"default": True}),
                "DEEP_LEARNING_FRAMEWORKS_INFO": ("BOOLEAN", {"default": True}),
                "ALL_INSTALLED_PACKAGES_INFO": ("BOOLEAN", {"default": True}),
                "CUSTOM_NODES_FOLDERS_INFO": ("BOOLEAN", {"default": True}),
            },
            "optional": {"anything": (s.any_type, {})},
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("INFO", "anything")
    OUTPUT_NODE = True
    FUNCTION = "Environment_INFO"
    CATEGORY = "MakkiTools"

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
                "query_text": ("STRING", {"multiline": True}),
                "translator": (
                    list(ts.translators_pool),
                    {"default": list(ts.translators_pool)[0]},
                ),
                "from_language": (
                    ["auto"] + Supported_Languages,
                    {"default": "auto"},
                ),
                "to_language": (
                    Supported_Languages,
                    {"default": Supported_Languages[0]},
                ),
                "if_use_preacceleration": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translators"
    CATEGORY = "MakkiTools"

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

        output = self.ts.translate_text(
            query_text,
            translator=translator,
            from_language=from_language,
            to_language=to_language,
            if_use_preacceleration=if_use_preacceleration,
        )

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
                "query_text": ("STRING", {"multiline": True}),
                "model": (
                    [
                        "facebook/m2m100_418M",
                        "facebook/m2m100_1.2B",
                        "facebook/m2m100-12B-avg-5-ckpt",
                        "facebook/m2m100-12B-avg-10-ckpt",
                        "facebook/m2m100-12B-last-ckpt",
                    ],
                    {"default": "facebook/m2m100_418M"},
                ),
                "from_language": (
                    ["auto"] + m2m100map,
                    {"default": "auto"},
                ),
                "to_language": (
                    m2m100map,
                    {"default": "English (en)"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "8bit"},
                ),
                "attention": (
                    ["flash_attention_2", "sdpa", "eager"],
                    {"default": "sdpa"},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "translator_m2m100"
    CATEGORY = "MakkiTools"

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
        return {"optional": {"anything": (s.any_type, {})}}

    RETURN_TYPES = (any_type, "INT", "FLOAT")
    RETURN_NAMES = ("any", "int", "float")
    FUNCTION = "random_any"
    CATEGORY = "MakkiTools"

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
                "dimension": (["horizontal", "vertical"], {"default": "horizontal"}),
                "reference_type": (
                    ["first image", "custom"],
                    {"default": "first image"},
                ),
                "reference_value": (
                    "INT",
                    {"default": 512, "min": 1, "max": 4096, "step": 1},
                ),
                "upscale_method": (
                    s.upscale_methods,
                    {"default": "bicubic"},
                ),
                "crop": (s.crop_methods, {"default": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "AnyImageStitch"
    CATEGORY = "MakkiTools"

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
                "conditioning": ("CONDITIONING",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "AnyImagetoConditioning_flux_kontext"
    CATEGORY = "MakkiTools"

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
        return {"optional": {"anything": (s.any_type, {})}}

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "show_type"
    CATEGORY = "MakkiTools"

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
                "mode": (["start", "stop"],),
                "format": (["ms", "s", "m", "auto"], {"default": "auto"}),
            },
            "optional": {
                "any": (s.any_type, {}),
                "timer": ("TIMER_MAKKI",),
            },
        }

    RETURN_TYPES = (any_type, "TIMER_MAKKI", "INT", "STRING")
    RETURN_NAMES = ("any", "timer", "time_ms", "time_str")
    OUTPUT_NODE = True
    FUNCTION = "timer"
    CATEGORY = "MakkiTools"

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
                "upscale_method": (s.upscale_methods, {"default": "auto"}),
                "crop_method": (s.crop_methods, {"default": "center"}),
                "size": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "GCD": ("INT", {"default": 1, "min": 1, "max": 512, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {"min": 1, "max": 8192, "step": 1, "defaultInput": True},
                ),
                "height": (
                    "INT",
                    {"min": 1, "max": 8192, "step": 1, "defaultInput": True},
                ),
                "short_side": (
                    "INT",
                    {"min": 1, "max": 8192, "step": 1, "defaultInput": True},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "Image_Resize"
    CATEGORY = "MakkiTools"

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
                "exterior_image": ("IMAGE",),
                "interior_image": ("IMAGE",),
                "exterior_level": (
                    "INT",
                    {"default": 42, "min": 0, "max": 255, "step": 1},
                ),
                "interior_level": (
                    "INT",
                    {"default": 24, "min": 0, "max": 255, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "Prism_Mirage"
    CATEGORY = "MakkiTools"

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
                    {"default": "count"},
                ),
            }
        }

    RETURN_TYPES = (any_type,)
    FUNCTION = "int_calculate_statistics"
    CATEGORY = "MakkiTools"

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
                    {"tooltip": "The diffusion model the LoRA will be applied to."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model the LoRA will be applied to."},
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
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"strength_model_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How strongly to modify the diffusion model. This value can be negative.",
                },
            )
            inputs["required"][f"strength_clip_{i}"] = (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": -100.0,
                    "max": 100.0,
                    "step": 0.01,
                    "tooltip": "How strongly to modify the CLIP model. This value can be negative.",
                },
            )

        return inputs

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "BatchLoraLoader"

    CATEGORY = "MakkiTools"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

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
                        "tooltip": "The name of the package to install, such as 'requests' or 'tipo-kgen==0.2.0'.",
                    },
                ),
                "backend": (["pip", "uv"], {"default": "pip"}),
                "mirror_source": (
                    ["default", "tsinghua", "aliyun", "douban", "ustc", "custom"],
                    {
                        "default": "default",
                        "tooltip": "Select the mirror source, use the content in extra_index_url when custom.",
                    },
                ),
                "update_if_no_version": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "When the package name does not specify a version, should the -U parameter be used to update to the latest version.",
                    },
                ),
            },
            "optional": {
                "extra_index_url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Additional index URLs (e.g., private sources), optional. Only used when 'mirror_source' is set to 'custom'.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "UniversalInstaller"
    CATEGORY = "MakkiTools"

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
                        "tooltip": "要扫描的文件夹路径，例如 'C:/Users/MyUser/Documents' 或 '/home/user/documents'",
                    },
                ),
                "file_index": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "要获取的第几个文件名（按名称排序），从1开始计数",
                    },
                ),
            },
            "optional": {
                "include_subfolders": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "是否包含子文件夹中的文件",
                    },
                ),
                "file_filter": (
                    "STRING",
                    {
                        "default": "*",
                        "tooltip": "文件筛选器，例如 '*.txt' 或 'image*.png'",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "LIST", "STRING", "LIST", "STRING", "STRING")
    RETURN_NAMES = (
        "filenames_str",
        "filenames_list",
        "full_paths_str",
        "full_paths_list",
        "nth_filename",
        "tree_structure",
    )
    FUNCTION = "get_folder_info"
    CATEGORY = "MakkiTools"

    def get_folder_info(
        self, folder_path, file_index, include_subfolders=False, file_filter="*"
    ):
        import os
        import fnmatch

        if include_subfolders:
            all_items = []
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if fnmatch.fnmatch(file, file_filter):
                        rel_path = os.path.relpath(
                            os.path.join(root, file), folder_path
                        )
                        all_items.append(rel_path)
        else:
            all_items = [
                item
                for item in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, item))
                and fnmatch.fnmatch(item, file_filter)
            ]

        # 分离文件和文件夹（如果需要）
        files = sorted(all_items)  # 直接排序所有匹配的文件

        # 获取所有文件的完整路径
        if include_subfolders:
            full_paths = [os.path.join(folder_path, item) for item in files]
        else:
            full_paths = [os.path.join(folder_path, item) for item in files]

        # 输出第n个文件名（按名称排序，n从1开始计数）
        nth_file = (
            files[file_index - 1] if 0 < file_index <= len(files) else "索引超出范围"
        )

        # 生成表状文件夹结构字符串
        tree_structure = self.generate_tree_structure(
            folder_path, files, include_subfolders
        )

        # 返回所有要求的信息
        return (
            "\n".join(files),  # 所有文件名（字符串，每行一个）
            files,  # 所有文件名（列表）
            "\n".join(full_paths),  # 所有文件完整路径（字符串，每行一个）
            full_paths,  # 所有文件完整路径（列表）
            nth_file,  # 第n个文件名
            tree_structure,  # 表状文件夹结构
        )

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
}
