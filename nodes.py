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
        from .environment_info import AlwaysEqualProxy

        any_type = AlwaysEqualProxy("*")
        return {
            "required": {
                "SYSTEM_INFO": ("BOOLEAN", {"default": True}),
                "HARDWARE_INFO": ("BOOLEAN", {"default": True}),
                "GPU_INFO": ("BOOLEAN", {"default": True}),
                "DEEP_LEARNING_FRAMEWORKS_INFO": ("BOOLEAN", {"default": True}),
                "ALL_INSTALLED_PACKAGES_INFO": ("BOOLEAN", {"default": True}),
                "CUSTOM_NODES_FOLDERS_INFO": ("BOOLEAN", {"default": True}),
            },
            "optional": {"anything": (any_type, {})},
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
        from .environment_info import AlwaysEqualProxy

        any_type = AlwaysEqualProxy("*")
        return {"optional": {"anything": (any_type, {})}}

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
    @classmethod
    def INPUT_TYPES(s):
        from .environment_info import AlwaysEqualProxy

        any_type = AlwaysEqualProxy("*")
        return {"optional": {"anything": (any_type, {})}}

    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "show_type"
    CATEGORY = "MakkiTools"

    def show_type(self, anything):
        type_name = type(anything).__name__.lower()
        return {"ui": {"info": (type_name,)}, "result": (type_name,)}


NODE_CLASS_MAPPINGS = {
    "GetImageNthCount": GetImageNthCount,
    "ImageChannelSeparate": ImageChannelSeparate,
    "ImageCountConcatenate": ImageCountConcatenate,
    "MergeImageChannels": MergeImageChannels,
    "ImageWidthStitch": ImageWidthStitch,
    "ImageHeigthStitch": ImageHeigthStitch,
    "AutoLoop_create_pseudo_loop_video": AutoLoop_create_pseudo_loop_video,
    "Environment_INFO": Environment_INFO,
    "translators": translators,
    "translator_m2m100": translator_m2m100,
    "random_any": random_any,
    "AnyImageStitch": AnyImageStitch,
    "AnyImagetoConditioning_flux_kontext": AnyImagetoConditioning_flux_kontext,
    "show_type": show_type,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GetImageNthCount": "GetImageNthCount",
    "ImageChannelSeparate": "ImageChannelSeparate",
    "ImageCountConcatenate": "ImageCountConcatenate",
    "MergeImageChannels": "MergeImageChannels",
    "ImageWidthStitch": "ImageWidthStitch",
    "ImageHeigthStitch": "ImageHeigthStitch",
    "AutoLoop_create_pseudo_loop_video": "AutoLoop_create_pseudo_loop_video",
    "Environment_INFO": "Environment_INFO",
    "translators": "translators",
    "translator_m2m100": "translator_m2m100",
    "random_any": "random_any",
    "AnyImageStitch": "AnyImageStitch",
    "AnyImagetoConditioning_flux_kontext": "AnyImagetoConditioning_flux_kontext",
    "show_type": "show_type",
}
