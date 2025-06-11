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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("image",)
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
    RETURN_NAMES = ("image",)
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


from .environment_info import (
    AlwaysEqualProxy,
    format_environment_info,
    get_environment_info,
)

any_type = AlwaysEqualProxy("*")


class Environment_INFO:
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
            "optional": {"anything": (any_type, {})},
        }

    RETURN_TYPES = ("STRING", any_type)
    RETURN_NAMES = ("INFO", "anything")
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
        env_info = get_environment_info()
        full_report = format_environment_info(
            env_info,
            SYSTEM_INFO,
            HARDWARE_INFO,
            GPU_INFO,
            DEEP_LEARNING_FRAMEWORKS_INFO,
            ALL_INSTALLED_PACKAGES_INFO,
            CUSTOM_NODES_FOLDERS_INFO,
        )

        return (full_report, anything)


import translators as ts

Supported_Languages = [
    "english(en)",
    "chinese(zh)",
    "arabic(ar)",
    "russian(ru)",
    "french(fr)",
    "german(de)",
    "spanish(es)",
    "portuguese(pt)",
    "italian(it)",
    "japanese(ja)",
    "korean(ko)",
    "greek(el)",
    "dutch(nl)",
    "hindi(hi)",
    "turkish(tr)",
    "malay(ms)",
    "thai(th)",
    "vietnamese(vi)",
    "indonesian(id)",
    "hebrew(he)",
    "polish(pl)",
    "mongolian(mn)",
    "czech(cs)",
    "hungarian(hu)",
    "estonian(et)",
    "bulgarian(bg)",
    "danish(da)",
    "finnish(fi)",
    "romanian(ro)",
    "swedish(sv)",
    "slovenian(sl)",
    "persian/farsi(fa)",
    "bosnian(bs)",
    "serbian(sr)",
    "fijian(fj)",
    "filipino(tl)",
    "haitiancreole(ht)",
    "catalan(ca)",
    "croatian(hr)",
    "latvian(lv)",
    "lithuanian(lt)",
    "urdu(ur)",
    "ukrainian(uk)",
    "welsh(cy)",
    "tahiti(ty)",
    "tongan(to)",
    "swahili(sw)",
    "samoan(sm)",
    "slovak(sk)",
    "afrikaans(af)",
    "norwegian(no)",
    "bengali(bn)",
    "malagasy(mg)",
    "maltese(mt)",
    "queretaro otomi(otq)",
    "klingon/tlhingan hol(tlh)",
    "gujarati(gu)",
    "tamil(ta)",
    "telugu(te)",
    "punjabi(pa)",
    "amharic(am)",
    "azerbaijani(az)",
    "bashkir(ba)",
    "belarusian(be)",
    "cebuano(ceb)",
    "chuvash(cv)",
    "esperanto(eo)",
    "basque(eu)",
    "irish(ga)",
    "emoji(emj)",
    "Chinese(简体)(zh-CHS)",
    "Chinese(繁体)(zh-CHT)",
    "Chinese(文言文)(wyw)",
    "Chinese(粤语)(yue)",
    "Chinese(内蒙语)(mn)",
    "Chinese(维吾尔语)(uy)",
    "Chinese(藏语)(ti)",
    "Chinese(白苗文)(mww)",
    "Chinese(彝语)(ii)",
    "Chinese(苗语)(hmn)",
    "Chinese(壮语)(zyb)",
]
LANGUAGE_MAPPING = {
    "google": {
        "he": "iw",
        "jw": "jv",
        "zh-CHS": "zh-CN",
        "zh-CHT": "zh-TW",
    },
    "yandex": {
        "zh-CHS": "zh",
    },
    "bing": {
        "bs": "bs-Latn",
        "sr": "sr-Latn",
        "tl": "fil",
        "bn": "bn-BD",
        "zh-CHS": "zh-Hans",
        "zh-CHT": "zh-Hant",
    },
    "baidu": {
        "ar": "ara",
        "fr": "fra",
        "es": "spa",
        "ja": "jp",
        "ko": "kor",
        "vi": "vie",
        "et": "est",
        "bg": "bul",
        "da": "dan",
        "fi": "fin",
        "ro": "rom",
        "sv": "swe",
        "sl": "slo",
        "zh-CHS": "zh",
        "zh-CHT": "cht",
    },
    "alibaba": {
        "zh-CHS": "zh",
        "zh-CHT": "zh-TW",
    },
    "tencent": {
        "zh-CHS": "zh",
    },
    "sogou": {
        "bs": "bs-Latn",
        "sr": "sr-Latn",
        "tl": "fil",
    },
    "iciba": {
        "zh-CHS": "zh",
        "zh-CHT": "cnt",
    },
    "iflytek": {
        "zh-CHS": "zh",
    },
    "caiyun": {
        "zh-CHS": "zh",
    },
    "deepl": {
        "zh-CHS": "zh",
    },
    "argos": {
        "zh-CHS": "zh",
    },
    "itranslate": {
        "zh-CHS": "zh-CN",
        "zh-CHT": "zh-TW",
        "yue": "zh-HK",
    },
    "reverso": {
        "zh-CHS": "zh",
    },
    "papago": {
        "zh-CHS": "zh-CN",
        "zh-CHT": "zh-TW",
    },
    "utibet": {
        "zh-CHS": "zh",
    },
}


class translators:
    _pre_acceleration_done = False

    @classmethod
    def INPUT_TYPES(s):
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
    RETURN_NAMES = ("text",)
    FUNCTION = "translators"
    CATEGORY = "MakkiTools"

    def normalize_language_code(self, translator, Language):
        return LANGUAGE_MAPPING.get(translator, {}).get(Language, Language)

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

        if if_use_preacceleration and not self.__class__._pre_acceleration_done:
            _ = ts.preaccelerate_and_speedtest()
            self.__class__._pre_acceleration_done = True

        output = ts.translate_text(
            query_text,
            translator=translator,
            from_language=from_language,
            to_language=to_language,
            if_use_preacceleration=if_use_preacceleration,
        )

        return (output,)


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
}
