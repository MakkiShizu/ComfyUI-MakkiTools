import os
import re
import folder_paths
from langdetect import detect
from transformers import (
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    BitsAndBytesConfig,
)
from huggingface_hub import snapshot_download


class M2M100Translator:
    def __init__(self, model_repo: str, quantization: str, attention: str):
        model_directory = os.path.join(folder_paths.models_dir, "m2m100")
        os.makedirs(model_directory, exist_ok=True)
        model_name = model_repo.rsplit("/", 1)[-1]
        self.model_path = os.path.join(model_directory, model_name)
        self.model_repo = model_repo
        self.model = None
        self.tokenizer = None

        # 确保模型存在
        self._download_model()
        # 加载模型
        self._load_model(quantization, attention)

    def _download_model(self):
        """下载模型到本地目录（如果不存在）"""
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_repo} model to: {self.model_path}")
            os.makedirs(self.model_path, exist_ok=True)
            snapshot_download(
                repo_id=self.model_repo,
                local_dir=self.model_path,
                local_dir_use_symlinks=False,
            )

    def _load_model(self, quantization: str, attention: str):
        """从本地路径加载模型和分词器"""
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
            )
        elif quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        else:
            quantization_config = None

        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attention,
            quantization_config=quantization_config,
        )
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_path)

    def _translate_segment(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """翻译单个文本段"""
        if src_lang == "auto":
            src_lang = detect(text)
        self.tokenizer.src_lang = src_lang
        encoded_text = self.tokenizer(text, return_tensors="pt")
        encoded_text = encoded_text.to(self.model.device)
        generated_tokens = self.model.generate(
            **encoded_text, forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang)
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[
            0
        ]

    def translate_preserve_format(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        保持原始格式的翻译
        :param text: 要翻译的文本（包含段落）
        :param src_lang: 源语言代码
        :param tgt_lang: 目标语言代码
        :return: 保持格式的翻译文本
        """
        if src_lang == "auto":
            src_lang = detect(text)
            src_lang = self.normalize_language_code(src_lang)

        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not initialized properly")

        # 分割文本为段落（保留空行）
        segments = re.split(r"(\n\s*\n)", text)  # 使用正则表达式分割段落

        translated_segments = []

        for segment in segments:
            if re.match(r"^\n\s*\n$", segment):
                # 保留空行
                translated_segments.append(segment)
            elif segment.strip():
                # 处理非空段落
                # 对于列表项，按行翻译
                if segment.count("\n") >= 1:
                    lines = segment.split("\n")
                    translated_lines = []
                    for line in lines:
                        if line.strip():  # 跳过空行
                            translated_line = self._translate_segment(
                                line, src_lang, tgt_lang
                            )
                            translated_lines.append(translated_line)
                        else:
                            translated_lines.append("")
                    translated_segments.append("\n".join(translated_lines))
                else:
                    # 普通段落整体翻译
                    translated_segments.append(
                        self._translate_segment(segment, src_lang, tgt_lang)
                    )

        # 合并所有翻译段落
        return "".join(translated_segments)

    def normalize_language_code(self, Language):
        return LANGUAGE_MAPPING.get("langdetect", {}).get(Language, Language)


m2m100map = [
    "Afrikaans (af)",
    "Amharic (am)",
    "Arabic (ar)",
    "Asturian (ast)",
    "Azerbaijani (az)",
    "Bashkir (ba)",
    "Belarusian (be)",
    "Bulgarian (bg)",
    "Bengali (bn)",
    "Breton (br)",
    "Bosnian (bs)",
    "Catalan; Valencian (ca)",
    "Cebuano (ceb)",
    "Czech (cs)",
    "Welsh (cy)",
    "Danish (da)",
    "German (de)",
    "Greeek (el)",
    "English (en)",
    "Spanish (es)",
    "Estonian (et)",
    "Persian (fa)",
    "Fulah (ff)",
    "Finnish (fi)",
    "French (fr)",
    "Western Frisian (fy)",
    "Irish (ga)",
    "Gaelic; Scottish Gaelic (gd)",
    "Galician (gl)",
    "Gujarati (gu)",
    "Hausa (ha)",
    "Hebrew (he)",
    "Hindi (hi)",
    "Croatian (hr)",
    "Haitian; Haitian Creole (ht)",
    "Hungarian (hu)",
    "Armenian (hy)",
    "Indonesian (id)",
    "Igbo (ig)",
    "Iloko (ilo)",
    "Icelandic (is)",
    "Italian (it)",
    "Japanese (ja)",
    "Javanese (jv)",
    "Georgian (ka)",
    "Kazakh (kk)",
    "Central Khmer (km)",
    "Kannada (kn)",
    "Korean (ko)",
    "Luxembourgish; Letzeburgesch (lb)",
    "Ganda (lg)",
    "Lingala (ln)",
    "Lao (lo)",
    "Lithuanian (lt)",
    "Latvian (lv)",
    "Malagasy (mg)",
    "Macedonian (mk)",
    "Malayalam (ml)",
    "Mongolian (mn)",
    "Marathi (mr)",
    "Malay (ms)",
    "Burmese (my)",
    "Nepali (ne)",
    "Dutch; Flemish (nl)",
    "Norwegian (no)",
    "Northern Sotho (ns)",
    "Occitan (post 1500) (oc)",
    "Oriya (or)",
    "Panjabi; Punjabi (pa)",
    "Polish (pl)",
    "Pushto; Pashto (ps)",
    "Portuguese (pt)",
    "Romanian; Moldavian; Moldovan (ro)",
    "Russian (ru)",
    "Sindhi (sd)",
    "Sinhala; Sinhalese (si)",
    "Slovak (sk)",
    "Slovenian (sl)",
    "Somali (so)",
    "Albanian (sq)",
    "Serbian (sr)",
    "Swati (ss)",
    "Sundanese (su)",
    "Swedish (sv)",
    "Swahili (sw)",
    "Tamil (ta)",
    "Thai (th)",
    "Tagalog (tl)",
    "Tswana (tn)",
    "Turkish (tr)",
    "Ukrainian (uk)",
    "Urdu (ur)",
    "Uzbek (uz)",
    "Vietnamese (vi)",
    "Wolof (wo)",
    "Xhosa (xh)",
    "Yiddish (yi)",
    "Yoruba (yo)",
    "Chinese (zh)",
    "Zulu (zu)",
]
LANGUAGE_MAPPING = {"langdetect": {"zh-cn": "zh", "zh-tw": "zh"}}
