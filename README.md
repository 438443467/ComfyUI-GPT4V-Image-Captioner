# ComfyUI-GPT4V-Image-Captioner
你可以通过填写KEY和URL快捷地调用GPT视觉模型进行识别和打标。该项目是我个人对GPT4V-Image-Captioner repository (https://github.com/jiayev/GPT4V-Image-Captioner)项目的移植，以方便在ComfyUI中快捷地使用GPT进行打标。我衷心感谢GPT4V-Image-Captioner存储库的原作者们，感谢他们的宝贵贡献。该项目临时储存了部分我的个人节点，导致项目显得混乱和臃肿，对此我计划逐步停止更新此项目，并将其拆分为ComfyUI-GPT4V-Image-Captioner和ComfyUI-SanMian-Nodes两个部分来发布。

### Input Image Processing:
The project automatically handles image processing, eliminating the need for manual scaling.

### GPT4V Integration:
To utilize GPT4V for image annotation, you need to provide the API key and API URL.

### Seed and OpenAI Labeling:
The seed value determines the consistency of the labeling process. If the seed and image remain unchanged, there will be no additional calls to OpenAI for labeling.
If you are not satisfied with the labeling results, you can modify the seed value to get different outputs.

### Prompt Types:
There are two prompt types available: "generic" and "figure".
Under the "figure" type, labels will exclude color terms, backgrounds, watermarks, signatures, etc., focusing only on the overall characteristics of the characters.

### Weighted Labels:
The option to enable weights for labels is available.
When weights are enabled, corresponding weight values will be assigned to the labels.

### Excluding Unwanted Words:
If there are specific words in the labels that you want to exclude, you can enter them in the "exclude_words" field.
The system will automatically exclude the specified words from the generated labels.

