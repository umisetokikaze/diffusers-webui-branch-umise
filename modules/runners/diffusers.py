import gc
import os
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import *

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from api.models.diffusion import DenoiseLatentData, ImageGenerationOptions
from modules import config, utils
from modules.diffusion.lpw import LongPromptWeightingPipeline
from modules.images import save_image
from modules.model import StableDiffusionModel

from .runner import BaseRunner


class DiffusersDiffusionRunner(BaseRunner):
    def __init__(self, model: StableDiffusionModel) -> None:
        super().__init__(model)
        self.activate()

    def activate(self) -> None:
        self.loading = True
        cache_dir = os.path.join(config.get("model_dir"), "diffusers")
        os.makedirs(cache_dir, exist_ok=True)
        self.pipe: Union[
            StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        ] = StableDiffusionPipeline.from_pretrained(
            self.model.model_id,
            use_auth_token=config.get("hf_token"),
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
        ).to(
            torch.device("cuda")
        )

        self.pipe.safety_checker = None
        self.pipe.enable_attention_slicing()
        if utils.is_installed("xformers") and config.get("xformers"):
            self.pipe.enable_xformers_memory_efficient_attention()
        self.loading = False

        self.lpw = LongPromptWeightingPipeline(
            self.pipe.text_encoder, self.pipe.tokenizer, self.pipe.device
        )

        def _encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ):
            return self.lpw(prompt, negative_prompt, num_images_per_prompt)

        self.pipe._encode_prompt = _encode_prompt

    def teardown(self) -> None:
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()

    def generate(self, opts: ImageGenerationOptions):
        self.wait_loading()
        if opts.seed is None or opts.seed == -1:
            opts.seed = random.randrange(0, 4294967294, 1)

        results = []

        for i in range(opts.batch_count):
            manual_seed = opts.seed + i

            generator = torch.Generator(device=self.pipe.device).manual_seed(
                manual_seed
            )

            queue = Queue()
            done = object()

            def callback(
                step: int,
                timestep: torch.Tensor,
                latents: torch.Tensor,
            ):
                data = DenoiseLatentData(
                    step=(opts.steps * i) + step,
                )
                queue.put(data)

            def on_done(feature):
                queue.put(done)

            with ThreadPoolExecutor() as executer:
                feature = executer.submit(
                    self.pipe,
                    prompt=[opts.prompt] * opts.batch_size,
                    negative_prompt=[opts.negative_prompt] * opts.batch_size,
                    height=opts.image_height,
                    width=opts.image_width,
                    guidance_scale=opts.scale,
                    num_inference_steps=opts.steps,
                    generator=generator,
                    callback=callback,
                )
                feature.add_done_callback(on_done)

                while True:
                    data = queue.get()
                    if data is done:
                        break
                    else:
                        yield data

                images = feature.result().images

            results.append(
                (
                    images,
                    ImageGenerationOptions.parse_obj(
                        {"seed": manual_seed, **opts.dict()}
                    ),
                )
            )
            for img in images:
                save_image(img, opts)

        yield results
