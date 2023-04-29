import gc
import os
import random
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import *
from typing import Optional, Union

import torch

from api.models.diffusion import DenoiseLatentData, ImageGenerationOptions
from lib.tensorrt.pipeline_stable_diffusion import TensorRTStableDiffusionPipeline
from lib.tensorrt.pipeline_stable_diffusion_img2img import (
    TensorRTStableDiffusionImg2ImgPipeline,
)
from modules import config
from modules.acceleration.tensorrt.text_encoder import TensorRTCLIPTextModel
from modules.diffusion.lpw import LongPromptWeightingPipeline
from modules.images import save_image
from modules.model import StableDiffusionModel

from .runner import BaseRunner


class TensorRTDiffusionRunner(BaseRunner):
    def __init__(self, model: StableDiffusionModel) -> None:
        super().__init__(model)

        model_dir = model.get_trt_path()

        self.engine_dir = os.path.join(model_dir, "engine")
        self.onnx_dir = os.path.join(model_dir, "onnx")
        self.activate()

    def activate(self):
        self.loading = True
        self.pipe: Union[
            TensorRTStableDiffusionPipeline, TensorRTStableDiffusionImg2ImgPipeline
        ] = TensorRTStableDiffusionPipeline.from_pretrained(
            model_id=self.model.model_id,
            onnx_dir=self.onnx_dir,
            engine_dir=self.engine_dir,
            use_auth_token=config.get("hf_token"),
            device=torch.device("cuda"),
            max_batch_size=1,
        )
        self.loading = False
        self.text_encoder = TensorRTCLIPTextModel(
            self.pipe.engine["clip"], self.pipe.stream
        )

        self.lpw = LongPromptWeightingPipeline(
            self.text_encoder, self.pipe.tokenizer, self.pipe.device
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
            return self.lpw(
                prompt,
                negative_prompt,
                num_images_per_prompt,
                max_embeddings_multiples=1,
            ).to(dtype=torch.float16)

        self.pipe._encode_prompt = _encode_prompt

    def teardown(self):
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

            def callback(
                step: int,
                timestep: torch.Tensor,
                latents: torch.Tensor,
            ):
                data = DenoiseLatentData(
                    step=(opts.steps * i) + step,
                )
                queue.put(data)

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
                    prompt=opts.prompt,
                    negative_prompt=opts.negative_prompt,
                    image_height=opts.image_height,
                    image_width=opts.image_width,
                    guidance_scale=opts.scale,
                    num_inference_steps=opts.steps,
                    generator=generator,
                    batch_size=opts.batch_size,
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

        return results
