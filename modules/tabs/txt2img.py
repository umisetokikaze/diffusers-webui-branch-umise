import gradio as gr

from api.models.diffusion import DenoiseLatentData, ImageGenerationOptions
from modules import model_manager
from modules.components import image_generation_options
from modules.ui import Tab


class Txt2Img(Tab):
    def title(self):
        return "txt2img"

    def sort(self):
        return 1

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        sampler_name: str,
        sampling_steps: int,
        batch_size: int,
        batch_count: int,
        cfg_scale: float,
        width: int = 512,
        height: int = 512,
        seed: int = -1,
    ):
        if model_manager.runner is None:
            yield None, "Please select a model."

        yield [], "Generating...", gr.Button.update(
            value="Generating...", variant="secondary", interactive=False
        )

        for data in model_manager.runner.generate(
            ImageGenerationOptions(
                prompt=prompt,
                negative_prompt=negative_prompt,
                batch_size=batch_size,
                batch_count=batch_count,
                scheduler_id=sampler_name,
                steps=sampling_steps,
                scale=cfg_scale,
                image_height=height,
                image_width=width,
                seed=seed,
            )
        ):
            if isinstance(data, DenoiseLatentData):
                progress = data.step / (batch_count * sampling_steps)
                yield [], f"Progress: {progress * 100:.2f}%, Step: {data.step}", gr.Button.update(
                    value="Generating...", variant="secondary", interactive=False
                )
            else:
                image = data

        results = []
        for images, opts in image:
            results.extend(images)

        yield results, "Finished", gr.Button.update(
            value="Generate", variant="primary", interactive=True
        )

    def ui(self, outlet):
        generate_button, prompts, options, outputs = image_generation_options.ui()

        generate_button.click(
            fn=self.generate_image,
            inputs=[*prompts, *options],
            outputs=[*outputs, generate_button],
        )
