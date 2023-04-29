import gradio as gr

from modules import model_manager


def model_list_str():
    return [x.model_id for x in model_manager.sd_models]


def change_model(model_id: str):
    if model_id not in model_list_str():
        raise ValueError("Model not found.")
    model_manager.set_model(model_id)
    return model_id


def add_model(model_id: str):
    if model_id not in model_list_str():
        searched = model_manager.search_model(model_id)
        if len(searched) < 1:
            raise ValueError("Model not found.")
        model_manager.add_model(model_id)
    return gr.Dropdown.update(choices=model_list_str())


def ui():
    model_id = (
        model_manager.sd_model.model_id if model_manager.sd_model is not None else None
    )
    with gr.Row():
        with gr.Column(scale=0.25):
            with gr.Row():
                model_id_dropdown = gr.Dropdown(
                    value=model_id,
                    choices=model_list_str(),
                    show_label=False,
                )
                reload_models_button = gr.Button("🔄", elem_classes=["tool-button"])

        with gr.Column(scale=0.25):
            with gr.Row():
                add_model_textbox = gr.Textbox(
                    placeholder="Add model",
                    show_label=False,
                )
                add_model_button = gr.Button("💾", elem_classes=["tool-button"])

    model_id_dropdown.change(
        fn=change_model, inputs=[model_id_dropdown], outputs=[model_id_dropdown]
    )
    reload_models_button.click(
        fn=lambda: gr.Dropdown.update(
            choices=model_list_str(),
            value=model_manager.sd_model.model_id
            if model_manager.sd_model is not None
            else None,
        ),
        inputs=[],
        outputs=[model_id_dropdown],
    )
    add_model_button.click(
        fn=add_model, inputs=[add_model_textbox], outputs=[model_id_dropdown]
    )
