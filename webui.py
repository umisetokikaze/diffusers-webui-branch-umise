from modules import config, model_manager, ui


def pre_load():
    config.init()
    model_manager.set_default_model()


def webui():
    pre_load()
    app = ui.create_ui()
    app.queue(64)
    app, local_url, share_url = app.launch(
        server_name=config.get("host"),
        server_port=config.get("port"),
        share=config.get("share"),
    )


if __name__ == "__main__":
    webui()
