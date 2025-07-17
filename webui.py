
import logging
from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
from typing import Optional

logger = logging.getLogger(__name__)

import gradio as gr

from src.i18n import i18n
from src.utils.agent_state import AgentState
from src.utils import utils
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot
from src.config import Config
from src.browser_manager import BrowserManager
from src.agent_runner import AgentRunner

# Global variables for persistence
_global_browser_manager: Optional[BrowserManager] = None
_global_agent_state = AgentState()

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state

    try:
        _global_agent_state.request_stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        return (
            message,
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False),
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(config: Config):
    global _global_agent_state, _global_browser_manager
    _global_agent_state.clear_stop()

    try:
        if not config.enable_recording:
            config.save_recording_path = None

        if config.save_recording_path:
            os.makedirs(config.save_recording_path, exist_ok=True)

        existing_videos = set()
        if config.save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(config.save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(config.save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        llm = utils.get_llm_model(
            provider=config.llm_provider,
            model_name=config.llm_model_name,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url,
            api_key=config.llm_api_key,
        )

        if _global_browser_manager is None:
            _global_browser_manager = BrowserManager(config)

        agent_runner = AgentRunner(config, _global_browser_manager, _global_agent_state)
        history, history_file = await agent_runner.run(llm)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(config.save_trace_path)

        latest_video = None
        if config.save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(config.save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(config.save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file.get(".zip"),
            history_file,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',
            errors,
            '',
            '',
            None,
            None,
            None,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )
    finally:
        if not config.keep_browser_open:
            if _global_browser_manager:
                await _global_browser_manager.close()
                _global_browser_manager = None

async def run_with_stream(config: Config):
    global _global_agent_state, _global_browser_manager
    stream_vw = 80
    stream_vh = int(80 * config.window_h // config.window_w)
    if not config.headless:
        result = await run_browser_agent(config)
        html_content = f'''
        <head>
            <meta name="description" content="A modern, animated, and SEO-optimized landing page for a construction company.">
        </head>
        <h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>
        '''
        yield [html_content] + list(result)
    else:
        try:
            _global_agent_state.clear_stop()
            agent_task = asyncio.create_task(run_browser_agent(config))

            while not agent_task.done():
                if _global_browser_manager and _global_browser_manager._context:
                    screenshot = await capture_screenshot(_global_browser_manager._context, stream_vw, stream_vh)
                    yield [screenshot, None, None, None, None, None, None, gr.update(interactive=True), gr.update(interactive=True)]
                await asyncio.sleep(1)

            result = await agent_task
            yield [None] + list(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            errors = str(e) + "\n" + traceback.format_exc()
            yield [None, None, errors, None, None, None, None, gr.update(interactive=True), gr.update(interactive=True)]

def create_ui(config: Config):
    with gr.Blocks(
        title="Agent Browser",
        theme=config.theme,
        css=".gradio-container { max-width: 100% !important; }",
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                # Left panel for configuration
                with gr.Accordion("General Settings", open=True):
                    with gr.Row():
                        agent_type = gr.Radio(
                            ["custom", "org"],
                            value=config.agent_type,
                            label="Agent Type",
                        )
                        theme_dropdown = gr.Dropdown(
                            ["origin", "citrus", "default", "glass", "monochrome", "ocean", "soft"],
                            value=config.theme,
                            label="Theme",
                        )
                with gr.Accordion("LLM Settings", open=True):
                    llm_provider = gr.Dropdown(
                        ["Ollama", "OpenAI", "Groq", "Anthropic", "Google", "Mistral"],
                        value=config.llm_provider,
                        label="LLM Provider",
                    )
                    llm_model_name = gr.Dropdown(
                        update_model_dropdown(config.llm_provider),
                        value=config.llm_model_name,
                        label="LLM Model",
                        interactive=True,
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=config.llm_temperature,
                        label="LLM Temperature",
                    )
                    llm_base_url = gr.Textbox(
                        value=config.llm_base_url,
                        label="LLM Base URL (Optional)",
                    )
                    llm_api_key = gr.Textbox(
                        value=config.llm_api_key,
                        label="LLM API Key (Optional)",
                        type="password",
                    )

                with gr.Accordion("Browser Settings", open=True):
                    use_own_browser = gr.Checkbox(
                        value=config.use_own_browser,
                        label="Use own browser (via CHROME_PATH and CHROME_USER_DATA in .env)",
                    )
                    keep_browser_open = gr.Checkbox(
                        value=config.keep_browser_open,
                        label="Keep browser open after run",
                    )
                    headless = gr.Checkbox(
                        value=config.headless,
                        label="Headless mode",
                    )
                    disable_security = gr.Checkbox(
                        value=config.disable_security,
                        label="Disable security measures (use with caution)",
                    )
                    with gr.Row():
                        window_w = gr.Slider(
                            minimum=1024,
                            maximum=1920,
                            step=1,
                            value=config.window_w,
                            label="Window Width",
                        )
                        window_h = gr.Slider(
                            minimum=768,
                            maximum=1080,
                            step=1,
                            value=config.window_h,
                            label="Window Height",
                        )

                with gr.Accordion("Agent Settings", open=True):
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=config.max_steps,
                        label="Max Steps",
                    )
                    use_vision = gr.Checkbox(
                        value=config.use_vision,
                        label="Use Vision",
                    )
                    max_actions_per_step = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=config.max_actions_per_step,
                        label="Max actions per step",
                    )
                    tool_calling_method = gr.Radio(
                        ["tool_calling", "json"],
                        value=config.tool_calling_method,
                        label="Tool Calling Method",
                    )

                with gr.Accordion("Files Settings", open=True):
                    with gr.Row():
                        save_recording_path = gr.Textbox(
                            value=config.save_recording_path,
                            label="Save Recording Path",
                        )
                        enable_recording = gr.Checkbox(
                            value=config.enable_recording,
                            label="Enable Recording",
                        )
                    save_agent_history_path = gr.Textbox(
                        value=config.save_agent_history_path,
                        label="Save Agent History Path",
                    )
                    save_trace_path = gr.Textbox(
                        value=config.save_trace_path,
                        label="Save Trace Path",
                    )

            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.TabItem("Agent"):
                        with gr.Column():
                            # Main content area
                            with gr.Row():
                                task = gr.Textbox(
                                    label="Task",
                                    placeholder="Enter the task for the agent",
                                    lines=1,
                                    scale=4,
                                )
                                run_button = gr.Button("Run", variant="primary")
                                stop_button = gr.Button("Stop", variant="stop")

                            add_infos = gr.Textbox(
                                label="Additional Information",
                                placeholder="Enter additional information for the agent",
                                lines=3,
                            )

                            # Outputs
                            with gr.Accordion("Final Result", open=True):
                                final_result_output = gr.Markdown()

                            with gr.Accordion("Screenshots", open=False):
                                screenshot_output = gr.HTML()

                            with gr.Accordion("Video", open=True):
                                video_output = gr.Video()

                            with gr.Accordion("Errors", open=False):
                                errors_output = gr.Textbox(label="Errors", lines=5)

                            with gr.Accordion("Model Actions", open=False):
                                model_actions_output = gr.Code(
                                    label="Model Actions", language="json"
                                )

                            with gr.Accordion("Model Thoughts", open=False):
                                model_thoughts_output = gr.Markdown()

                            with gr.Accordion("Files", open=True):
                                with gr.Row():
                                    trace_file_output = gr.File(label="Trace File")
                                    history_file_output = gr.File(label="History File")

        # Event handlers
        llm_provider.change(
            fn=update_model_dropdown, inputs=llm_provider, outputs=llm_model_name
        )
        theme_dropdown.change(
            fn=None,
            inputs=theme_dropdown,
            js='''
            (theme) => {
                if (theme) {
                    const url = new URL(window.location);
                    url.searchParams.set('__theme', theme);
                    window.location.href = url.href;
                }
            }
            ''',
        )

        run_button.click(
            fn=lambda *args: run_with_stream(Config(*args)),
            inputs=[
                agent_type,
                llm_provider,
                llm_model_name,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                window_w,
                window_h,
                save_recording_path,
                save_agent_history_path,
                save_trace_path,
                enable_recording,
                task,
                add_infos,
                max_steps,
                use_vision,
                max_actions_per_step,
                tool_calling_method,
            ],
            outputs=[
                screenshot_output,
                final_result_output,
                errors_output,
                model_actions_output,
                model_thoughts_output,
                video_output,
                trace_file_output,
                history_file_output,
                stop_button,
                run_button,
            ],
        )

        stop_button.click(
            fn=stop_agent,
            inputs=[],
            outputs=[errors_output, stop_button, run_button],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Run the Gradio UI for the agent browser")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = Config()
    if os.path.exists(args.config):
        config = load_config_from_file(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    demo = create_ui(config)
    demo.launch()



if __name__ == "__main__":
    main()
