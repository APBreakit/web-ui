import logging
from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse

logger = logging.getLogger(__name__)

import gradio as gr

from src.i18n import i18n
from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot


# Global variables for persistence
_global_browser = None
_global_browser_context = None

# Create the global agent state instance
_global_agent_state = AgentState()

async def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            message,                                        # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
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
        tool_calling_method
):
    global _global_agent_state
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',                                         # final_result
            errors,                                     # errors
            '',                                         # model_actions
            '',                                         # model_thoughts
            None,                                       # latest_video
            None,                                       # history_file
            None,                                       # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )


async def initialize_browser_and_context(
    use_own_browser,
    headless,
    disable_security,
    window_w,
    window_h,
    save_trace_path,
    save_recording_path,
    agent_type,
):
    global _global_browser, _global_browser_context
    extra_chromium_args = [f"--window-size={window_w},{window_h}"]
    if use_own_browser:
        chrome_path = os.getenv("CHROME_PATH", None)
        if chrome_path == "":
            chrome_path = None
        chrome_user_data = os.getenv("CHROME_USER_DATA", None)
        if chrome_user_data:
            extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
    else:
        chrome_path = None

    browser_class = CustomBrowser if agent_type == "custom" else Browser
    if _global_browser is None:
        _global_browser = browser_class(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                chrome_instance_path=chrome_path,
                extra_chromium_args=extra_chromium_args,
            )
        )

    if _global_browser_context is None:
        context_config_class = (
            CustomBrowserContext if agent_type == "custom" else BrowserContextConfig
        )
        _global_browser_context = await _global_browser.new_context(
            config=context_config_class(
                trace_path=save_trace_path if save_trace_path else None,
                save_recording_path=save_recording_path
                if save_recording_path
                else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(
                    width=window_w, height=window_h
                ),
            )
        )


async def close_browser_and_context():
    global _global_browser, _global_browser_context
    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None


async def run_org_agent(
    llm,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    task,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method,
):
    try:
        global _global_agent_state
        _global_agent_state.clear_stop()

        await initialize_browser_and_context(
            use_own_browser,
            headless,
            disable_security,
            window_w,
            window_h,
            save_trace_path,
            save_recording_path,
            "org",
        )

        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser=_global_browser,
            browser_context=_global_browser_context,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
        )
        history = await agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            trace_file.get(".zip"),
            history_file,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return "", errors, "", "", None, None
    finally:
        if not keep_browser_open:
            await close_browser_and_context()


async def run_custom_agent(
    llm,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method,
):
    try:
        global _global_agent_state
        _global_agent_state.clear_stop()

        await initialize_browser_and_context(
            use_own_browser,
            headless,
            disable_security,
            window_w,
            window_h,
            save_trace_path,
            save_recording_path,
            "custom",
        )

        controller = CustomController()
        agent = CustomAgent(
            task=task,
            add_infos=add_infos,
            use_vision=use_vision,
            llm=llm,
            browser=_global_browser,
            browser_context=_global_browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            max_actions_per_step=max_actions_per_step,
            agent_state=_global_agent_state,
            tool_calling_method=tool_calling_method,
        )
        history = await agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            trace_file.get(".zip"),
            history_file,
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return "", errors, "", "", None, None
    finally:
        if not keep_browser_open:
            await close_browser_and_context()


async def run_with_stream(
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
    tool_calling_method
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        # Add HTML content at the start of the result array
                html_content = f"""
        <head>
            <meta name="description" content="A modern, animated, and SEO-optimized landing page for a construction company.">
        </head>
        <h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>
        """
        yield [html_content] + list(result)
    else:
        try:
            _global_agent_state.clear_stop()
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method
                )
            )

            # Stream screenshots while the agent is running
            while not agent_task.done():
                if _global_agent_state.is_stopped():
                    logger.info("Agent stopped, halting screenshot streaming.")
                    break
                screenshot_path = capture_screenshot(_global_browser_context)
                if screenshot_path:
                    html_content = f"""
                    <div style='width:{stream_vw}vw; height:{stream_vh}vh; display: flex; justify-content: center; align-items: center;'>
                        <img src='file://{screenshot_path}' style='width: 100%; height: 100%; object-fit: contain;'/>
                    </div>
                    """
                    yield [html_content, "", "", "", None, None, None, gr.update(interactive=True), gr.update(interactive=True)]
                await asyncio.sleep(1)

            # Get the final result from the agent task
            result = await agent_task
            yield [""] + list(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            errors = str(e) + "\n" + traceback.format_exc()
            yield ["", "", errors, "", "", None, None, None, gr.update(interactive=True), gr.update(interactive=True)]
   
def create_ui(config):
    with gr.Blocks(
        title="Agent Browser",
        theme=config.get("theme", "origin"),
        css=".gradio-container { max-width: 100% !important; }",
    ) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                # Left panel for configuration
                with gr.Accordion("General Settings", open=True):
                    with gr.Row():
                        agent_type = gr.Radio(
                            ["custom", "org"],
                            value=config.get("agent_type", "custom"),
                            label="Agent Type",
                        )
                        theme_dropdown = gr.Dropdown(
                            ["origin", "citrus", "default", "glass", "monochrome", "ocean", "soft"],
                            value=config.get("theme", "origin"),
                            label="Theme",
                        )
                with gr.Accordion("LLM Settings", open=True):
                    llm_provider = gr.Dropdown(
                        ["Ollama", "OpenAI", "Groq", "Anthropic", "Google", "Mistral"],
                        value=config.get("llm_provider", "Ollama"),
                        label="LLM Provider",
                    )
                    llm_model_name = gr.Dropdown(
                        update_model_dropdown(config.get("llm_provider", "Ollama")),
                        value=config.get("llm_model_name", "llama3"),
                        label="LLM Model",
                        interactive=True,
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=config.get("llm_temperature", 0.7),
                        label="LLM Temperature",
                    )
                    llm_base_url = gr.Textbox(
                        value=config.get("llm_base_url", "http://localhost:11434"),
                        label="LLM Base URL (Optional)",
                    )
                    llm_api_key = gr.Textbox(
                        value=config.get("llm_api_key", ""),
                        label="LLM API Key (Optional)",
                        type="password",
                    )

                with gr.Accordion("Browser Settings", open=True):
                    use_own_browser = gr.Checkbox(
                        value=config.get("use_own_browser", False),
                        label="Use own browser (via CHROME_PATH and CHROME_USER_DATA in .env)",
                    )
                    keep_browser_open = gr.Checkbox(
                        value=config.get("keep_browser_open", False),
                        label="Keep browser open after run",
                    )
                    headless = gr.Checkbox(
                        value=config.get("headless", False),
                        label="Headless mode",
                    )
                    disable_security = gr.Checkbox(
                        value=config.get("disable_security", False),
                        label="Disable security measures (use with caution)",
                    )
                    with gr.Row():
                        window_w = gr.Slider(
                            minimum=1024,
                            maximum=1920,
                            step=1,
                            value=config.get("window_w", 1280),
                            label="Window Width",
                        )
                        window_h = gr.Slider(
                            minimum=768,
                            maximum=1080,
                            step=1,
                            value=config.get("window_h", 800),
                            label="Window Height",
                        )

                with gr.Accordion("Agent Settings", open=True):
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        step=1,
                        value=config.get("max_steps", 15),
                        label="Max Steps",
                    )
                    use_vision = gr.Checkbox(
                        value=config.get("use_vision", False),
                        label="Use Vision",
                    )
                    max_actions_per_step = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=config.get("max_actions_per_step", 1),
                        label="Max actions per step",
                    )
                    tool_calling_method = gr.Radio(
                        ["tool_calling", "json"],
                        value=config.get("tool_calling_method", "tool_calling"),
                        label="Tool Calling Method",
                    )

                with gr.Accordion("Files Settings", open=True):
                    with gr.Row():
                        save_recording_path = gr.Textbox(
                            value=config.get("save_recording_path", "recordings"),
                            label="Save Recording Path",
                        )
                        enable_recording = gr.Checkbox(
                            value=config.get("enable_recording", False),
                            label="Enable Recording",
                        )
                    save_agent_history_path = gr.Textbox(
                        value=config.get("save_agent_history_path", "history"),
                        label="Save Agent History Path",
                    )
                    save_trace_path = gr.Textbox(
                        value=config.get("save_trace_path", "traces"),
                        label="Save Trace Path",
                    )

                with gr.Accordion("Configuration", open=True):
                    with gr.Row():
                        save_config_button = gr.Button("Save Config")
                        load_config_button = gr.Button("Load Config")
                    config_file_dropdown = gr.Dropdown(
                        choices=utils.get_config_files(),
                        label="Select Config File",
                        interactive=True,
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
            js="""
            (theme) => {
                if (theme) {
                    const url = new URL(window.location);
                    url.searchParams.set('__theme', theme);
                    window.location.href = url.href;
                }
            }
            """,
        )

        # Agent tab run button
        run_button.click(
            fn=run_with_stream,
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

        # Agent tab stop button
        stop_button.click(
            fn=stop_agent,
            inputs=[],
            outputs=[errors_output, stop_button, run_button],
        )

        # Configuration management
        ui_inputs = [
            agent_type,
            theme_dropdown,
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
            max_steps,
            use_vision,
            max_actions_per_step,
            tool_calling_method,
            save_recording_path,
            enable_recording,
            save_agent_history_path,
            save_trace_path,
            researcher_provider,
            researcher_model,
            researcher_base_url,
            researcher_api_key,
            researcher_search_depth,
            researcher_max_search_results,
            researcher_max_words,
            researcher_report_type,
            user_agent,
        ]

        save_config_button.click(
            fn=save_current_config,
            inputs=ui_inputs,
            outputs=[config_file_dropdown],
        )

        load_config_button.click(
            fn=load_config_from_file,
            inputs=[config_file_dropdown],
            outputs=ui_inputs,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Gradio UI for the agent browser")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    config = default_config
    if os.path.exists(args.config):
        config = load_config_from_file(args.config)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    demo = create_ui(config)
    demo.launch()
