
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from src.agent.custom_browser import CustomBrowser
from src.browser.custom_context import CustomBrowserContext
import os

class BrowserManager:
    def __init__(self, config):
        self.config = config
        self._browser = None
        self._context = None

    async def get_browser_and_context(self):
        if self._browser is None or self._context is None:
            await self._initialize()
        return self._browser, self._context

    async def _initialize(self):
        extra_chromium_args = [f"--window-size={self.config.window_w},{self.config.window_h}"]
        if self.config.use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        browser_class = CustomBrowser if self.config.agent_type == "custom" else Browser
        if self._browser is None:
            self._browser = browser_class(
                config=BrowserConfig(
                    headless=self.config.headless,
                    disable_security=self.config.disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if self._context is None:
            context_config_class = (
                CustomBrowserContext if self.config.agent_type == "custom" else BrowserContextConfig
            )
            self._context = await self._browser.new_context(
                config=context_config_class(
                    trace_path=self.config.save_trace_path if self.config.save_trace_path else None,
                    save_recording_path=self.config.save_recording_path
                    if self.config.save_recording_path
                    else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=self.config.window_w, height=self.config.window_h
                    ),
                )
            )

    async def close(self):
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
