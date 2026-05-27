"""
Full end-to-end test for Meeting Intelligence Assistant.
Run this ON the JarvisLabs server (where the app is running).
It connects to localhost:7860.

Usage:
    pip install playwright
    playwright install chromium
    python test_local.py
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Force UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

from playwright.async_api import async_playwright, Page

APP_URL = "http://localhost:7860"
SCREENSHOTS_DIR = Path("/tmp/meeting_test_screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Sample files — generated via gTTS + reportlab on the Windows machine,
# then uploaded to JarvisLabs. Adjust paths if needed.
AUDIO_FILE = str(Path("sample_inputs/sample_meeting.mp3").resolve())
PDF_FILE   = str(Path("sample_inputs/sample_slides.pdf").resolve())


async def shot(page: Page, name: str) -> str:
    p = SCREENSHOTS_DIR / f"{name}.png"
    await page.screenshot(path=str(p), full_page=False)
    print(f"    [screenshot] {p}")
    return str(p)


async def wait_for_status_change(page: Page, timeout_s: int = 180) -> str:
    """Poll Status textarea until it changes from the placeholder."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            val = await page.locator("textarea[aria-label='Status']").input_value()
            if val and "Upload files" not in val:
                return val
        except Exception:
            pass
        await asyncio.sleep(3)
    return "TIMEOUT"


async def get_first_markdown(page: Page) -> str:
    """Return inner text of the first visible markdown/prose block."""
    for selector in [
        ".prose",
        "[data-testid='markdown']",
        ".svelte-1ed2p3z",
        ".output-markdown",
        ".gr-prose",
    ]:
        try:
            els = await page.locator(selector).all()
            for el in els:
                txt = await el.inner_text()
                if txt.strip():
                    return txt.strip()
        except Exception:
            pass
    return ""


async def run_tests():
    results: list[tuple[str, str, str]] = []

    def record(name: str, status: str, detail: str = ""):
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        results.append((name, status, detail))
        print(f"\n  {icon} {name}")
        if detail:
            print(f"         {detail[:200]}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()

        # ── Abort if app unreachable ──────────────────────────────
        print(f"\nConnecting to {APP_URL} ...")
        try:
            await page.goto(APP_URL, timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=20000)
            print("  Connected OK\n")
        except Exception as e:
            print(f"\n[BLOCKED] Cannot reach {APP_URL}: {e}")
            print("Make sure app.py is running on port 7860.")
            await browser.close()
            return

        # ==========================================================
        # TEST 1 — UI Sanity
        # ==========================================================
        print("=" * 55)
        print("TEST 1: UI Sanity Check")
        print("=" * 55)
        try:
            await shot(page, "01_initial_load")

            # Title
            try:
                header = await page.locator("h1").first.inner_text()
            except Exception:
                header = await page.title()
            record("1a_title",
                   "PASS" if "Meeting Intelligence" in header else "FAIL",
                   f"Header: '{header}'")

            # Status bar (GPU/CPU line)
            try:
                status_md = await page.locator(".header-status").first.inner_text()
            except Exception:
                status_md = ""
            record("1b_status_bar",
                   "PASS" if any(x in status_md for x in ["GPU", "CPU", "VRAM", "Indexed"]) else "FAIL",
                   f"'{status_md[:100]}'")

            # Refresh button
            await page.get_by_role("button", name="Refresh Status").click()
            await asyncio.sleep(1)
            record("1c_refresh_btn", "PASS", "Clicked without error")

            # All 3 tabs visible
            tabs_text = await page.locator(".tab-nav button, [role='tab']").all_inner_texts()
            has_tabs = (
                any("Upload" in t for t in tabs_text) and
                any("Ask" in t for t in tabs_text) and
                any("Summary" in t or "Meeting" in t for t in tabs_text)
            )
            record("1d_three_tabs",
                   "PASS" if has_tabs else "FAIL",
                   f"Tabs found: {tabs_text}")

        except Exception as e:
            record("1_ui_sanity", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 2 — Empty submission
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 2: Empty Submission (no files)")
        print("=" * 55)
        try:
            # Make sure we're on Tab 1
            upload_tab = page.get_by_role("tab").filter(has_text="Upload")
            await upload_tab.click()
            await asyncio.sleep(1)

            await page.get_by_role("button", name="Process Files").click()
            await asyncio.sleep(3)
            await shot(page, "02_empty_submit")

            status_val = await page.locator("textarea[aria-label='Status']").input_value()
            record("2_empty_submit",
                   "PASS" if any(x in status_val.lower() for x in ["at least one", "please upload", "no file"]) else "FAIL",
                   f"Status: '{status_val}'")
        except Exception as e:
            record("2_empty_submit", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 3 — Ask without processing first
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 3: Ask Questions — no data loaded")
        print("=" * 55)
        try:
            await page.get_by_role("tab").filter(has_text="Ask").click()
            await asyncio.sleep(1)

            q_box = page.locator("textarea").filter(
                has_text=""
            ).nth(0)
            # Find question textarea by placeholder
            q_box = page.get_by_placeholder("e.g. What was the final decision")
            await q_box.fill("What was the pricing decision?")
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(4)
            await shot(page, "03_ask_no_data")

            answer = await get_first_markdown(page)
            record("3_ask_no_data",
                   "PASS" if any(x in answer.lower() for x in ["no meeting data", "upload", "process", "first"]) else "FAIL",
                   f"Answer: '{answer[:200]}'")
        except Exception as e:
            record("3_ask_no_data", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 4 — Summary without processing first
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 4: Summary — no data loaded")
        print("=" * 55)
        try:
            await page.get_by_role("tab").filter(has_text="Summary").click()
            await asyncio.sleep(1)
            await page.get_by_role("button", name="Generate Summary").click()
            await asyncio.sleep(4)
            await shot(page, "04_summary_no_data")

            summary = await get_first_markdown(page)
            record("4_summary_no_data",
                   "PASS" if any(x in summary.lower() for x in ["no meeting", "upload", "process", "first"]) else "FAIL",
                   f"Output: '{summary[:200]}'")
        except Exception as e:
            record("4_summary_no_data", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 5 — Example question buttons
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 5: Example Question Buttons")
        print("=" * 55)
        try:
            await page.get_by_role("tab").filter(has_text="Ask").click()
            await asyncio.sleep(1)

            ex_btn = page.get_by_role("button", name="What was the final decision on the pricing change?")
            if await ex_btn.count() > 0:
                await ex_btn.first.click()
                await asyncio.sleep(1)
                q_val = await page.get_by_placeholder("e.g. What was the final decision").input_value()
                record("5_example_btns",
                       "PASS" if "pricing" in q_val.lower() else "FAIL",
                       f"Question box filled: '{q_val[:100]}'")
            else:
                record("5_example_btns", "FAIL", "Example button not found in DOM")
        except Exception as e:
            record("5_example_btns", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 6 — MAIN: Upload PDF + Audio and Process
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 6: Upload & Process (PDF + Audio)")
        print("=" * 55)

        # Check test files exist
        if not Path(PDF_FILE).exists():
            record("6_process_files", "FAIL", f"PDF not found: {PDF_FILE}")
        elif not Path(AUDIO_FILE).exists():
            record("6_process_files", "FAIL", f"Audio not found: {AUDIO_FILE}")
        else:
            try:
                await page.get_by_role("tab").filter(has_text="Upload").click()
                await asyncio.sleep(1)

                # Upload PDF (first file input)
                file_inputs = page.locator("input[type='file']")
                await file_inputs.nth(0).set_input_files(PDF_FILE)
                await asyncio.sleep(2)
                print("    Uploaded PDF")

                # Upload Audio (second file input)
                await file_inputs.nth(1).set_input_files(AUDIO_FILE)
                await asyncio.sleep(2)
                print("    Uploaded Audio")

                await shot(page, "06a_files_uploaded")

                # Disable diarization (no HF token in this env)
                cb = page.locator("input[type='checkbox']").first
                if await cb.is_checked():
                    await cb.click()
                    print("    Diarization disabled")

                # Process
                await page.get_by_role("button", name="Process Files").click()
                print("    Processing... (waiting up to 3 min for Whisper)")

                status_val = await wait_for_status_change(page, timeout_s=200)
                await shot(page, "06b_after_processing")

                is_success = "complete" in status_val.lower() or (
                    "chunk" in status_val.lower() or "indexed" in status_val.lower()
                )
                is_error = status_val.startswith("❌") or "error" in status_val.lower() or status_val == "TIMEOUT"

                record("6_process_files",
                       "PASS" if is_success and not is_error else "FAIL",
                       f"Status: '{status_val[:250]}'")

                if is_success and not is_error:
                    # 6b: Transcript preview
                    try:
                        txts = await page.locator("textarea").all()
                        transcript = ""
                        for t in txts:
                            ph = await t.get_attribute("placeholder") or ""
                            if "Transcript" in ph or "transcript" in ph:
                                transcript = await t.input_value()
                                break
                        record("6b_transcript",
                               "PASS" if len(transcript.strip()) > 20 else "FAIL",
                               f"Transcript ({len(transcript)} chars): '{transcript[:120]}'")
                    except Exception as te:
                        record("6b_transcript", "FAIL", str(te)[:100])

                    # 6c: Stats shown
                    try:
                        stats = await get_first_markdown(page)
                        record("6c_stats",
                               "PASS" if any(x in stats.lower() for x in ["slide", "chunk", "indexed", "frame"]) else "FAIL",
                               f"Stats: '{stats[:200]}'")
                    except Exception as se:
                        record("6c_stats", "FAIL", str(se)[:100])

                    # 6d: Errors box
                    try:
                        err_areas = await page.locator("textarea").all()
                        for ea in err_areas:
                            lbl = await ea.get_attribute("aria-label") or ""
                            if "Warning" in lbl or "Error" in lbl:
                                errs = await ea.input_value()
                                record("6d_warnings", "PASS", f"Warnings/Errors: '{errs[:150] or 'None'}'")
                                break
                    except Exception:
                        pass

            except Exception as e:
                record("6_process_files", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 7 — Ask Questions (3 targeted)
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 7: Ask Questions (3 targeted queries)")
        print("=" * 55)

        qa_cases = [
            ("7a_pricing",
             "What was the final decision on the pricing change?",
             ["49", "15%", "fifteen", "price", "pricing", "increase"]),
            ("7b_budget_slide",
             "Which slide was being discussed when the budget came up?",
             ["slide 9", "slide nine", "9", "budget", "two million", "2,000", "2 million"]),
            ("7c_timeline",
             "Who disagreed with the timeline and what did they propose instead?",
             ["amit", "nine month", "9 month", "disagree", "nine", "longer"]),
        ]

        for tid, question, keywords in qa_cases:
            try:
                await page.get_by_role("tab").filter(has_text="Ask").click()
                await asyncio.sleep(1)

                q_box = page.get_by_placeholder("e.g. What was the final decision")
                await q_box.clear()
                await q_box.fill(question)
                await page.get_by_role("button", name="Ask").click()
                print(f"    Asked '{question[:50]}...' — waiting for Qwen inference")
                await asyncio.sleep(20)
                await shot(page, tid)

                answer = await get_first_markdown(page)
                answer_lower = answer.lower()

                has_kw  = any(k.lower() in answer_lower for k in keywords)
                has_timing = any(x in answer_lower for x in ["retrieval:", "inference:", "total:"])

                record(tid,
                       "PASS" if has_kw else "FAIL",
                       f"Keywords hit: {has_kw} | Timing footer: {has_timing} | "
                       f"Answer[:200]: '{answer[:200]}'")
            except Exception as e:
                record(tid, "FAIL", str(e)[:200])

        # 7d: Top-K slider
        try:
            await page.get_by_role("tab").filter(has_text="Ask").click()
            await asyncio.sleep(1)
            slider = page.locator("input[type='range']").first
            await slider.fill("3")
            val = await slider.input_value()
            record("7d_topk_slider",
                   "PASS" if val in ["3", "3.0"] else "FAIL",
                   f"Slider value: {val}")
        except Exception as e:
            record("7d_topk_slider", "FAIL", str(e)[:100])

        # 7e: Include images checkbox
        try:
            img_cb = page.get_by_label("Include slide/frame images in query")
            is_checked = await img_cb.is_checked()
            await img_cb.click()
            after = await img_cb.is_checked()
            record("7e_images_checkbox",
                   "PASS" if after != is_checked else "FAIL",
                   f"Was: {is_checked}, Now: {after}")
        except Exception as e:
            record("7e_images_checkbox", "FAIL", str(e)[:100])

        # ==========================================================
        # TEST 8 — Meeting Summary
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 8: Meeting Summary Generation")
        print("=" * 55)
        try:
            await page.get_by_role("tab").filter(has_text="Summary").click()
            await asyncio.sleep(1)
            await page.get_by_role("button", name="Generate Summary").click()
            print("    Generating summary (waiting up to 90s)...")
            await asyncio.sleep(45)
            await shot(page, "08_summary")

            summary = await get_first_markdown(page)
            s_lower = summary.lower()

            has_actions   = "action" in s_lower
            has_decisions = "decision" in s_lower or "key" in s_lower
            has_priya     = "priya" in s_lower
            has_amit      = "amit" in s_lower
            is_long       = len(summary) > 100

            record("8_summary",
                   "PASS" if (has_actions and has_decisions and is_long) else "FAIL",
                   f"Actions:{has_actions} Decisions:{has_decisions} "
                   f"Priya:{has_priya} Amit:{has_amit} len:{len(summary)}")
        except Exception as e:
            record("8_summary", "FAIL", str(e)[:200])

        # ==========================================================
        # TEST 9 — Edge Cases
        # ==========================================================
        print("\n" + "=" * 55)
        print("TEST 9: Edge Cases")
        print("=" * 55)

        # 9a: Completely unrelated question
        try:
            await page.get_by_role("tab").filter(has_text="Ask").click()
            await asyncio.sleep(1)
            q_box = page.get_by_placeholder("e.g. What was the final decision")
            await q_box.clear()
            await q_box.fill("What is the capital of France?")
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(15)
            await shot(page, "09a_unrelated")
            answer = await get_first_markdown(page)
            record("9a_unrelated_question",
                   "PASS" if len(answer.strip()) > 10 else "FAIL",
                   f"Got response ({len(answer)} chars): '{answer[:200]}'")
        except Exception as e:
            record("9a_unrelated_question", "FAIL", str(e)[:200])

        # 9b: Empty question
        try:
            await page.get_by_role("tab").filter(has_text="Ask").click()
            await asyncio.sleep(1)
            q_box = page.get_by_placeholder("e.g. What was the final decision")
            await q_box.clear()
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(3)
            await shot(page, "09b_empty_q")
            answer = await get_first_markdown(page)
            record("9b_empty_question",
                   "PASS" if any(x in answer.lower() for x in ["please enter", "enter a question", "empty", "❌"]) else "FAIL",
                   f"Response: '{answer[:150]}'")
        except Exception as e:
            record("9b_empty_question", "FAIL", str(e)[:200])

        # 9c: Ask same question twice (idempotency)
        try:
            for i in range(2):
                await page.get_by_role("tab").filter(has_text="Ask").click()
                await asyncio.sleep(1)
                q_box = page.get_by_placeholder("e.g. What was the final decision")
                await q_box.clear()
                await q_box.fill("What is the pricing change?")
                await page.get_by_role("button", name="Ask").click()
                await asyncio.sleep(15)
            await shot(page, "09c_idempotent")
            answer = await get_first_markdown(page)
            record("9c_idempotent",
                   "PASS" if len(answer.strip()) > 10 else "FAIL",
                   f"Second answer ({len(answer)} chars): '{answer[:120]}'")
        except Exception as e:
            record("9c_idempotent", "FAIL", str(e)[:200])

        # 9d: Enter key submits question
        try:
            await page.get_by_role("tab").filter(has_text="Ask").click()
            await asyncio.sleep(1)
            q_box = page.get_by_placeholder("e.g. What was the final decision")
            await q_box.clear()
            await q_box.fill("What was discussed in the meeting?")
            await q_box.press("Enter")
            await asyncio.sleep(15)
            await shot(page, "09d_enter_key")
            answer = await get_first_markdown(page)
            record("9d_enter_submits",
                   "PASS" if len(answer.strip()) > 10 else "FAIL",
                   f"Response ({len(answer)} chars)")
        except Exception as e:
            record("9d_enter_submits", "FAIL", str(e)[:200])

        await browser.close()

    # ==============================================================
    # FINAL REPORT
    # ==============================================================
    passes = sum(1 for _, s, _ in results if s == "PASS")
    fails  = sum(1 for _, s, _ in results if s == "FAIL")
    total  = len(results)

    print("\n" + "=" * 55)
    print("FINAL TEST REPORT")
    print("=" * 55)
    for name, status, detail in results:
        icon = "[PASS]" if status == "PASS" else "[FAIL]"
        print(f"  {icon} {name:<38} {detail[:80]}")

    print(f"\n  Result: {passes}/{total} PASS  |  {fails} FAIL")
    print(f"  Screenshots: {SCREENSHOTS_DIR}")

    if fails == 0:
        print("\n  ALL TESTS PASSED")
    else:
        failed_tests = [n for n, s, _ in results if s == "FAIL"]
        print(f"\n  FAILED TESTS: {', '.join(failed_tests)}")

    return results


if __name__ == "__main__":
    asyncio.run(run_tests())
