"""
Full end-to-end test for the live Meeting Intelligence Assistant app.
Uses Playwright to drive the Gradio UI at the live URL.
"""

import asyncio
import os
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, expect

import sys
sys.stdout.reconfigure(encoding="utf-8")

APP_URL = "https://d06d16d27a76d23194.gradio.live"
SCREENSHOTS_DIR = Path("test_screenshots")
SCREENSHOTS_DIR.mkdir(exist_ok=True)

AUDIO_FILE = str(Path("sample_inputs/sample_meeting.mp3").resolve())
PDF_FILE   = str(Path("sample_inputs/sample_slides.pdf").resolve())


async def shot(page: Page, name: str):
    p = SCREENSHOTS_DIR / f"{name}.png"
    await page.screenshot(path=str(p), full_page=False)
    print(f"  📸 Screenshot: {p}")
    return str(p)


async def wait_for_status(page: Page, timeout_s: int = 180) -> str:
    """Poll the Status textbox until it's non-empty and non-placeholder."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        val = await page.locator("textarea[aria-label='Status']").input_value()
        if val and "Upload files" not in val:
            return val
        await asyncio.sleep(2)
    return "TIMEOUT"


async def run_tests():
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await ctx.new_page()

        # ── Helper ────────────────────────────────────────────────
        def record(name, status, detail=""):
            icon = "✅" if status == "PASS" else "❌"
            results.append((name, status, detail))
            print(f"\n{icon} [{name}] {detail}")

        # ═══════════════════════════════════════════════════════════
        # TEST 1 – UI Sanity check
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 1: UI Sanity Check")
        print("="*60)
        try:
            await page.goto(APP_URL, timeout=60000)
            await page.wait_for_load_state("networkidle", timeout=20000)
            await shot(page, "01_initial_load")

            # Header
            header = await page.locator("h1, .gr-markdown h1").first.inner_text()
            has_title = "Meeting Intelligence" in header
            record("1a_title", "PASS" if has_title else "FAIL", f"Title: '{header}'")

            # GPU status bar
            status_md = await page.locator(".header-status").first.inner_text()
            has_gpu = "GPU" in status_md or "CPU" in status_md
            record("1b_status_bar", "PASS" if has_gpu else "FAIL", f"Status: '{status_md[:80]}'")

            # Refresh button
            await page.get_by_role("button", name="Refresh Status").click()
            await asyncio.sleep(1)
            record("1c_refresh_btn", "PASS", "Clicked without error")

            # Tabs visible
            tabs = await page.locator(".tab-nav button").all_inner_texts()
            has_all_tabs = all(
                any(t in tab for tab in tabs)
                for t in ["Upload", "Ask", "Summary"]
            )
            record("1d_tabs", "PASS" if has_all_tabs else "FAIL", f"Tabs: {tabs}")

        except Exception as e:
            record("1_ui_sanity", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 2 – Empty submission
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 2: Empty Submission")
        print("="*60)
        try:
            await page.get_by_role("button", name="Process Files").click()
            await asyncio.sleep(2)
            await shot(page, "02_empty_submit")
            status_val = await page.locator("textarea[aria-label='Status']").input_value()
            passed = "at least one file" in status_val.lower() or "please upload" in status_val.lower()
            record("2_empty_submit", "PASS" if passed else "FAIL", f"Status: '{status_val}'")
        except Exception as e:
            record("2_empty_submit", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 3 – Tab 2 ask without processing
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 3: Ask Without Processing First")
        print("="*60)
        try:
            await page.get_by_role("tab", name="Ask Questions").click()
            await asyncio.sleep(1)
            await page.locator("textarea[placeholder*='What was the final decision']").fill(
                "What was the pricing decision?"
            )
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(3)
            await shot(page, "03_ask_no_data")
            answer = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
            passed = "no meeting data" in answer.lower() or "upload" in answer.lower() or "process" in answer.lower()
            record("3_ask_no_data", "PASS" if passed else "FAIL", f"Answer: '{answer[:150]}'")
        except Exception as e:
            record("3_ask_no_data", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 4 – Tab 3 summary without processing
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 4: Summary Without Processing First")
        print("="*60)
        try:
            await page.get_by_role("tab", name="Meeting Summary").click()
            await asyncio.sleep(1)
            await page.get_by_role("button", name="Generate Summary").click()
            await asyncio.sleep(3)
            await shot(page, "04_summary_no_data")
            summary = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
            passed = "no meeting data" in summary.lower() or "upload" in summary.lower()
            record("4_summary_no_data", "PASS" if passed else "FAIL", f"Output: '{summary[:150]}'")
        except Exception as e:
            record("4_summary_no_data", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 5 – Example question buttons populate input
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 5: Example Question Buttons")
        print("="*60)
        try:
            await page.get_by_role("tab", name="Ask Questions").click()
            await asyncio.sleep(1)
            example_btns = await page.get_by_role("button", name="What was the final decision on the pricing change?").all()
            if example_btns:
                await example_btns[0].click()
                await asyncio.sleep(1)
                q_val = await page.locator("textarea[placeholder*='What was the final decision']").input_value()
                passed = "pricing" in q_val.lower()
                record("5_example_btns", "PASS" if passed else "FAIL", f"Question box: '{q_val[:100]}'")
            else:
                record("5_example_btns", "FAIL", "Example buttons not found")
        except Exception as e:
            record("5_example_btns", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 6 – MAIN: Upload PDF + Audio and Process
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 6: Main Upload & Process (PDF + Audio)")
        print("="*60)
        try:
            await page.get_by_role("tab", name="Upload & Process").click()
            await asyncio.sleep(1)

            # Upload PDF
            pdf_input = page.locator("input[type='file']").nth(0)
            await pdf_input.set_input_files(PDF_FILE)
            await asyncio.sleep(2)
            print("  Uploaded PDF")

            # Upload Audio
            audio_input = page.locator("input[type='file']").nth(1)
            await audio_input.set_input_files(AUDIO_FILE)
            await asyncio.sleep(2)
            print("  Uploaded Audio")

            await shot(page, "06a_files_uploaded")

            # Disable diarization (no HF token needed)
            diarize_cb = page.locator("input[type='checkbox']").first
            is_checked = await diarize_cb.is_checked()
            if is_checked:
                await diarize_cb.click()
                print("  Disabled diarization (no HF token)")

            # Click Process
            await page.get_by_role("button", name="Process Files").click()
            print("  Clicked Process Files — waiting for completion (up to 3 min)...")

            # Wait for processing
            status_val = await wait_for_status(page, timeout_s=180)
            await shot(page, "06b_after_processing")

            passed = "complete" in status_val.lower() or "processing" in status_val.lower()
            is_error = status_val.startswith("❌") or "error" in status_val.lower()
            if is_error:
                passed = False
            record("6_process_files", "PASS" if passed else "FAIL", f"Status: '{status_val[:200]}'")

            if passed:
                # Check transcript preview
                try:
                    transcript = await page.locator("textarea[placeholder*='Transcript']").input_value()
                    has_transcript = len(transcript.strip()) > 20
                    record("6b_transcript_preview", "PASS" if has_transcript else "FAIL",
                           f"Transcript preview ({len(transcript)} chars): '{transcript[:100]}'")
                except Exception as te:
                    record("6b_transcript_preview", "FAIL", str(te)[:100])

                # Check stats
                try:
                    stats = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
                    has_stats = "slide" in stats.lower() or "chunk" in stats.lower() or "indexed" in stats.lower()
                    record("6c_stats", "PASS" if has_stats else "FAIL", f"Stats: '{stats[:200]}'")
                except Exception as se:
                    record("6c_stats", "FAIL", str(se)[:100])

                # Check errors box
                try:
                    errors = await page.locator("textarea[aria-label*='Warning']").input_value()
                    record("6d_errors_box", "PASS", f"Warnings: '{errors[:150] or 'None'}'")
                except:
                    pass

        except Exception as e:
            record("6_process_files", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 7 – Ask Questions (3 targeted questions)
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 7: Ask Questions")
        print("="*60)

        qa_tests = [
            ("pricing_decision",
             "What was the final decision on the pricing change?",
             ["49", "15%", "price", "pricing", "increase", "dollar"]),
            ("budget_slide",
             "Which slide was being discussed when the budget came up?",
             ["slide 9", "slide nine", "9", "budget", "two million", "2,000"]),
            ("timeline_disagreement",
             "Who disagreed with the timeline and what did they propose instead?",
             ["amit", "nine month", "9 month", "disagree", "propose"]),
        ]

        for test_id, question, keywords in qa_tests:
            try:
                await page.get_by_role("tab", name="Ask Questions").click()
                await asyncio.sleep(1)

                q_box = page.locator("textarea[placeholder*='What was the final decision']")
                await q_box.clear()
                await q_box.fill(question)
                await asyncio.sleep(0.5)

                await page.get_by_role("button", name="Ask").click()
                print(f"  Asked: '{question}' — waiting for answer...")
                await asyncio.sleep(15)  # Qwen inference can take time
                await shot(page, f"07_{test_id}")

                answer_el = page.locator(".prose, [data-testid='markdown']").first
                answer = await answer_el.inner_text()
                answer_lower = answer.lower()

                has_keyword = any(kw.lower() in answer_lower for kw in keywords)
                has_timing = "retrieval:" in answer_lower or "inference:" in answer_lower
                passed = has_keyword
                record(f"7_{test_id}",
                       "PASS" if passed else "FAIL",
                       f"Keywords found: {has_keyword} | Timing footer: {has_timing} | Answer: '{answer[:250]}'")

            except Exception as e:
                record(f"7_{test_id}", "FAIL", str(e)[:200])

        # Test top-K slider
        try:
            await page.get_by_role("tab", name="Ask Questions").click()
            await asyncio.sleep(1)
            slider = page.locator("input[type='range']").first
            await slider.fill("3")
            slider_val = await slider.input_value()
            record("7d_topk_slider", "PASS" if slider_val in ["3", "3.0"] else "FAIL",
                   f"Slider set to: {slider_val}")
        except Exception as e:
            record("7d_topk_slider", "FAIL", str(e)[:100])

        # ═══════════════════════════════════════════════════════════
        # TEST 8 – Meeting Summary
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 8: Meeting Summary")
        print("="*60)
        try:
            await page.get_by_role("tab", name="Meeting Summary").click()
            await asyncio.sleep(1)
            await page.get_by_role("button", name="Generate Summary").click()
            print("  Generating summary — waiting (up to 90s)...")
            await asyncio.sleep(30)
            await shot(page, "08_summary")

            summary_el = page.locator(".prose, [data-testid='markdown']").first
            summary = await summary_el.inner_text()
            summary_lower = summary.lower()

            has_actions = "action" in summary_lower
            has_decisions = "decision" in summary_lower or "key" in summary_lower
            has_priya = "priya" in summary_lower
            has_amit = "amit" in summary_lower

            passed = has_actions and has_decisions and len(summary) > 100
            record("8_summary",
                   "PASS" if passed else "FAIL",
                   f"Actions: {has_actions} | Decisions: {has_decisions} | "
                   f"Priya: {has_priya} | Amit: {has_amit} | Length: {len(summary)}")
            if not passed:
                print(f"  Summary output: '{summary[:400]}'")
        except Exception as e:
            record("8_summary", "FAIL", str(e)[:200])

        # ═══════════════════════════════════════════════════════════
        # TEST 9 – Edge cases
        # ═══════════════════════════════════════════════════════════
        print("\n" + "="*60)
        print("TEST 9: Edge Cases")
        print("="*60)

        # 9a: Unrelated question
        try:
            await page.get_by_role("tab", name="Ask Questions").click()
            await asyncio.sleep(1)
            q_box = page.locator("textarea[placeholder*='What was the final decision']")
            await q_box.clear()
            await q_box.fill("What is the capital of France?")
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(12)
            await shot(page, "09a_unrelated_question")
            answer = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
            # App should either answer or gracefully say not in meeting context
            passed = len(answer.strip()) > 10
            record("9a_unrelated_question", "PASS" if passed else "FAIL",
                   f"Got response: '{answer[:200]}'")
        except Exception as e:
            record("9a_unrelated_question", "FAIL", str(e)[:200])

        # 9b: Empty question
        try:
            await page.get_by_role("tab", name="Ask Questions").click()
            await asyncio.sleep(1)
            q_box = page.locator("textarea[placeholder*='What was the final decision']")
            await q_box.clear()
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(3)
            await shot(page, "09b_empty_question")
            answer = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
            passed = "please enter" in answer.lower() or "❌" in answer
            record("9b_empty_question", "PASS" if passed else "FAIL",
                   f"Response: '{answer[:150]}'")
        except Exception as e:
            record("9b_empty_question", "FAIL", str(e)[:200])

        # 9c: Ask same question twice (idempotency)
        try:
            await page.get_by_role("tab", name="Ask Questions").click()
            await asyncio.sleep(1)
            q_box = page.locator("textarea[placeholder*='What was the final decision']")
            await q_box.clear()
            await q_box.fill("What is the pricing change?")
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(12)
            answer1 = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
            await q_box.clear()
            await q_box.fill("What is the pricing change?")
            await page.get_by_role("button", name="Ask").click()
            await asyncio.sleep(12)
            answer2 = await page.locator(".prose, [data-testid='markdown']").first.inner_text()
            passed = len(answer1) > 10 and len(answer2) > 10
            record("9c_idempotent_question", "PASS" if passed else "FAIL",
                   f"Both responses non-empty. Lengths: {len(answer1)}, {len(answer2)}")
        except Exception as e:
            record("9c_idempotent_question", "FAIL", str(e)[:200])

        await browser.close()

    # ═══════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("FULL TEST REPORT")
    print("="*60)
    passes = sum(1 for _, s, _ in results if s == "PASS")
    fails  = sum(1 for _, s, _ in results if s == "FAIL")
    total  = len(results)

    for name, status, detail in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"{icon} {name:<35} {detail[:100]}")

    print(f"\nResult: {passes}/{total} PASS  |  {fails} FAIL")
    print(f"Screenshots saved in: {SCREENSHOTS_DIR.resolve()}")
    return results


if __name__ == "__main__":
    asyncio.run(run_tests())
