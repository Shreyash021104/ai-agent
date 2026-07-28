"""Record a captioned walkthrough of the agent chat UI: it plans and calls tools,
recovers from a deliberate tool failure, and (on a second run) recalls a memory
from the first. Prereqs: API running on :8095 with a clean workspace/memory, plus
playwright + chromium. Usage: python scripts/record_demo.py [out_dir]"""
from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/agent-demo"
URL = "http://127.0.0.1:8095"
os.makedirs(OUT, exist_ok=True)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1180, "height": 860}, device_scale_factor=2,
                            record_video_dir=OUT, record_video_size={"width": 1180, "height": 860})
        pg = ctx.new_page()

        def cap(text, hold=0.0):
            pg.evaluate("""(t)=>{let e=document.getElementById('__c');if(!e){e=document.createElement('div');e.id='__c';e.style.cssText='position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:99999;background:rgba(10,12,18,.95);color:#fff;padding:12px 22px;border-radius:999px;font:600 16px/1.25 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;box-shadow:0 10px 34px rgba(0,0,0,.5);border:1px solid #2a3446;';document.body.appendChild(e);}e.textContent=t;}""", text)
            if hold:
                time.sleep(hold)

        def run_task(text):
            pg.fill("#task", text)
            pg.click("#go")

        def wait_for(sel, timeout=25):
            end = time.time() + timeout
            while time.time() < end:
                if pg.query_selector(sel):
                    return True
                time.sleep(0.2)
            return False

        pg.goto(URL)
        time.sleep(1.0)
        cap("Loop — a hand-written tool-calling agent (keyless mock LLM)", 3.0)

        cap("Give it a task — it plans and calls tools one at a time", 1.2)
        run_task("Compute the sum of 1..100 and save it to a file")
        wait_for(".tc")
        time.sleep(1.5)
        wait_for(".te")                      # the deliberate tool failure
        cap("A tool call failed (file not found) — watch it recover", 3.0)
        wait_for(".final")
        cap("Done: computed via the code tool, recovered, saved to file", 3.5)
        time.sleep(1.0)

        cap("New run — now ask about the previous task", 1.5)
        run_task("What did I ask you to compute last time?")
        wait_for(".mem")
        cap("It recalls a memory from the earlier run (long-term memory)", 3.5)
        wait_for(".final")
        cap("Answered using memory from a past session", 3.5)

        ctx.close()
        b.close()

    video = next((f for f in os.listdir(OUT) if f.endswith(".webm")), None)
    print(os.path.join(OUT, video) if video else "no video")


if __name__ == "__main__":
    main()
