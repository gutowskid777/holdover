"""Drive a whole call through the real server by typed text. python tests/talk.py "line 1" "line 2" ..."""
import asyncio, json, sys, websockets
async def main(lines):
    async with websockets.connect("ws://127.0.0.1:8770/ui") as ws:
        async def until_state():
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "state":
                    return m
                if m["type"] == "report":
                    print("  REPORT:", m["summary"]); [print("    ", l) for l in m["timeline"]]
        await ws.send(json.dumps({"type": "start"}))
        s = await until_state(); print("DISPATCH:", " | ".join(s["say"]))
        for l in lines:
            print(f"\nCALLER: {l}")
            await ws.send(json.dumps({"type": "text", "text": l}))
            s = await until_state()
            print(f"DISPATCH [{s['protocol']}.{s['node']} {s.get('think_ms')}ms]:", " | ".join(s["say"]))
            if s.get("capture"): print("  (camera capture requested:", s["capture"] + ")")
        await ws.send(json.dumps({"type": "report"}))
        await asyncio.wait_for(until_state(), 1) if False else None
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if m["type"] == "report":
                print("\nHANDOFF:", m["summary"]); [print("   ", l) for l in m["timeline"]]; break
asyncio.run(main(sys.argv[1:]))
