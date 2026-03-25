import argparse
import asyncio
import multiprocessing
from pathlib import Path

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

from webrtc import HumanPlayer

WEB_ROOT = Path(__file__).resolve().parent / "web"

nerfreal = None
human_active = False
pcs = set()


def json_ok(data="ok"):
    return web.json_response({"code": 0, "data": data})


async def offer(request):
    global human_active

    params = await request.json()
    offer_sdp = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if human_active:
        print("single human is busy")
        return web.json_response({"code": 1, "message": "human is busy"}, status=409)

    human_active = True
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        global human_active

        print(f"Connection state is {pc.connectionState}")
        if pc.connectionState == "failed":
            await pc.close()
        if pc.connectionState in {"closed", "failed"}:
            pcs.discard(pc)
            human_active = False

    try:
        player = HumanPlayer(nerfreal)
        pc.addTrack(player.audio)
        pc.addTrack(player.video)

        await pc.setRemoteDescription(offer_sdp)
        await pc.setLocalDescription(await pc.createAnswer())

        return web.json_response(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        )
    except Exception:
        human_active = False
        pcs.discard(pc)
        await pc.close()
        raise


async def human(request):
    params = await request.json()
    text = params.get("text", "").strip()

    if not text:
        return web.json_response({"code": 1, "message": "missing text"}, status=400)

    nerfreal.pause_talk()
    nerfreal.put_msg_txt(text)
    return json_ok()


async def on_shutdown(app):
    global human_active

    await asyncio.gather(*(pc.close() for pc in pcs), return_exceptions=True)
    pcs.clear()
    human_active = False


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avatar_id", type=str, default="wav2lip_avatar1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("-l", type=int, default=10)
    parser.add_argument("-r", type=int, default=10)
    parser.add_argument("--listenport", type=int, default=8010)
    return parser


def parse_options():
    parser = create_arg_parser()
    opt, unknown_args = parser.parse_known_args()
    if unknown_args:
        print("未知参数:", " ".join(unknown_args))
    return opt

# 创建数字人对象
def create_avatar(opt):
    from lipreal import LipReal

    print(opt)
    return LipReal(opt)


def create_app():
    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_post("/offer", offer)
    app.router.add_post("/human", human)
    app.router.add_static("/", path=str(WEB_ROOT))
    return app


def run_server(app, opt):
    web.run_app(app, host="0.0.0.0", port=opt.listenport)


def main():
    global nerfreal

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    opt = parse_options()
    nerfreal = create_avatar(opt)
    run_server(create_app(), opt)


if __name__ == "__main__":
    main()
