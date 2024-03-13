import whisper
import json
import re
from pathlib import Path
import torch
import fire
import ffmpeg
import dotenv
import os
from pyannote.audio import Pipeline
import uuid
from pydub import AudioSegment


def main(file_path: str):
    audio_path = prep_file(file_path)
    dz_path = make_diarization(audio_path)
    groups, group_paths = postprocess_dz(audio_path, dz_path)
    caption_paths = run_whisper_on_audio_files(group_paths)
    build_html(audio_path, groups, audio_path.stem)


def make_diarization(file_path: Path):
    access_token = get_access_token()
    pipeline = load_pipeline(access_token)

    dz = pipeline(file_path)

    name = f"dz_{uuid.uuid4()}"
    path = Path(f"data/tmp/{name}.txt")
    with open(str(path), "w") as text_file:
        text_file.write(str(dz))

    return path


def postprocess_dz(audio_path: Path, dz_path: Path) -> tuple[list, list[Path]]:
    groups = group_dz_segments(dz_path)
    group_paths = save_groups(audio_path, groups)
    return groups, group_paths


def save_groups(audio_path: Path, groups) -> list[Path]:
    """Save the audio part corresponding to each diarization group."""
    audio = AudioSegment.from_wav(audio_path)
    gidx = -1
    group_paths = []
    for g in groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        gidx += 1
        group_path = Path(f"data/tmp/{str(gidx)}.wav")
        audio[start:end].export(group_path, format="wav")
        group_paths.append(group_path)
    return group_paths


def group_dz_segments(file_path: Path):
    """Grouping the diarization segments according to the speaker."""
    dzs = open(file_path).read().splitlines()

    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []

        g.append(d)

        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
        end = millisec(end)
        if lastend > end:  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
        if g:
            groups.append(g)
    return groups


def run_whisper_on_audio_files(group_paths: list[Path]) -> list[Path]:
    model = load_whisper_model()
    text_paths = []
    for audio_path in group_paths:
        out_path = audio_path.with_suffix(".json")
        result = model.transcribe(audio=str(audio_path), language="en", word_timestamps=True)
        with out_path.open("w") as outfile:
            json.dump(result, outfile, indent=4)
        text_paths.append(out_path)
    return text_paths


def load_whisper_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("large", device=device)
    return model


def load_pipeline(access_token: str):
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=access_token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    return pipeline


def prep_file(file_path) -> Path:
    # write to a temp file with a uuid name
    name = f"audio_{uuid.uuid4()}"
    path = Path(f"data/tmp/{name}.wav")
    (ffmpeg.input(file_path).output(str(path), format="s16le", acodec="pcm_s16le", ac=1, ar="16k"))
    return path


def get_access_token() -> str:
    dotenv.load_dotenv()
    hf_access_token = os.getenv("HF_ACCESS_TOKEN")
    assert hf_access_token, "No access token found in .env"
    return hf_access_token


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


def build_html(original_audio_path: Path, groups, audio_title: str):
    speakers = {
        "SPEAKER_00": ("Speaker 1", "#e1ffc7", "darkgreen"),
        "SPEAKER_01": ("Speaker 2", "white", "darkorange"),
        "SPEAKER_02": ("Speaker 3", "#e1ffc7", "blue"),
        "SPEAKER_03": ("Speaker 4", "white", "red"),
    }
    def_boxclr = "white"
    def_spkrclr = "orange"

    preS = (
        '\n<!DOCTYPE html>\n<html lang="en">\n\n<head>\n\t<meta charset="UTF-8">\n\t<meta name="viewport" content="whtmlidth=device-width, initial-scale=1.0">\n\t<meta http-equiv="X-UA-Compatible" content="ie=edge">\n\t<title>'
        + audio_title
        + "</title>\n\t<style>\n\t\tbody {\n\t\t\tfont-family: sans-serif;\n\t\t\tfont-size: 14px;\n\t\t\tcolor: #111;\n\t\t\tpadding: 0 0 1em 0;\n\t\t\tbackground-color: #efe7dd;\n\t\t}\n\n\t\ttable {\n\t\t\tborder-spacing: 10px;\n\t\t}\n\n\t\tth {\n\t\t\ttext-align: left;\n\t\t}\n\n\t\t.lt {\n\t\t\tcolor: inherit;\n\t\t\ttext-decoration: inherit;\n\t\t}\n\n\t\t.l {\n\t\t\tcolor: #050;\n\t\t}\n\n\t\t.s {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.c {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t.e {\n\t\t\t/*background-color: white; Changing background color */\n\t\t\tborder-radius: 10px;\n\t\t\t/* Making border radius */\n\t\t\twidth: 50%;\n\t\t\t/* Making auto-sizable width */\n\t\t\tpadding: 0 0 0 0;\n\t\t\t/* Making space around letters */\n\t\t\tfont-size: 14px;\n\t\t\t/* Changing font size */\n\t\t\tmargin-bottom: 0;\n\t\t}\n\n\t\t.t {\n\t\t\tdisplay: inline-block;\n\t\t}\n\n\t\t#player-div {\n\t\t\tposition: sticky;\n\t\t\ttop: 20px;\n\t\t\tfloat: right;\n\t\t\twidth: 40%\n\t\t}\n\n\t\t#player {\n\t\t\taspect-ratio: 16 / 9;\n\t\t\twidth: 100%;\n\t\t\theight: auto;\n\t\t}\n\n\t\ta {\n\t\t\tdisplay: inline;\n\t\t}\n\t</style>"
    )
    preS += '\n\t<script>\n\twindow.onload = function () {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\tvar player;\n\t\t\tvar lastword = null;\n\n\t\t\t// So we can compare against new updates.\n\t\t\tvar lastTimeUpdate = "-1";\n\n\t\t\tsetInterval(function () {\n\t\t\t\t// currentTime is checked very frequently (1 millisecond),\n\t\t\t\t// but we only care about whole second changes.\n\t\t\t\tvar ts = (player.currentTime).toFixed(1).toString();\n\t\t\t\tts = (Math.round((player.currentTime) * 5) / 5).toFixed(1);\n\t\t\t\tts = ts.toString();\n\t\t\t\tconsole.log(ts);\n\t\t\t\tif (ts !== lastTimeUpdate) {\n\t\t\t\t\tlastTimeUpdate = ts;\n\n\t\t\t\t\t// Its now up to you to format the time.\n\t\t\t\t\tword = document.getElementById(ts)\n\t\t\t\t\tif (word) {\n\t\t\t\t\t\tif (lastword) {\n\t\t\t\t\t\t\tlastword.style.fontWeight = "normal";\n\t\t\t\t\t\t}\n\t\t\t\t\t\tlastword = word;\n\t\t\t\t\t\t//word.style.textDecoration = "underline";\n\t\t\t\t\t\tword.style.fontWeight = "bold";\n\n\t\t\t\t\t\tlet toggle = document.getElementById("autoscroll");\n\t\t\t\t\t\tif (toggle.checked) {\n\t\t\t\t\t\t\tlet position = word.offsetTop - 20;\n\t\t\t\t\t\t\twindow.scrollTo({\n\t\t\t\t\t\t\t\ttop: position,\n\t\t\t\t\t\t\t\tbehavior: "smooth"\n\t\t\t\t\t\t\t});\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\t\t\t\t}\n\t\t\t}, 0.1);\n\t\t}\n\n\t\tfunction jumptoTime(timepoint, id) {\n\t\t\tvar player = document.getElementById("audio_player");\n\t\t\thistory.pushState(null, null, "#" + id);\n\t\t\tplayer.pause();\n\t\t\tplayer.currentTime = timepoint;\n\t\t\tplayer.play();\n\t\t}\n\t\t</script>\n\t</head>'
    preS += (
        "\n\n<body>\n\t<h2>"
        + audio_title
        + f'</h2>\n\t<i>Click on a part of the transcription, to jump to its portion of audio, and get an anchor to it in the address\n\t\tbar<br><br></i>\n\t<div id="player-div">\n\t\t<div id="player">\n\t\t\t<audio controls="controls" id="audio_player">\n\t\t\t\t<source src={str(original_audio_path)} />\n\t\t\t</audio>\n\t\t</div>\n\t\t<div><label for="autoscroll">auto-scroll: </label>\n\t\t\t<input type="checkbox" id="autoscroll" checked>\n\t\t</div>\n\t</div>\n'
    )

    postS = "\t</body>\n</html>"

    html = list(preS)
    txt = list("")
    gidx = -1
    for g in groups:
        shift = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        shift = millisec(shift)
        shift = max(shift, 0)

        gidx += 1

        caption_path = Path(f"data/tmp/{gidx}.json")
        captions = json.load(open(caption_path))["segments"]

        if captions:
            speaker = g[0].split()[-1]
            boxclr = def_boxclr
            spkrclr = def_spkrclr
            if speaker in speakers:
                speaker, boxclr, spkrclr = speakers[speaker]

            html.append(f'<div class="e" style="background-color: {boxclr}">\n')
            html.append(
                '<p  style="margin:0;padding: 5px 10px 10px 10px;word-wrap:normal;white-space:normal;">\n'
            )
            html.append(
                f'<span style="color:{spkrclr};font-weight: bold;">{speaker}</span><br>\n\t\t\t\t'
            )

            for c in captions:
                start = shift + c["start"] * 1000.0
                start = start / 1000.0  # time resolution ot youtube is Second.
                end = (shift + c["end"] * 1000.0) / 1000.0
                txt.append(f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n')

                for i, w in enumerate(c["words"]):
                    if w == "":
                        continue
                    start = (shift + w["start"] * 1000.0) / 1000.0
                    # end = (shift + w['end']) / 1000.0   #time resolution ot youtube is Second.
                    html.append(
                        f'<a href="#{timeStr(start)}" id="{"{:.1f}".format(round(start*5)/5)}" class="lt" onclick="jumptoTime({int(start)}, this.id)">{w["word"]}</a><!--\n\t\t\t\t-->'
                    )
                # html.append('\n')
            html.append("</p>\n")
            html.append(f"</div>\n")

        html.append(postS)

        with open(f"capspeaker.txt", "w", encoding="utf-8") as file:
            s = "".join(txt)
            file.write(s)
            print("captions saved to capspeaker.txt:")
            print(s + "\n")

        with open(
            f"capspeaker.html", "w", encoding="utf-8"
        ) as file:  # TODO: proper html embed tag when video/audio from file
            s = "".join(html)
            file.write(s)
            print("captions saved to capspeaker.html:")
            print(s + "\n")


def timeStr(t):
    return "{0:02d}:{1:02d}:{2:06.2f}".format(round(t // 3600), round(t % 3600 // 60), t % 60)


if __name__ == "__main__":
    fire.Fire(main)
