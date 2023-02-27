# coding=utf-8
# Copyright 2021 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
# Copyright (c) 2023 Jim O'Regan for Språkbanken Tal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Datasets loader to create the Riksdag data"""
from pathlib import Path
from pydub import AudioSegment

import datasets
from datasets.tasks import AutomaticSpeechRecognition
from datasets.features import Audio

ALIGNMENTS = Path("alignments")
TMP = Path("/tmp")
parameters=["-ac", "1", "-acodec", "pcm_s16le", "-ar", "16000"]


class RDDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.1.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="speech", version=VERSION, description="Data for speech recognition"),
    ]

    def _info(self):
        features = datasets.Features(
            {
                "audio": datasets.Audio(sampling_rate=16_000),
                "text": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description="Riksdag speech data",
            features=features,
            supervised_keys=None,
            task_templates=[
                AutomaticSpeechRecognition(audio_column="audio", transcription_column="text")
            ],
        )

    def _split_generators(self, dl_manager):
       return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                },
            ),
        ]
    
    def _generate_examples(self, split):
        for file in ALIGNMENTS.glob("*"):
            segments = []
            with open(str(file)) as alignment:
                for line in alignment.readlines():
                    if line.startswith("FILE"):
                        continue
                    parts = line.split("\t")
                    if parts[3] == "MISALIGNED":
                        continue
                    vidid = parts[0]
                    temp_wav = f"/tmp/{vidid}.wav"
                    if Path(temp_wav).exists():
                        audio = AudioSegment.from_wav(temp_wav)
                    else:
                        video_file = Path("/sbtal/riksdag-video/") / f"{parts[0]}_480p.mp4"
                        if video_file.exists():
                            vid_as = AudioSegment.from_file(str(video_file), "mp4")
                            vid_as.export(temp_wav, format="wav", parameters=parameters)
                            audio = AudioSegment.from_wav(temp_wav)
                        else:
                            continue
                    start = int(float(parts[1]) * 1000)
                    end = int(float(parts[2]) * 1000)
                    text = parts[4]
                    yield vidid, {
                        "id": vidid,
                        "audio": {
                            "array": audio[start:end],
                            "sampling_rate": 16_000
                        },
                        "text": text
                    }
