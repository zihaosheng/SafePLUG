import os
import torch
import cv2
import decord
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from decord import VideoReader, cpu
from torchvision import transforms
from transformers import ProcessorMixin, BatchEncoding
from transformers.image_processing_utils import BatchFeature
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, ToTensor
from torchvision.transforms._transforms_video import NormalizeVideo, RandomCropVideo, RandomHorizontalFlipVideo, CenterCropVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample

decord.bridge.set_bridge('torch')

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

def make_list_of_images(x):
    if not isinstance(x, list):
        return [x]
    return x

def add_frame_id(video_frames, frame_id_list):
    video_frames_with_id = []
    for img, frame_id in zip(video_frames, frame_id_list):
        draw = ImageDraw.Draw(img)
        width, height = img.size
        font_size = int(min(width, height) * 0.08)  # 字体大小为图像短边的8%
        font = ImageFont.truetype("./assets/fonts/DejaVuSans.ttf", font_size)  # 这里需要把字体文件放到assets/fonts/目录下
        text = str(frame_id)
        # 获取文本尺寸
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = width - text_width - 20
        y = 20  # 距离顶部20像素
        draw.text((x, y), text, fill='red', font=font)
        video_frames_with_id.append(img)
    return video_frames

def get_video_transform(config):
    config = config.vision_config
    if config.video_decode_backend == 'pytorchvideo':
        transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(config.num_frames),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            ),
        )

    elif config.video_decode_backend == 'decord':

        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )

    elif config.video_decode_backend == 'opencv':
        transform = Compose(
            [
                # UniformTemporalSubsample(num_frames),
                Lambda(lambda x: x / 255.0),
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
    else:
        raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
    return transform


def load_and_transform_video(
        video_path,
        transform,
        video_decode_backend='opencv',
        clip_start_sec=0.0,
        clip_end_sec=None,
        num_frames=8,
):
    if isinstance(video_path, list):
        video_frames = [Image.open(v_path).convert('RGB') for v_path in video_path]
        if len(video_frames) < num_frames:
            video_frames = video_frames + [video_frames[-1]] * (num_frames - len(video_frames))
        frame_id_list = [int(x.split('/')[-1].split('.')[0]) for x in video_path]  # 这里需要把帧号加进去
        video_frames = add_frame_id(video_frames, frame_id_list)
        # # 保存视频帧到临时目录
        # video_name = video_path[0].split("/")[-3]
        # os.makedirs('./tmp/{}'.format(video_name), exist_ok=True)
        # for i, frame in enumerate(video_frames):
        #     frame.save(f'./tmp/{video_name}/frame_{i:03d}.jpg')
        to_tensor = transforms.ToTensor()  # 会自动把像素归一化到[0,1]
        video_frames = [to_tensor(frame) for frame in video_frames]  # (T, C, H, W)
        video_frames = torch.stack(video_frames, dim=1) # (C, T, H, W)
        
        video_transform = Compose(
            [
                NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                ShortSideScale(size=224),
                CenterCropVideo(224),
                RandomHorizontalFlipVideo(p=0.5),
            ]
        )
        video_outputs = video_transform(video_frames)
        return video_outputs # (C, T, H, W)
    else:
        if os.path.isdir(video_path):
            video_frames = [os.path.join(video_path, file) for file in os.listdir(video_path)]
            video_frames = sorted(video_frames, key=lambda x: int(x.split('/')[-1].split('.')[0]))
            duration = len(video_frames)
            if duration > num_frames:
                frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
            else:
                frame_id_list = np.arange(duration)
            video_frames = [video_frames[i] for i in frame_id_list]
            video_frames = [Image.open(frame).convert('RGB') for frame in video_frames]  # 在这里需要把帧号加进去
            video_frames = add_frame_id(video_frames, frame_id_list)
            # # 保存视频帧到临时目录
            # video_name = video_path.split("/")[-2]
            # os.makedirs('./tmp/{}'.format(video_name), exist_ok=True)
            # for i, frame in enumerate(video_frames):
            #     frame.save(f'./tmp/{video_name}/frame_{i:03d}.jpg')
            to_tensor = transforms.ToTensor()  # 会自动把像素归一化到[0,1]
            video_frames = [to_tensor(frame) for frame in video_frames]  # (T, C, H, W)
            video_frames = torch.stack(video_frames, dim=1) # (C, T, H, W)
            
            video_transform = Compose(
                [
                    NormalizeVideo(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD),
                    ShortSideScale(size=224),
                    CenterCropVideo(224),
                    RandomHorizontalFlipVideo(p=0.5),
                ]
            )
            video_outputs = video_transform(video_frames)
            return video_outputs # (C, T, H, W)

        elif os.path.isfile(video_path):
            if video_decode_backend == 'pytorchvideo':
                #  decord pyav
                video = EncodedVideo.from_path(video_path, decoder="decord", decode_audio=False)
                duration = video.duration
                start_sec = clip_start_sec  # secs
                end_sec = clip_end_sec if clip_end_sec is not None else duration  # secs
                video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)
                video_outputs = transform(video_data)

            elif video_decode_backend == 'decord':
                decord.bridge.set_bridge('torch')
                decord_vr = VideoReader(video_path, ctx=cpu(0))
                duration = len(decord_vr)
                frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)
                video_data = decord_vr.get_batch(frame_id_list)
                video_data = video_data.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
                video_outputs = transform(video_data)

            elif video_decode_backend == 'opencv':
                cv2_vr = cv2.VideoCapture(video_path)
                duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_id_list = np.linspace(0, duration-1, num_frames, dtype=int)

                video_data = []
                for frame_idx in frame_id_list:
                    cv2_vr.set(1, frame_idx)
                    _, frame = cv2_vr.read()
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_data.append(torch.from_numpy(frame).permute(2, 0, 1))
                cv2_vr.release()
                video_data = torch.stack(video_data, dim=1)
                video_outputs = transform(video_data)
            else:
                raise NameError('video_decode_backend should specify in (pytorchvideo, decord, opencv)')
        else:
            raise FileNotFoundError(f'video_path {video_path} not found')
    return video_outputs

class LanguageBindVideoProcessor(ProcessorMixin):
    attributes = []
    tokenizer_class = ("LanguageBindVideoTokenizer")

    def __init__(self, config, tokenizer=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.transform = get_video_transform(config)
        self.image_processor = load_and_transform_video
        self.tokenizer = tokenizer

    def __call__(self, images=None, text=None, context_length=77, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")

        if text is not None:
            encoding = self.tokenizer(text, max_length=context_length, padding='max_length',
                                      truncation=True, return_tensors=return_tensors, **kwargs)

        if images is not None:
            images = make_list_of_images(images)
            image_features = [self.image_processor(image, self.transform,  # TODO：看看num_frames、video_decode_backend是什么
                                                   video_decode_backend=self.config.vision_config.video_decode_backend,
                                                   num_frames=self.config.vision_config.num_frames) for image in images]
            image_features = torch.stack(image_features) # (B, C, T, H, W)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features
            return encoding
        elif text is not None:
            return encoding
        else:
            return {"pixel_values": image_features}

    def preprocess(self, images, return_tensors):
        return self.__call__(images=images, return_tensors=return_tensors)

    def batch_decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)

    def decode(self, skip_special_tokens=True, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, skip_special_tokens=skip_special_tokens, **kwargs)
