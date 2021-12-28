from dataclasses import dataclass
from datetime import date, datetime
from glob import glob
from io import BytesIO
from subprocess import PIPE
from typing import Callable, Dict, Iterator, List, TypeVar
from PIL import Image
from pathlib import Path
import cv2
import gc
import joblib
import json
import numpy as np
import rawpy
import subprocess
import sys
import time


pdaf_scale = 10
focus_location_size = 120
pdaf_point_size = 85

histogram_threshold = 0.05


@dataclass
class FocusPoint:
    x: int
    y: int
    width: int
    height: int
    face_tracking: bool


@dataclass
class FocusData:
    image_width: int
    image_height: int
    focus_points: List[FocusPoint]
    face_tracking: bool


@dataclass
class ImageMetadata:
    source_file: str
    file_name: str
    create_date: date
    focus_data: FocusData


@dataclass
class Score:
    wide_mean: float
    focus_point_mean: float


@dataclass
class ScoredItem:
    metadata: ImageMetadata
    score: Score


@dataclass
class RatedItem:
    metadata: ImageMetadata
    score: Score
    rating: int


# https://github.com/musselwhizzle/Focus-Points/blob/master/focuspoints.lrdevplugin/SonyDelegates.lua
# https://github.com/SK-Hardwired/s_afv/blob/s-afv-python-27/afv.py
def extract_focus_data(item: Dict) -> FocusData:
    focus_point = item["FocusLocation"].split(" ")
    image_width = int(focus_point[0])
    image_height = int(focus_point[1])
    x = int(focus_point[2])
    y = int(focus_point[3])

    face_tracking = item.get("AFTracking", None) == "Face tracking"

    points = [
        FocusPoint(
            x=x,
            y=y,
            width=focus_location_size,
            height=focus_location_size,
            face_tracking=face_tracking,
        ),
    ]

    num_pdaf_points = int(item["FocalPlaneAFPointsUsed"])

    if num_pdaf_points > 0:
        pdaf_dimensions = item["FocalPlaneAFPointArea"].split(" ")
        pdaf_width = int(pdaf_dimensions[0])
        pdaf_height = int(pdaf_dimensions[1])
        pdaf_scaled_width = pdaf_width * pdaf_scale
        pdaf_scaled_height = pdaf_height * pdaf_scale
        pdaf_x_offset = (image_width - pdaf_scaled_width) / 2
        pdaf_y_offset = (image_height - pdaf_scaled_height) / 2

        for i in range(num_pdaf_points):
            pdaf_point = item["FocalPlaneAFPointLocation"+str(i+1)].split(" ")
            x = int(pdaf_point[0])
            y = int(pdaf_point[1])
            pdaf_x = pdaf_x_offset + (x * pdaf_scale)
            pdaf_y = pdaf_y_offset + (y * pdaf_scale)
            points.append(
                FocusPoint(
                    x=pdaf_x,
                    y=pdaf_y,
                    width=pdaf_point_size,
                    height=pdaf_point_size,
                    face_tracking=False,
                )
            )

    return FocusData(
        image_width=image_width,
        image_height=image_height,
        focus_points=points,
        face_tracking=face_tracking,
    )


def extract_thumbnail(metadata: ImageMetadata) -> np.ndarray:
    with rawpy.imread(metadata.source_file) as raw:
        # raises rawpy.LibRawNoThumbnailError if thumbnail missing
        # raises rawpy.LibRawUnsupportedThumbnailError if unsupported format
        thumb = raw.extract_thumb()

    return np.asarray(Image.open(BytesIO(thumb.data)))


def extract_rgb(metadata: ImageMetadata) -> np.ndarray:
    with rawpy.imread(metadata.source_file) as raw:
        rgb = raw.postprocess(
            output_bps=8,
            median_filter_passes=1,
            half_size=True, user_flip=0, use_camera_wb=True, no_auto_bright=False)

    return rgb


def get_histogram(metadata: ImageMetadata) -> np.ndarray:
    image = cv2.resize(extract_thumbnail(metadata), (200, 200))

    histograms = []
    for i in range(3):  # each channel
        histograms.append(
            cv2.calcHist([image], channels=[i],
                         mask=None, histSize=[256], ranges=[0, 256]))

    array = np.array(histograms)
    return array.reshape(array.shape[0] * array.shape[1], 1)


def pad_for_dft(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    optimal_image = np.zeros(
        (cv2.getOptimalDFTSize(h), cv2.getOptimalDFTSize(w)))
    optimal_image[:h, :w] = image
    return optimal_image


def high_pass_filter(metadata: ImageMetadata, size: int = 120) -> np.ndarray:
    image = extract_rgb(metadata)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.fastNlMeansDenoising(
        gray_image, h=10, templateWindowSize=7, searchWindowSize=21)
    original_h, original_w = gray_image.shape

    optimal_image = pad_for_dft(gray_image)
    h, w = optimal_image.shape
    cx = int(w/2)
    cy = int(h/2)

    f = np.fft.fft2(np.float32(optimal_image))
    fshift = np.fft.fftshift(f)

    # Mask low-frequencies (high-pass filter)
    fshift[cy-size:cy+size, cx-size:cx+size] = 0

    ifshift = np.fft.ifftshift(fshift)
    ifft = np.fft.ifft2(ifshift)
    ifft = ifft[0:original_h, 0:original_w]

    magnitude = 20*np.log(np.abs(ifft))
    return magnitude


def build_metadata(item: Dict, file: str) -> ImageMetadata:
    create_date = datetime.strptime(
        item["CreateDate"]+item["OffsetTime"], "%Y:%m:%d %H:%M:%S%z")
    return ImageMetadata(
        source_file=file,  # item["SourceFile"] is not correct in Windows
        file_name=item["FileName"],
        create_date=create_date,
        focus_data=extract_focus_data(item),
    )


def expand_paths(paths: List[str]) -> List[str]:
    files = []
    for path in paths:
        for file in glob(path):
            files.append(file)

    return sorted(files)


def load_metadata(paths: List[str]) -> Iterator[ImageMetadata]:
    for file in expand_paths(paths):
        proc = subprocess.run(["exiftool", "-json", file],
                              check=True, encoding="utf-8", stdout=PIPE)

        for item in json.loads(proc.stdout):
            yield build_metadata(item, file)


# https://exiftool.org/metafiles.html
def write_metadata(path: str, rating: int):
    raw_path = Path(path)
    raw_suffix = raw_path.suffix
    xmp_path = raw_path.with_suffix(".xmp")
    xmp_exiftool_tmp_path = raw_path.with_suffix(".xmp_exiftool_tmp")

    xmp_exiftool_tmp_path.unlink(missing_ok=True)

    if not xmp_path.is_file():
        # Create xmp sidecar first.
        subprocess.run(["exiftool",
                        "-ext", raw_suffix,
                        "-o", "%d%f.xmp",
                        "-r", raw_path],
                       check=True, encoding="utf-8")

    # Write rating to xmp sidecar.
    subprocess.run(["exiftool",
                    "-overwrite_original_in_place",
                    "-ext", raw_suffix,
                    "-rating={}".format(rating),
                    "-srcfile", "%d%f.xmp",
                    "-srcfile", "@",
                    raw_path],
                   check=True, encoding="utf-8")


T = TypeVar("T")


def group_by(data: Iterator[T], same_group: Callable[[T, T], bool]) -> Iterator[Iterator[T]]:
    first = next(data, None)
    if first == None:
        return

    group = [first]
    for item in data:
        prev_item = group[-1]
        if same_group(item, prev_item):
            group.append(item)
        else:
            yield iter(group)
            group = [item]

    yield iter(group)


def split_groups(groups: Iterator[Iterator[T]], grouping: Callable[[Iterator[T]], Iterator[Iterator[T]]]) -> Iterator[Iterator[T]]:
    for group in groups:
        for splited_group in grouping(iter(group)):
            yield splited_group


def similar_date(a: ImageMetadata, b: ImageMetadata, threshold: int = 1) -> bool:
    return abs((a.create_date - b.create_date).seconds) <= threshold


def similar_histogram(a: ImageMetadata, b: ImageMetadata) -> bool:
    ha = get_histogram(a)
    hb = get_histogram(b)
    # CV_COMP_BHATTACHARYYA: same=0.0
    return cv2.compareHist(ha, hb, 3) < histogram_threshold


def plot_focus_points(image: np.ndarray, metadata: ImageMetadata) -> np.ndarray:
    image = image.copy()

    h, w, _ = image.shape
    x_scale = w / metadata.focus_data.image_width
    y_scale = h / metadata.focus_data.image_height

    for point in metadata.focus_data.focus_points:
        x = point.x
        y = point.y
        pw = point.width / 2
        ph = point.height / 2
        color = (0, 255, 0) if point.face_tracking else (0, 255, 255)
        cv2.rectangle(image,
                      (int((x-pw)*x_scale), int((y-ph)*x_scale)),
                      (int((x+pw)*x_scale), int((y+ph)*y_scale)),
                      color=color, thickness=3)

    return image


# https://www.pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
def calc_mean(image: np.ndarray, x: int, y: int, w: int, h: int) -> float:
    image_height, image_width = image.shape
    x_min = max(int(x-w/2), 0)
    x_max = min(int(x+w/2), image_width-1)
    y_min = max(int(y-h/2), 0)
    y_max = min(int(y+h/2), image_height-1)

    return image[y_min:y_max, x_min:x_max].mean()


def process_image(metadata: ImageMetadata, group_focus_data: List[FocusData]) -> ScoredItem:
    image_width = metadata.focus_data.image_width
    image_height = metadata.focus_data.image_height

    filtered_image = high_pass_filter(metadata)
    filtered_height, filtered_width = filtered_image.shape
    x_scale = filtered_width / image_width
    y_scale = filtered_height / image_height

    focus_points = sum([f.focus_points for f in group_focus_data], [])

    point_mean = max([
        calc_mean(
            filtered_image,
            x=int(p.x*x_scale),
            y=int(p.y*y_scale),
            w=int(p.width*x_scale),
            h=int(p.height*y_scale),
        ) for p in focus_points
    ])

    wide_mean = calc_mean(
        filtered_image,
        x=int(sum([p.x for p in focus_points])/len(focus_points)*x_scale),
        y=int(sum([p.y for p in focus_points])/len(focus_points)*y_scale),
        w=int(filtered_width/8),
        h=int(filtered_height/8),
    )

    if False:
        thumb_image = cv2.cvtColor(
            extract_thumbnail(metadata), cv2.COLOR_BGR2RGB)
        thumb_height, thumb_width, _ = thumb_image.shape

        filtered_small_image = cv2.resize(
            filtered_image, dsize=(thumb_width, thumb_height))
        filtered_small_image = filtered_small_image.clip(
            0, 255).astype(np.uint8)
        filtered_small_image = cv2.cvtColor(
            filtered_small_image, cv2.COLOR_GRAY2RGB)

        cv2.imwrite(metadata.file_name+".thumb.jpg",
                    plot_focus_points(thumb_image, metadata))
        cv2.imwrite(metadata.file_name+".filtered.jpg",
                    plot_focus_points(filtered_small_image, metadata))

    return ScoredItem(
        metadata=metadata,
        score=Score(
            focus_point_mean=point_mean,
            wide_mean=wide_mean,
        ),
    )


def main():
    data = load_metadata(sys.argv[1:])

    groups = iter([data])
    groups = split_groups(groups, lambda d: group_by(d, similar_date))
    groups = split_groups(groups, lambda d: group_by(d, similar_histogram))

    for group in groups:
        start_time = time.perf_counter()

        group = list(group)
        print([item.source_file for item in group], file=sys.stderr)

        group_focus_data = [m.focus_data for m in group]

        scored_items: List[ScoredItem] = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(process_image)(metadata, group_focus_data) for metadata in group)

        max_mean = max([max(item.score.wide_mean, item.score.focus_point_mean)
                       for item in scored_items])

        rated: List[RatedItem] = []
        for item in scored_items:
            rating = 0
            mean = max(item.score.focus_point_mean,
                       item.score.wide_mean)
            if mean >= 10:
                if item.metadata.focus_data.face_tracking:
                    rating += 1
                if mean >= 20:
                    rating += 1
                if mean >= 40:
                    rating += 1
                if mean >= max_mean * 0.95:
                    rating += 1

            rated.append(
                RatedItem(
                    metadata=item.metadata,
                    score=item.score,
                    rating=rating,
                )
            )

        rated = sorted(rated,
                       key=lambda item: (item.rating, max(
                           item.score.focus_point_mean, item.score.wide_mean)),
                       reverse=True)

        for item in rated:
            print(item.metadata.source_file, item.metadata.focus_data, item.score, item.rating,
                  file=sys.stderr)

        if True:
            for item in rated:
                write_metadata(item.metadata.source_file,
                               rating=item.rating)

        print("%.1f sec" % (time.perf_counter() - start_time), file=sys.stderr)

        gc.collect()


if __name__ == '__main__':
    main()
