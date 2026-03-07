from PIL import Image
import time
import yaml
import onnxruntime
import numpy as np
from typing import Tuple, List


def _softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class PARSEQ:
    def __init__(self,
                 model_path: str,
                 charlist: [str],
                 original_size: Tuple[int, int] = (384, 32),
                 device: str = "CPU",
                 tcy_min_line_width: int = 30,
                 tcy_det_margin_ratio: float = 0.1,
                 tcy_ocr_margin_ratio: float = 0.5,
                 tcy_min_components: int = 2,
                 tcy_max_aspect_ratio: float = 1.0,
                 tcy_seg_min_gap: int = 5,
                 tcy_ink_threshold_ratio: float = 0.10) -> None:
        self.model_path = model_path
        self.charlist = charlist

        self.device = device
        self.image_width, self.image_height = original_size

        self.tcy_min_line_width = tcy_min_line_width
        self.tcy_det_margin_ratio = tcy_det_margin_ratio
        self.tcy_ocr_margin_ratio = tcy_ocr_margin_ratio
        self.tcy_min_components = tcy_min_components
        self.tcy_max_aspect_ratio = tcy_max_aspect_ratio
        self.tcy_seg_min_gap = tcy_seg_min_gap
        self.tcy_ink_threshold_ratio = tcy_ink_threshold_ratio

        self.create_session()

    def create_session(self) -> None:
        opt_session = onnxruntime.SessionOptions()
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        #opt_session.enable_cpu_mem_arena = False
        #opt_session.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
        #opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        providers = ['CPUExecutionProvider']
        if self.device.casefold() == "cpu":
            opt_session.intra_op_num_threads = 1
            opt_session.inter_op_num_threads = 1
        elif self.device.casefold() == "cuda":
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        session = onnxruntime.InferenceSession(self.model_path,opt_session, providers=providers)
        self.session = session
        self.model_inputs = self.session.get_inputs()
        self.input_names = [self.model_inputs[i].name for i in range(len(self.model_inputs))]
        self.input_shape = self.model_inputs[0].shape
        self.model_output = self.session.get_outputs()
        self.output_names = [self.model_output[i].name for i in range(len(self.model_output))]
        self.input_height, self.input_width = self.input_shape[2:]

    def postprocess(self, outputs):
        predictions = np.squeeze(outputs).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_thresold, :]
        scores = scores[scores > self.conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)
        if pil_image.height>pil_image.width:
            pil_image =pil_image.transpose(Image.ROTATE_90)
        pil_resized = pil_image.resize((self.input_width, self.input_height))

        resized = np.array(pil_resized, dtype=np.float32)
        resized = resized[:,:,::-1]
        input_image = resized / 255.0
        input_image = 2.0*(input_image-0.5)
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _preprocess_no_rotation(self, img: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(img)
        pil_resized = pil_image.resize((self.input_width, self.input_height))
        resized = np.array(pil_resized, dtype=np.float32)
        resized = resized[:,:,::-1]
        input_image = resized / 255.0
        input_image = 2.0*(input_image-0.5)
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def _read_with_confidence(self, img: np.ndarray, rotate: bool = True) -> Tuple[str, List[float]]:
        if img is None:
            return "", []
        if rotate:
            input_tensor = self.preprocess(img)
        else:
            input_tensor = self._preprocess_no_rotation(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        probs = _softmax(outputs, axis=2)
        indices = np.argmax(probs, axis=2)[0]
        max_probs = np.max(probs, axis=2)[0]
        stop_idx = np.where(indices == 0)[0]
        end_pos = stop_idx[0] if stop_idx.size > 0 else len(indices)
        char_indices = indices[:end_pos].tolist()
        confidences = max_probs[:end_pos].tolist()
        text = "".join([self.charlist[i - 1] for i in char_indices])
        return text, confidences

    def _segment_blocks(self, img: np.ndarray) -> List[Tuple[int, int]]:
        min_gap = self.tcy_seg_min_gap
        if img.ndim == 3:
            gray = np.mean(img, axis=2).astype(np.uint8)
        else:
            gray = img
        threshold = int(np.mean(gray))
        binary = (gray < threshold).astype(np.int32)
        proj = np.sum(binary, axis=1)
        is_ink = proj > 0
        blocks: List[Tuple[int, int]] = []
        in_block = False
        start = 0
        for y in range(len(is_ink)):
            if is_ink[y] and not in_block:
                start = y
                in_block = True
            elif not is_ink[y] and in_block:
                blocks.append((start, y))
                in_block = False
        if in_block:
            blocks.append((start, len(is_ink)))
        merged: List[Tuple[int, int]] = []
        for b in blocks:
            if merged and b[0] - merged[-1][1] < min_gap:
                merged[-1] = (merged[-1][0], b[1])
            else:
                merged.append(b)
        return merged

    def _count_horizontal_components(self, segment: np.ndarray) -> int:
        if segment.ndim == 3:
            gray = np.mean(segment, axis=2).astype(np.uint8)
        else:
            gray = segment
        threshold = int(np.mean(gray))
        binary = (gray < threshold).astype(np.int32)
        col_sum = np.sum(binary, axis=0)
        if col_sum.max() == 0:
            return 0
        ink_threshold = col_sum.max() * self.tcy_ink_threshold_ratio
        is_ink = col_sum > ink_threshold
        components = 0
        in_component = False
        for v in is_ink:
            if v and not in_component:
                components += 1
                in_component = True
            elif not v:
                in_component = False
        return components

    def _detect_and_fix_tatechuyoko(self, img: np.ndarray) -> str:
        h, w = img.shape[:2]
        full_text, full_conf = self._read_with_confidence(img, rotate=True)
        if not full_text:
            return full_text
        blocks = self._segment_blocks(img)
        if not blocks or w < self.tcy_min_line_width:
            return full_text

        # Classify each block as tate-chuu-yoko candidate
        # Use small margin for detection to avoid including neighboring blocks
        tcy_flags: List[bool] = []
        for y_start, y_end in blocks:
            block_height = y_end - y_start
            det_margin = max(2, int(block_height * self.tcy_det_margin_ratio))
            y0 = max(0, y_start - det_margin)
            y1 = min(h, y_end + det_margin)
            block_img = img[y0:y1, :, :] if img.ndim == 3 else img[y0:y1, :]
            is_tcy = (self._count_horizontal_components(block_img) >= self.tcy_min_components
                       and block_height <= w * self.tcy_max_aspect_ratio)
            tcy_flags.append(is_tcy)

        if not any(tcy_flags):
            return full_text

        # Build block-by-block result
        block_parts: List[str] = []
        i = 0
        n = len(blocks)
        while i < n:
            if tcy_flags[i]:
                y_start, y_end = blocks[i]
                block_height = y_end - y_start
                ocr_margin = max(5, int(block_height * self.tcy_ocr_margin_ratio))
                y0 = max(0, y_start - ocr_margin)
                y1 = min(h, y_end + ocr_margin)
                block_img = img[y0:y1, :, :] if img.ndim == 3 else img[y0:y1, :]
                if block_img.ndim == 2:
                    block_img = np.stack([block_img] * 3, axis=-1)
                seg_text, _ = self._read_with_confidence(block_img, rotate=False)
                block_parts.append(seg_text)
                i += 1
            else:
                group_start = i
                while i < n and not tcy_flags[i]:
                    i += 1
                if group_start > 0 and tcy_flags[group_start - 1]:
                    crop_y0 = blocks[group_start - 1][1]
                else:
                    crop_y0 = blocks[group_start][0]
                if i < n and tcy_flags[i]:
                    crop_y1 = blocks[i][0]
                else:
                    crop_y1 = blocks[i - 1][1]
                group_img = img[crop_y0:crop_y1, :, :] if img.ndim == 3 else img[crop_y0:crop_y1, :]
                if group_img.shape[0] > 0 and group_img.shape[1] > 0:
                    group_text, _ = self._read_with_confidence(group_img, rotate=True)
                    block_parts.append(group_text)

        # If block-by-block result has more characters, it likely recovered
        # tate-chuu-yoko text that the full rotated OCR missed.
        block_text = "".join(block_parts)
        if len(block_text) > len(full_text):
            return block_text
        return full_text

    def read(self, img: np.ndarray) -> List:
        if img is None:
            return None
        h, w = img.shape[:2]
        if h > w:
            return self._detect_and_fix_tatechuyoko(img)
        input_tensor = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})[0]
        indices = np.argmax(outputs, axis=2)[0]
        stop_idx = np.where(indices == 0)[0]
        end_pos = stop_idx[0] if stop_idx.size > 0 else len(indices)
        resval = indices[:end_pos].tolist()
        resstr = "".join([self.charlist[i - 1] for i in resval])
        return resstr